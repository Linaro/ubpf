/*
 * Copyright 2015 Big Switch Networks, Inc
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <errno.h>
#include <assert.h>
#include "ubpf_int.h"

#if !defined(_countof)
#define _countof(array) (sizeof(array) / sizeof(array[0]))
#endif

/* Special values for target_pc in struct jump */
#define TARGET_PC_EXIT -1
#define TARGET_PC_DIV_BY_ZERO -2

#define BAD_OPCODE -1

struct jump {
    uint32_t offset_loc;
    uint32_t target_pc;
};

struct string_reference {
    uint32_t offset_loc;
    uint32_t string_id;
};

struct jit_state {
    uint8_t *buf;
    uint32_t offset;
    uint32_t size;
    uint32_t *pc_locs;
    uint32_t exit_loc;
    uint32_t div_by_zero_loc;
    uint32_t unwind_loc;
    struct jump *jumps;
    int num_jumps;
    struct string_reference* strings;
    int num_strings;
    uint32_t string_table_loc;
    uint32_t stack_size;
    uint32_t string_table_register_pointer; // Offset in buf that the string_table_register points to.
};

#define REGISTER_MAP_SIZE 11

enum Registers {R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, SP, RZ = 31};

// Callee saved registers - this must be a multiple of two because of how we save the stack later on.
static enum Registers callee_saved_registers[] = {R19, R20, R21, R22, R23, R24, R25, R26};
// Caller saved registers (and parameter registers)
static enum Registers caller_saved_registers[] = {R0, R1, R2, R3, R4};
// Temp register for immediate generation
static enum Registers temp_register = R24; 
// Temp register for division results
static enum Registers temp_div_register = R25;
// Register to hold string table base pointer
static enum Registers string_table_register = R26;

// Register assignments:
//   BPF        Arm64       Usage
//   r0         r5          Return value from calls (see note)
//   r1 - r5    r0 - r4     Function parameters, caller-saved
//   r6 - r10   r19 - r23   Callee-saved registers
//              r24         Temp - used for generating 32-bit immediates
//              r25         Temp - used for modulous calculations
//              r26         String table pointer.
//
// Note that the AArch64 ABI uses r0 both for function parameters and result.  We use r5 to hold
// the result during the function and do an extra final move at the end of the function to copy the
// result to the correct place.
static enum Registers register_map[REGISTER_MAP_SIZE] = {
    5, // result
    0, 1, 2, 3, 4, // parameters
    19, 20, 21, 22, 23, // callee-saved
};

/* Return the Arm64 register for the given eBPF register */
static enum Registers
map_register(int r)
{
    assert(r < REGISTER_MAP_SIZE);
    return register_map[r % REGISTER_MAP_SIZE];
}

static void emit_movewide_immediate(struct jit_state *state, bool sixty_four, enum Registers rd, uint64_t imm);
static void divmod(struct jit_state *state, uint16_t pc, uint8_t opcode, int rd, int rn, int rm);

static void
emit_bytes(struct jit_state *state, void *data, uint32_t len)
{
    assert(len <= state->size);
    assert(state->offset <= state->size - len);
    if ((state->offset + len) > state->size) {
        state->offset = state->size;
        return;
    }
    memcpy(state->buf + state->offset, data, len);
    state->offset += len;
}

static void emit_instruction(struct jit_state *state, uint32_t instr)
{
    assert(instr != BAD_OPCODE);
    emit_bytes(state, &instr, 4);
}

enum AddSubOpcode { AS_ADD = 0, AS_ADDS = 1, AS_SUB = 2, AS_SUBS = 3 };

/* Emit an add/sub immediate.  Evaluates rd = rn OP imm
 * sixty_four sets whether this is a 64- or 32-bit operation.  imm12 should either be:
 *  - in the range 0x00000000 - 0x00000fff inclsive; or
 *  - be representable as 0x00nnn000 (where n can be any value).
 */
static void
emit_addsub_immediate(struct jit_state *state, bool sixty_four, enum AddSubOpcode op, enum Registers rd, enum Registers rn, uint32_t imm12)
{
    bool sh = imm12 >= (1 << 12);
    assert(!sh || (imm12 & 0xfff) == 0);
    if (sh) { 
        imm12 >>= 12; 
    }
    uint32_t instr = (sixty_four << 31) | (op << 29) | (1 << 28) | (1 << 24) | (sh << 22) | (imm12 << 10) | (rn << 5) | (rd << 0);
    emit_bytes(state, &instr, 4);
}

static void
emit_addsub_register(struct jit_state *state, bool sixty_four, enum AddSubOpcode op, enum Registers rd, enum Registers rn, enum Registers rm)
{
    uint32_t instr = (sixty_four << 31) | (op << 29) | (1 << 27) | (3 << 24) | (rm << 16) | (rn << 5) | (rd << 0);
    emit_bytes(state, &instr, 4);
}

enum LoadStoreOpcode {
                            // sz    V   op
    LS_STRB   = 0x39000000, // 0011_1001_0000_0000_0000_0000_0000_0000
    LS_LDRB   = 0x39400000, // 0011_1001_0100_0000_0000_0000_0000_0000
    LS_LDRSBX = 0x39800000, // 0011_1001_1000_0000_0000_0000_0000_0000
    LS_LDRSBW = 0x39c00000, // 0011_1001_1100_0000_0000_0000_0000_0000
    LS_STRH   = 0x79000000, // 0111_1001_0000_0000_0000_0000_0000_0000
    LS_LDRH   = 0x79400000, // 0111_1001_0100_0000_0000_0000_0000_0000
    LS_LDRSHX = 0x79800000, // 0111_1001_1000_0000_0000_0000_0000_0000
    LS_LDRSHW = 0x79c00000, // 0111_1001_1100_0000_0000_0000_0000_0000
    LS_STRW   = 0xb9000000, // 1011_1001_0000_0000_0000_0000_0000_0000
    LS_LDRW   = 0xb9400000, // 1011_1001_0100_0000_0000_0000_0000_0000
    LS_LDRSW  = 0xb9800000, // 1011_1001_1000_0000_0000_0000_0000_0000
    LS_STRX   = 0xf9000000, // 1111_1001_0000_0000_0000_0000_0000_0000
    LS_LDRX   = 0xf9400000, // 1111_1001_0100_0000_0000_0000_0000_0000
    LS_PRFM   = 0xf9800000, // 1111_1001_1000_0000_0000_0000_0000_0000
};

static void
emit_loadstore_immediate(struct jit_state *state, enum LoadStoreOpcode op, enum Registers rt, enum Registers rn, uint32_t imm12)
{
    assert((imm12 & ~UINT32_C(0xfff)) == 0);
    uint32_t instr = op | (imm12 << 10) | (rn << 5) | rt;
    emit_bytes(state, &instr, 4);
}

enum LoadStorePairOpcode {
                             // op    V    L
    LSP_STPW   = 0x29000000, // 0010_1001_0000_0000_0000_0000_0000_0000
    LSP_LDPW   = 0x29400000, // 0010_1001_0100_0000_0000_0000_0000_0000
    LSP_LDPSW  = 0x69400000, // 0110_1001_0100_0000_0000_0000_0000_0000
    LSP_STPX   = 0xa9000000, // 1010_1001_0000_0000_0000_0000_0000_0000
    LSP_LDPX   = 0xa9400000, // 1010_1001_0100_0000_0000_0000_0000_0000
};

static void
emit_loadstorepair_immediate(struct jit_state *state, enum LoadStorePairOpcode op, enum Registers rt, enum Registers rt2, enum Registers rn, int32_t imm7)
{
    int32_t imm_div = ((op == LSP_STPX) || (op == LSP_LDPX)) ? 8 : 4;
    assert(imm7 % imm_div == 0);
    imm7 /= imm_div;
    uint32_t instr = op | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt;
    emit_bytes(state, &instr, 4);
}

enum  LogicalOpcode {
                           //  op         N
    LOG_AND  = 0x00000000, // 0000_0000_0000_0000_0000_0000_0000_0000
    LOG_BIC  = 0x00200000, // 0000_0000_0010_0000_0000_0000_0000_0000
    LOG_ORR  = 0x20000000, // 0010_0000_0000_0000_0000_0000_0000_0000
    LOG_ORN  = 0x20200000, // 0010_0000_0010_0000_0000_0000_0000_0000
    LOG_EOR  = 0x40000000, // 0100_0000_0000_0000_0000_0000_0000_0000
    LOG_EON  = 0x40200000, // 0100_0000_0010_0000_0000_0000_0000_0000
    LOG_ANDS = 0x60000000, // 0110_0000_0000_0000_0000_0000_0000_0000
    LOG_BICS = 0x60200000, // 0110_0000_0010_0000_0000_0000_0000_0000
};

static void
emit_logical_register(struct jit_state *state, bool sixty_four, enum LogicalOpcode op, enum Registers rd, enum Registers rn, enum Registers rm)
{
    uint32_t instr = (sixty_four << 31) | op | (1 << 27) | (1 << 25) | (rm << 16) | (rn << 5) | rd;
    emit_bytes(state, &instr, 4);
}

enum UnconditionalBranchOpcode {
                         //         opc-|op2--|op3----|        op4|
    BR_BR  = 0xd61f0000, // 1101_0110_0001_1111_0000_0000_0000_0000
    BR_BLR = 0xd63f0000, // 1101_0110_0011_1111_0000_0000_0000_0000
    BR_RET = 0xd65f0000, // 1101_0110_0101_1111_0000_0000_0000_0000
};

static void
emit_unconditonalbranch_register(struct jit_state *state, enum UnconditionalBranchOpcode op, enum Registers rn)
{
    emit_instruction(state, op | (rn << 5));
}

static void emit_call(struct jit_state *state, uintptr_t func) {
    emit_movewide_immediate(state, true, temp_register, func);
    emit_unconditonalbranch_register(state, BR_BLR, temp_register);

    /* On exit need to move result from r0 to whichever register we've mapped EBPF r0 to.  */
    enum Registers dest = map_register(0);
    if (dest != R0) {
        emit_logical_register(state, true, LOG_ORR, dest, RZ, R0);
    }
}


enum UnconditionalBranchImmediateOpcode {
                         // O
    UBR_B =  0x14000000, // 0001_0100_0000_0000_0000_0000_0000_0000
    UBR_BL = 0x94000000, // 1001_0100_0000_0000_0000_0000_0000_0000
};

static void
note_jump(struct jit_state *state, uint32_t target_pc)
{
    if (state->num_jumps == UBPF_MAX_INSTS) {
        return;
    }
    struct jump *jump = &state->jumps[state->num_jumps++];
    jump->offset_loc = state->offset;
    jump->target_pc = target_pc;
}

static void
emit_unconditionalbranch_immediate(struct jit_state *state, enum UnconditionalBranchImmediateOpcode op, int32_t target_pc)
{
    note_jump(state, target_pc);
    emit_instruction(state, op);
}

enum Condition {
    COND_EQ, COND_NE, COND_CS, COND_CC, COND_MI, COND_PL, COND_VS, COND_VC, COND_HI, COND_LS, COND_GE, COND_LT, COND_GT, COND_LE, COND_AL, COND_NV,
    COND_HS = COND_CS, COND_LO = COND_CC
};

enum ConditionalBranchImmediateOpcode {
    BR_Bcond = 0x54000000
};

static void
emit_conditionalbranch_immediate(struct jit_state *state, enum Condition cond, uint32_t target_pc)
{
    note_jump(state, target_pc);
    emit_instruction(state, BR_Bcond | (0 << 5) | cond);
}

enum CompareBranchOpcode {
                          //          o
    CBR_CBZ  = 0x34000000, // 0011_0100_0000_0000_0000_0000_0000_0000
    CBR_CBNZ = 0x35000000, // 0011_0101_0000_0000_0000_0000_0000_0000
};

#if 0
static void
emit_comparebranch_immediate(struct jit_state *state, bool sixty_four, enum CompareBranchOpcode op, enum Registers rt, uint32_t target_pc)
{
    note_jump(state, target_pc);
    emit_instruction(state, (sixty_four << 31) | op | rt);
}
#endif

enum DP1Opcode {
                            //   S          op2--|op-----|
    DP1_REV16 = 0x5ac00400, // 0101_1010_1100_0000_0000_0100_0000_0000
    DP1_REV32 = 0x5ac00800, // 0101_1010_1100_0000_0000_1000_0000_0000
    DP1_REV64 = 0xdac00c00, // 0101_1010_1100_0000_0000_1100_0000_0000
};

static void
emit_dataprocessing_onesource(struct jit_state *state, bool sixty_four, enum DP1Opcode op, enum Registers rd, enum Registers rn)
{
    uint32_t instr = (sixty_four << 31) | op | (rn << 5) | rd;
    emit_bytes(state, &instr, 4);
}

enum DP2Opcode {
                           //   S                 opcode|
    DP2_UDIV = 0x1ac00800, // 0001_1010_1100_0000_0000_1000_0000_0000
    DP2_SDIV = 0x1ac00c00, // 0001_1010_1100_0000_0000_1100_0000_0000
    DP2_LSLV = 0x1ac02000, // 0001_1010_1100_0000_0010_0000_0000_0000
    DP2_LSRV = 0x1ac02400, // 0001_1010_1100_0000_0010_0100_0000_0000
    DP2_ASRV = 0x1ac02800, // 0001_1010_1100_0000_0010_1000_0000_0000
    DP2_RORV = 0x1ac02800, // 0001_1010_1100_0000_0010_1100_0000_0000
};

static void
emit_dataprocessing_twosource(struct jit_state *state, bool sixty_four, enum DP2Opcode op, enum Registers rd, enum Registers rn, enum Registers rm)
{
    uint32_t instr = (sixty_four << 31) | op | (rm << 16) | (rn << 5) | rd;
    emit_bytes(state, &instr, 4);
}

enum DP3Opcode {
                           //  54       31|       0
    DP3_MADD = 0x1b000000, // 0001_1011_0000_0000_0000_0000_0000_0000
    DP3_MSUB = 0x1b008000, // 0001_1011_0000_0000_1000_0000_0000_0000
};

static void
emit_dataprocessing_threesource(struct jit_state *state, bool sixty_four, enum DP3Opcode op, enum Registers rd, enum Registers rn, enum Registers rm, enum Registers ra)
{
    uint32_t instr = (sixty_four << 31) | op | (rm << 16) | (ra << 10) | (rn << 5) | rd;
    emit_bytes(state, &instr, 4);
}



enum MoveWideOpcode {
                          //  op
    MW_MOVN = 0x12800000, // 0001_0010_1000_0000_0000_0000_0000_0000
    MW_MOVZ = 0x52800000, // 0101_0010_1000_0000_0000_0000_0000_0000
    MW_MOVK = 0x72800000, // 0111_0010_1000_0000_0000_0000_0000_0000
};

static void
emit_movewide_immediate(struct jit_state *state, bool sixty_four, enum Registers rd, uint64_t imm)
{
    unsigned count0000 = 0;
    unsigned countffff = 0;
    for (unsigned i = 0; i < (sixty_four ? 64 : 32); i += 16) {
        if ((imm & (0xffff << i)) == (0xffff << i)) {
            ++countffff;
        }
        if ((imm & (0xffff << i)) == 0) {
            ++count0000;
        }
    }

    enum MoveWideOpcode op = (count0000 >= countffff) ? MW_MOVZ : MW_MOVN;
    bool invert = (count0000 < countffff);
    uint64_t skip_pattern = (count0000 >= countffff) ? 0 : 0xffff;
    for (unsigned i = 0; i < (sixty_four ? 4 : 2); ++i) {
        uint64_t imm16 = (imm >> (i * 16)) & 0xffff;
        if (imm16 != skip_pattern) {
            if (invert) {
                imm16 = ~imm16;
                imm16 &= 0xffff;
            }
            emit_instruction(state, (sixty_four << 31) | op | (i << 21) | (imm16 << 5) | rd);
            op = MW_MOVK;
            invert = false;
        }
    }

    /* Tidy up for the case imm = 0 or imm == -1.  */
    if (op != MW_MOVK) {
        emit_instruction(state, (sixty_four << 31) | op | (0 << 21) | (0 << 5) | rd);
    }
}

static void update_adr_immediate(struct jit_state *state, uint32_t offset, int64_t imm21)
{
    assert((imm21 >> 21) == 0 || (imm21 >> 21) == INT64_C(-1));

    uint32_t instr;
    memcpy(&instr, state->buf + offset, sizeof(uint32_t));
    instr |= (imm21 & 3) << 29;
    instr |= ((imm21 >> 2) & 0x3ffff) << 5;
    memcpy(state->buf + offset, &instr, sizeof(uint32_t));
}

static void update_branch_immediate(struct jit_state *state, uint32_t offset, int32_t imm)
{
    assert((imm & 3) == 0);
    uint32_t instr;
    imm >>= 2;
    memcpy(&instr, state->buf + offset, sizeof(uint32_t));
    if ((instr & 0xfe000000) == 0x54000000
        || (instr & 0x7e000000) == 0x34000000) {
        /* Conditional branch immediate.  */
        /* Compare and branch immediate.  */
        assert((imm >> 19) == INT64_C(-1) || (imm >> 19) == 0);
        instr |= (imm & 0x7ffff) << 5;
    }
    else if ((instr & 0x7c000000) == 0x14000000) {
        /* Unconditional branch immediate.  */
        assert((imm >> 26) == INT64_C(-1) || (imm >> 26) == 0);
        instr |= (imm & 0x03ffffff) << 0;
    }
    else {
        assert(false);
        instr = BAD_OPCODE;
    }
    memcpy(state->buf + offset, &instr, sizeof(uint32_t));
}


/* Generate the function prologue.
 *
 * We set the stack to look like:
 *   SP on entry
 *   ubpf_stack_size bytes of UBPF stack
 *   Callee saved registers
 *   Frame <- SP.
 */
static void
emit_function_prologue(struct jit_state *state, size_t ubpf_stack_size)
{
    uint32_t register_space =  _countof(callee_saved_registers) * 8 + 2 * 8;
    state->stack_size = (ubpf_stack_size + register_space + 15) & ~15U;
    emit_addsub_immediate(state, true, AS_SUB, SP, SP, state->stack_size);
    
    /* Set up frame */
    emit_loadstorepair_immediate(state, LSP_STPX, R29, R30, SP, 0);
    emit_addsub_immediate(state, true, AS_ADD, R29, SP, 0);

    /* Save callee saved registers */
    unsigned i;
    for (i = 0; i < _countof(callee_saved_registers); i += 2)
    {
        emit_loadstorepair_immediate(state, LSP_STPX, callee_saved_registers[i], callee_saved_registers[i + 1], SP, (i + 2) * 8);
    }

    /* Setup UBPF frame pointer. */
    emit_addsub_immediate(state, true, AS_ADD, map_register(10), SP, register_space);

    /* Setup string table pool pointer. */
    state->string_table_register_pointer = state->offset;
    // ADR string_table_register, #0
    emit_instruction(state, (0 << 29) | (1 << 28) | (0 << 5) | string_table_register);
}

static void
emit_string_load(struct jit_state *state, enum Registers dst, int string_id)
{
    if (state->num_strings == UBPF_MAX_INSTS) {
        return;
    }

    state->strings[state->num_strings].offset_loc = state->offset;
    state->strings[state->num_strings].string_id = string_id;
    emit_addsub_immediate(state, true, AS_ADD, dst, string_table_register, 0);
    state->num_strings++;
}

static void
emit_function_epilogue(struct jit_state *state)
{
    state->exit_loc = state->offset;

    /* Move register 0 into R0 */
    if (map_register(0) != R0) {
        emit_logical_register(state, true, LOG_ORR, R0, RZ, map_register(0));
    }

    /* Restore callee-saved registers).  */
    size_t i;
    for (i = 0; i < _countof(callee_saved_registers); i += 2) {
        emit_loadstorepair_immediate(state, LSP_LDPX, callee_saved_registers[i], callee_saved_registers[i + 1], SP, (i + 2) * 8);
    }
    emit_loadstorepair_immediate(state, LSP_LDPX, R29, R30, SP, 0);
    emit_addsub_immediate(state, true, AS_ADD, SP, SP, state->stack_size);
    emit_unconditonalbranch_register(state, BR_RET, R30);
}

static int
is_alu_imm_op(struct ebpf_inst const * inst)
{
    int class = inst->opcode & EBPF_CLS_MASK;
    int is_imm = (inst->opcode & EBPF_SRC_REG) == EBPF_SRC_IMM;
    int is_endian = (inst->opcode & EBPF_ALU_OP_MASK) == 0xd0;
    return is_imm && (class == EBPF_CLS_ALU || class == EBPF_CLS_ALU64) && !is_endian;
}

static int
is_alu64_op(struct ebpf_inst const * inst)
{
    return (inst->opcode & EBPF_CLS_MASK) == EBPF_CLS_ALU64;
}

static enum AddSubOpcode
to_addsub_opcode(int opcode) 
{
    switch (opcode)
    {
    case EBPF_OP_ADD_IMM:
    case EBPF_OP_ADD_REG:
    case EBPF_OP_ADD64_IMM:
    case EBPF_OP_ADD64_REG:
        return AS_ADD;
    case EBPF_OP_SUB_IMM:
    case EBPF_OP_SUB_REG:
    case EBPF_OP_SUB64_IMM:
    case EBPF_OP_SUB64_REG:
        return AS_SUB;
    default:
        assert(false);
        return (enum AddSubOpcode)BAD_OPCODE;
    }
}

static enum LogicalOpcode
to_logical_opcode(int opcode)
{
    switch (opcode)
    {
    case EBPF_OP_OR_IMM:
    case EBPF_OP_OR_REG:
    case EBPF_OP_OR64_IMM:
    case EBPF_OP_OR64_REG:
        return LOG_ORR;
    case EBPF_OP_AND_IMM:
    case EBPF_OP_AND_REG:
    case EBPF_OP_AND64_IMM:
    case EBPF_OP_AND64_REG:
        return LOG_AND;
    case EBPF_OP_XOR_IMM:
    case EBPF_OP_XOR_REG:
    case EBPF_OP_XOR64_IMM:
    case EBPF_OP_XOR64_REG:
        return LOG_EOR;
    default:
        assert(false);
        return (enum LogicalOpcode)BAD_OPCODE;
    }
}

static enum DP1Opcode
to_dp1_opcode(int opcode, uint32_t imm)
{
    switch (opcode)
    {
    case EBPF_OP_BE:
    case EBPF_OP_LE:
        switch (imm) {
        case 16:
            return DP1_REV16;
        case 32:
            return DP1_REV32;
        case 64:
            return DP1_REV64;
        default:
            assert(false);
            return 0;
        }
        break;
    default:
        assert(false);
        return (enum DP1Opcode)BAD_OPCODE;
    }
}

static enum DP2Opcode
to_dp2_opcode(int opcode) 
{
    switch (opcode)
    {
    case EBPF_OP_LSH_IMM:
    case EBPF_OP_LSH_REG:
    case EBPF_OP_LSH64_IMM:
    case EBPF_OP_LSH64_REG:
        return DP2_LSLV;
    case EBPF_OP_RSH_IMM:
    case EBPF_OP_RSH_REG:
    case EBPF_OP_RSH64_IMM:
    case EBPF_OP_RSH64_REG:
        return DP2_LSRV;
    case EBPF_OP_ARSH_IMM:
    case EBPF_OP_ARSH_REG:
    case EBPF_OP_ARSH64_IMM:
    case EBPF_OP_ARSH64_REG:
        return DP2_ASRV;
    case EBPF_OP_DIV_IMM:
    case EBPF_OP_DIV_REG:
    case EBPF_OP_DIV64_IMM:
    case EBPF_OP_DIV64_REG:
        return DP2_UDIV;
    default:
        assert(false);
        return (enum DP2Opcode)BAD_OPCODE;
    }
}

static enum LoadStoreOpcode
to_loadstore_opcode(int opcode)
{
    switch (opcode)
    {
    case EBPF_OP_LDXW:
        return LS_LDRW;
    case EBPF_OP_LDXH:
        return LS_LDRH;
    case EBPF_OP_LDXB:
        return LS_LDRB;
    case EBPF_OP_LDXDW:
        return LS_LDRX;
    case EBPF_OP_STW:
    case EBPF_OP_STXW:
        return LS_STRW;
    case EBPF_OP_STH:
    case EBPF_OP_STXH:
        return LS_STRH;
    case EBPF_OP_STB:
    case EBPF_OP_STXB:
        return LS_STRB;
    case EBPF_OP_STDW:
    case EBPF_OP_STXDW:
        return LS_STRX;
    default:
        assert(false);
        return (enum LoadStoreOpcode)BAD_OPCODE;
    }
}

static enum Condition
to_condition(int opcode)
{
    switch (opcode)
    {
    case EBPF_OP_JEQ_IMM:
    case EBPF_OP_JEQ_REG:
        return COND_EQ;
    case EBPF_OP_JGT_IMM:
    case EBPF_OP_JGT_REG:
        return COND_HI;
    case EBPF_OP_JGE_IMM:
    case EBPF_OP_JGE_REG:
        return COND_HS;
    case EBPF_OP_JLT_IMM:
    case EBPF_OP_JLT_REG:
        return COND_LO;
    case EBPF_OP_JLE_IMM:
    case EBPF_OP_JLE_REG:
        return COND_LS;
    case EBPF_OP_JSET_IMM:
    case EBPF_OP_JSET_REG:
        return COND_NE;
    case EBPF_OP_JNE_IMM:
    case EBPF_OP_JNE_REG:
        return COND_NE;
    case EBPF_OP_JSGT_IMM:
    case EBPF_OP_JSGT_REG:
        return COND_GT;
    case EBPF_OP_JSGE_IMM:
    case EBPF_OP_JSGE_REG:
        return COND_GE;
    case EBPF_OP_JSLT_IMM:
    case EBPF_OP_JSLT_REG:
        return COND_LE;
    case EBPF_OP_JSLE_IMM:
    case EBPF_OP_JSLE_REG:
        return COND_LE;
    default:
        assert(false);
        return COND_NV;
    }
}


static int
translate(struct ubpf_vm *vm, struct jit_state *state, char **errmsg)
{
    int i;

    emit_function_prologue(state, UBPF_STACK_SIZE);

    for (i = 0; i < vm->num_insts; i++) {
        struct ebpf_inst inst = vm->insts[i];
        state->pc_locs[i] = state->offset;

        enum Registers dst = map_register(inst.dst);
        enum Registers src = map_register(inst.src);
        uint32_t target_pc = i + inst.offset + 1;

        int sixty_four = is_alu64_op(&inst);

        if (is_alu_imm_op(&inst)) {
            emit_movewide_immediate(state, sixty_four, temp_register, inst.imm);
            src = temp_register;
        }

        switch (inst.opcode) {
        case EBPF_OP_ADD_IMM:
        case EBPF_OP_ADD_REG:
        case EBPF_OP_SUB_IMM:
        case EBPF_OP_SUB_REG:
        case EBPF_OP_ADD64_IMM:
        case EBPF_OP_ADD64_REG:
        case EBPF_OP_SUB64_IMM:
        case EBPF_OP_SUB64_REG:
            emit_addsub_register(state, sixty_four, to_addsub_opcode(inst.opcode), dst, dst, src);
            break;
        case EBPF_OP_LSH_IMM:
        case EBPF_OP_LSH_REG:
        case EBPF_OP_RSH_IMM:
        case EBPF_OP_RSH_REG:
        case EBPF_OP_ARSH_IMM:
        case EBPF_OP_ARSH_REG:
        case EBPF_OP_LSH64_IMM:
        case EBPF_OP_LSH64_REG:
        case EBPF_OP_RSH64_IMM:
        case EBPF_OP_RSH64_REG:
        case EBPF_OP_ARSH64_IMM:
        case EBPF_OP_ARSH64_REG:
            /* TODO: CHECK imm is small enough.  */
            emit_dataprocessing_twosource(state, sixty_four, to_dp2_opcode(inst.opcode), dst, dst, src);
            break;
        case EBPF_OP_MUL_IMM:
        case EBPF_OP_MUL_REG:
        case EBPF_OP_MUL64_IMM:
        case EBPF_OP_MUL64_REG:
            emit_dataprocessing_threesource(state, sixty_four, DP3_MADD, dst, dst, src, RZ);
            break;
        case EBPF_OP_DIV_IMM:
        case EBPF_OP_DIV_REG:
        case EBPF_OP_MOD_IMM:
        case EBPF_OP_MOD_REG:
        case EBPF_OP_DIV64_IMM:
        case EBPF_OP_DIV64_REG:
        case EBPF_OP_MOD64_IMM:
        case EBPF_OP_MOD64_REG:
            divmod(state, i, inst.opcode, dst, dst, src);
            break;
        case EBPF_OP_OR_IMM:
        case EBPF_OP_OR_REG:
        case EBPF_OP_AND_IMM:
        case EBPF_OP_AND_REG:
        case EBPF_OP_XOR_IMM:
        case EBPF_OP_XOR_REG:
        case EBPF_OP_OR64_IMM:
        case EBPF_OP_OR64_REG:
        case EBPF_OP_AND64_IMM:
        case EBPF_OP_AND64_REG:
        case EBPF_OP_XOR64_IMM:
        case EBPF_OP_XOR64_REG:
            emit_logical_register(state, sixty_four, to_logical_opcode(inst.opcode), dst, dst, src);
            break;
        case EBPF_OP_NEG:
        case EBPF_OP_NEG64:
            emit_addsub_register(state, sixty_four, AS_SUB, dst, RZ, src);
            break;
        case EBPF_OP_MOV_IMM:
        case EBPF_OP_MOV_REG:
        case EBPF_OP_MOV64_IMM:
        case EBPF_OP_MOV64_REG:
            emit_logical_register(state, sixty_four, LOG_ORR, dst, RZ, src);
            break;
        case EBPF_OP_LE:
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            /* No-op */
#else
            emit_dataprocessing_onesource(state, sixty_four, to_dp1_opcode(inst.opcode, inst.imm), dst, dst);
#endif
            if (inst.imm == 16) {
                emit_instruction(state, (1 << sixty_four) | 0x53003c00 | (dst << 5) | dst);
            }
            break;
        case EBPF_OP_BE:
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            emit_dataprocessing_onesource(state, sixty_four, to_dp1_opcode(inst.opcode, inst.imm), dst, dst);
#else
            /* No-op. */
#endif
            if (inst.imm == 16) {
                emit_instruction(state, (1 << sixty_four) | 0x53003c00 | (dst << 5) | dst);
            }
            break;

        /* TODO use 8 bit immediate when possible */
        case EBPF_OP_JA:
            emit_unconditionalbranch_immediate(state, UBR_B, target_pc);
            break;
        case EBPF_OP_JEQ_IMM:
        case EBPF_OP_JEQ_REG:
        case EBPF_OP_JGT_IMM:
        case EBPF_OP_JGT_REG:
        case EBPF_OP_JGE_IMM:
        case EBPF_OP_JGE_REG:
        case EBPF_OP_JLT_IMM:
        case EBPF_OP_JLT_REG:
        case EBPF_OP_JLE_IMM:
        case EBPF_OP_JLE_REG:
        case EBPF_OP_JNE_IMM:
        case EBPF_OP_JNE_REG:
        case EBPF_OP_JSGT_IMM:
        case EBPF_OP_JSGT_REG:
        case EBPF_OP_JSGE_IMM:
        case EBPF_OP_JSGE_REG:
        case EBPF_OP_JSLT_IMM:
        case EBPF_OP_JSLT_REG:
        case EBPF_OP_JSLE_IMM:
        case EBPF_OP_JSLE_REG:
            emit_addsub_register(state, sixty_four, AS_SUBS, RZ, dst, src);
            emit_conditionalbranch_immediate(state, to_condition(inst.opcode), target_pc);
            break;
        case EBPF_OP_JSET_IMM:
        case EBPF_OP_JSET_REG:
            emit_logical_register(state, sixty_four, LOG_ANDS, RZ, dst, src);
            emit_conditionalbranch_immediate(state, to_condition(inst.opcode), target_pc);
            break;
        case EBPF_OP_CALL:
            emit_call(state, (uintptr_t)vm->ext_funcs[inst.imm]);
            if (inst.imm == vm->unwind_stack_extension_index) {
                emit_addsub_immediate(state, true, AS_SUBS, RZ, map_register(0), 0);
                emit_conditionalbranch_immediate(state, COND_EQ, TARGET_PC_EXIT);
            }
            break;
        case EBPF_OP_EXIT:
            if (i != vm->num_insts - 1) {
                emit_unconditionalbranch_immediate(state, UBR_B, TARGET_PC_EXIT);
            }
            break;

        case EBPF_OP_LDXW:
        case EBPF_OP_LDXH:
        case EBPF_OP_LDXB:
        case EBPF_OP_LDXDW:
            emit_loadstore_immediate(state, to_loadstore_opcode(inst.opcode), dst, src, inst.offset);
            break;

        case EBPF_OP_STW:
        case EBPF_OP_STH:
        case EBPF_OP_STB:
        case EBPF_OP_STDW:
        case EBPF_OP_STXW:
        case EBPF_OP_STXH:
        case EBPF_OP_STXB:
        case EBPF_OP_STXDW:
            emit_loadstore_immediate(state, to_loadstore_opcode(inst.opcode), src, dst, inst.offset);
            break;

        case EBPF_OP_LDDW: {
            struct ebpf_inst inst2 = vm->insts[++i];
            uint64_t imm = (uint32_t)inst.imm | ((uint64_t)inst2.imm << 32);
            emit_movewide_immediate(state, true, dst, imm);
            break;
        }

        default:
            *errmsg = ubpf_error("Unknown instruction at PC %d: opcode %02x", i, inst.opcode);
            return -1;
        }
    }

    emit_function_epilogue(state);
    /* Division by zero handler */

    // Save the address of the start of the divide by zero handler.
    state->div_by_zero_loc = state->offset;

    /* error_printf(stderr, UBPF_STRING_ID_DIVIDE_BY_ZERO, pc);
       pc has already been set up by emit_divmod(). */
    emit_string_load(state, caller_saved_registers[1], UBPF_STRING_ID_DIVIDE_BY_ZERO);
    emit_movewide_immediate(state, true,caller_saved_registers[0], (uintptr_t)stderr);
    emit_call(state, (uintptr_t)vm->error_printf);

    emit_movewide_immediate(state, true, map_register(0), ((uint64_t)INT64_C(-1)));
    emit_unconditionalbranch_immediate(state, UBR_B, TARGET_PC_EXIT);

    // Emit string table.
    state->string_table_loc = state->offset;
    for (i = 0; i < _countof(ubpf_string_table); i++) {
        emit_bytes(state, (void*)ubpf_string_table[i], strlen(ubpf_string_table[i]) + 1);
    }

    return 0;
}

static void
divmod(struct jit_state *state, uint16_t pc, uint8_t opcode, int rd, int rn, int rm)
{
    bool mod = (opcode & EBPF_ALU_OP_MASK) == (EBPF_OP_MOD_IMM & EBPF_ALU_OP_MASK);
    bool sixty_four = (opcode & EBPF_CLS_MASK) == EBPF_CLS_ALU64;
    enum Registers div_dest = mod ? temp_div_register : rd;

    // Handle divide by zero case
    // CBNZ rm, .+12 (we don't need to note_jump() as we know the destination immediately).
    emit_instruction(state, (sixty_four << 31) | CBR_CBNZ | (3 << 5) | rm);
    emit_movewide_immediate(state, true, caller_saved_registers[2], pc);
    emit_unconditionalbranch_immediate(state, UBR_B, TARGET_PC_DIV_BY_ZERO);
    emit_dataprocessing_twosource(state, sixty_four, DP2_UDIV, div_dest, rn, rm);
    if (mod) {
        emit_dataprocessing_threesource(state, sixty_four, DP3_MSUB, rd, rm, div_dest, rn);
    }
}

static void
resolve_jumps(struct jit_state *state)
{
    for (unsigned i = 0; i < state->num_jumps; ++i) {
        struct jump jump = state->jumps[i];

        int32_t target_loc;
        if (jump.target_pc == TARGET_PC_EXIT) {
            target_loc = state->exit_loc;
        } else if (jump.target_pc == TARGET_PC_DIV_BY_ZERO) {
            target_loc = state->div_by_zero_loc;
        } else {
            target_loc = state->pc_locs[jump.target_pc];
        }

        int32_t rel = target_loc - jump.offset_loc;
        update_branch_immediate(state, jump.offset_loc, rel);
    }
}

static uint32_t
string_offset_from_id(struct jit_state *state, uint32_t string_id)
{
    uint32_t offset = state->string_table_loc;
    uint32_t i;
    for (i = 0; i < string_id; i ++) {
        offset += strlen(ubpf_string_table[i]) + 1;
    }
    return offset;
}

static void
resolve_strings(struct jit_state *state)
{
    int i;
    for (i = 0; i < state->num_strings; i++) {
        struct string_reference string = state->strings[i];
        int64_t rel = string_offset_from_id(state, string.string_id) - string.offset_loc;
        update_adr_immediate(state, string.offset_loc, rel);
    }
}


int
ubpf_translate_arm64(struct ubpf_vm *vm, uint8_t * buffer, size_t * size, char **errmsg)
{
    struct jit_state state;
    int result = -1;

    state.offset = 0;
    state.size = *size;
    state.buf = buffer;
    state.pc_locs = calloc(UBPF_MAX_INSTS+1, sizeof(state.pc_locs[0]));
    state.jumps = calloc(UBPF_MAX_INSTS, sizeof(state.jumps[0]));
    state.strings = calloc(UBPF_MAX_INSTS, sizeof(state.strings[0]));
    state.num_jumps = 0;
    state.num_strings = 0;

    if (translate(vm, &state, errmsg) < 0) {
        goto out;
    }

    if (state.num_jumps == UBPF_MAX_INSTS) {
        *errmsg = ubpf_error("Excessive number of jump targets");
        goto out;
    }

    if (state.num_strings == UBPF_MAX_INSTS) {
        *errmsg = ubpf_error("Excessive number of string targets");
        goto out;
    }

    if (state.offset == state.size) {
        *errmsg = ubpf_error("Target buffer too small");
        goto out;
    }

    resolve_jumps(&state);
    resolve_strings(&state);
    result = 0;

    *size = state.offset;

out:
    free(state.pc_locs);
    free(state.jumps);
    free(state.strings);
    return result;
}
