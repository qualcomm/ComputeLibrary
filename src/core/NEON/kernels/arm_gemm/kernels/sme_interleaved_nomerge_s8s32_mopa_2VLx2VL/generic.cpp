/*
 * Copyright (c) 2022-2024 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifdef ARM_COMPUTE_ENABLE_SME

// generic.cpp
#ifdef __aarch64__
#include <arm_neon.h>
__asm__(
    ".include \"src/core/NEON/kernels/arm_gemm/helper.inc\"\n"
);
#endif

#include "arm_gemm.hpp"

#include <cstdint>
#include "../../asmlib.hpp"
#include "../../utils.hpp"
namespace arm_gemm {

void sme_interleaved_nomerge_s8s32_mopa_2VLx2VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      int32_t *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,

      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(int32_t)),
        M(M), N(N), K(K),

        bias(bias),
        accumulator_buffer(accumulator_buffer),
        flags(0x0)
    {
      if (accumulate)
      {
        flags |= 1 << 0;  // FILL_ACCUMULATORS_FROM_BUFFER
      }
      if (C == nullptr)
      {
        flags |= 1 << 1;  // STORE_ACCUMULATORS_TO_BUFFER
      }
      }

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    int32_t *const C;
    const long ldcb;
    const long M, N, K;

    const int32_t *const bias;


    int32_t *const accumulator_buffer;
    uint64_t flags;
  };
          printf("[GEMM] Selected s8_s32  2VL x 2VL  \n");

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x16, [%x[args], %[offsetof_flags]]\n"
      "SMSTART\n"
      "ptrue p0.b\n"
      "ptrue p1.b\n"
      "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz w16, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      "ld1w_4 z24.s, z25.s, z26.s, z27.s, p1, x15\n"
      "ld1w_4 z28.s, z29.s, z30.s, z31.s, p1, x15\n"
      "ld1w_4 z16.s, z17.s, z18.s, z19.s, p1, x15\n"
      "ld1w_4 z8.s, z9.s, z10.s, z11.s,   p1, x15\n"
      "move_vector_tile za0h.s, z24.s, z25.s, z26.s, z27.s, w12, p1\n"
      "move_vector_tile za1h.s, z28.s, z29.s, z30.s, z31.s, w12, p1\n"
      "move_vector_tile za2h.s, z16.s, z17.s, z18.s, z19.s, w12, p1\n"
      "move_vector_tile za3h.s, z8.s, z9.s, z10.s, z11.s,   w12, p1\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w13, [%x[args], %[offsetof_M]]\n"
      "mov x11, #0x0\n"
      "mov x10, #0x0\n"
      "ldr w9, [%x[args], %[offsetof_N]]\n"
      "ldr x28, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x27, x28\n"
	  "whilelt p2.s, x10, x9\n"
	  "incw x10\n"
	  "whilelt p3.s, x10, x9\n"
	  "decw x10\n"
      "tbnz w16, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      "zero {za}\n"
      "cbz x20, 5f\n"
      "ld1w { z22.s, z30.s }, pn8/Z, [x20, x10, LSL #2]\n"
      "addha za0.s, p0/M, p0/M, z22.s\n"
      "addha za1.s, p0/M, p0/M, z30.s\n"
      "addha za2.s, p0/M, p0/M, z22.s\n"
      "addha za3.s, p0/M, p0/M, z30.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x10\n"
      "mov x21, x11\n"
      "incw x20, ALL, MUL #2\n"
      "incw x21, ALL, MUL #2\n"
      "cmp x20, x9\n"
      "mov x20, x16\n"
      "csel x21, x11, x21, LT\n"
      "bfm x16, XZR, #0x0, #0x0  // bfc x16, #0x0, #0x1\n"
      "cmp x21, x13\n"
      "csel x16, x20, x16, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "ldr x23, [%x[args], %[offsetof_B]]\n"
      "ldr x22, [%x[args], %[offsetof_kstride_bytes]]\n"
      "add x20, x20, #0x3\n"
      "lsr x20, x20, #0x2\n"
      "lsr x21, x20, #0x2\n"
      "madd x23, x10, x22, x23\n"  // bptr = B + n * kstride_bytes
      "and x20, x20, #0x3\n"
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      "ld1b_2 z6.b, z14.b, p1, x27\n"
      "ld1b_2 z2.b, z3.b, p1, x23\n"
      "ld1b_2 z26.b, z27.b, p1, x27\n"
      "ld1b_2 z22.b, z23.b, p1, x23\n"
      "ld1b_2 z5.b, z13.b, p1, x27\n"
      "ld1b_2 z21.b, z29.b, p1, x23\n"
      "ld1b_2 z0.b, z1.b, p1, x27\n"
      "ld1b_2 z17.b, z25.b, p1, x23\n"
      "ble 7f\n"
      "6:"  // K loop
      "smopa za0.s, p0/M, p0/M, z6.b, z2.b\n"
      "subs x21, x21, #0x1\n"
      "smopa za1.s, p0/M, p0/M, z6.b, z3.b\n"
      "smopa za2.s, p0/M, p0/M, z14.b, z2.b\n"
      "smopa za3.s, p0/M, p0/M, z14.b, z3.b\n"
	  "ld1b_2 z6.b, z14.b, p1, x27 \n"
      "smopa za0.s, p0/M, p0/M, z26.b, z22.b\n"
      "ld1b_2 z2.b, z3.b, p1, x23\n"
      "smopa za1.s, p0/M, p0/M, z26.b, z23.b\n"
      "smopa za2.s, p0/M, p0/M, z27.b, z22.b\n"
      "smopa za3.s, p0/M, p0/M, z27.b, z23.b\n"
	  "ld1b_2 z26.b,z27.b, p1, x27 \n"
      "smopa za0.s, p0/M, p0/M, z5.b, z21.b\n"
      "ld1b_2 z22.b, z23.b, p1, x23\n"
      "smopa za1.s, p0/M, p0/M, z5.b, z29.b\n"
      "smopa za2.s, p0/M, p0/M, z13.b, z21.b\n"
      "smopa za3.s, p0/M, p0/M, z13.b, z29.b\n"
	  "ld1b_2 z5.b, z13.b, p1, x27 \n"
      "ld1b_2 z21.b, z29.b, p1, x23\n"
      "smopa za0.s, p0/M, p0/M, z0.b, z17.b\n"
      "smopa za1.s, p0/M, p0/M, z0.b, z25.b\n"
      "smopa za2.s, p0/M, p0/M, z1.b, z17.b\n"
      "smopa za3.s, p0/M, p0/M, z1.b, z25.b\n"
	  "ld1b_2 z0.b,z1.b, p1, x27 \n"
      "ld1b_2 z17.b, z25.b, p1, x23\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      "smopa za0.s, p0/M, p0/M, z6.b, z2.b\n"
      "smopa za1.s, p0/M, p0/M, z6.b, z3.b\n"
      "smopa za2.s, p0/M, p0/M, z14.b, z2.b\n"
      "smopa za3.s, p0/M, p0/M, z14.b, z3.b\n"
      "smopa za0.s, p0/M, p0/M, z26.b, z22.b\n"
      "smopa za1.s, p0/M, p0/M, z26.b, z23.b\n"
      "smopa za2.s, p0/M, p0/M, z27.b, z22.b\n"
      "smopa za3.s, p0/M, p0/M, z27.b, z23.b\n"
      "smopa za0.s, p0/M, p0/M, z5.b, z21.b\n"
      "smopa za1.s, p0/M, p0/M, z5.b, z29.b\n"
      "smopa za2.s, p0/M, p0/M, z13.b, z21.b\n"
      "smopa za3.s, p0/M, p0/M, z13.b, z29.b\n"
      "smopa za0.s, p0/M, p0/M, z0.b, z17.b\n"
      "smopa za1.s, p0/M, p0/M, z0.b, z25.b\n"
      "smopa za2.s, p0/M, p0/M, z1.b, z17.b\n"
      "smopa za3.s, p0/M, p0/M, z1.b, z25.b\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      "ld1b_2 z23.b, z31.b, p1, x27\n"
      "subs x20, x20, #0x1\n"
      "ld1b_2 z7.b, z15.b, p1, x23\n"
      "smopa za0.s, p0/M, p0/M, z23.b, z7.b\n"
      "smopa za1.s, p0/M, p0/M, z23.b, z15.b\n"
      "smopa za2.s, p0/M, p0/M, z31.b, z7.b\n"
      "smopa za3.s, p0/M, p0/M, z31.b, z15.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x16, #1, 14f\n"
      "tbz x16, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      "ld1w_4 z24.s, z25.s, z26.s, z27.s, p1, x15\n"
      "move_tile_vector z28.s, z29.s, z30.s, z31.s, za0h.s, w12, p1\n"
      "move_tile_vector z0.s, z1.s, z2.s, z3.s, za1h.s, w12, p1\n"
      "ld1w_4 z4.s, z5.s, z6.s, z7.s, p1, x15\n"
      "move_tile_vector z8.s, z9.s, z10.s, z11.s, za2h.s, w12, p1\n"
      "move_tile_vector z12.s,z13.s, z14.s, z15.s, za3h.s, w12, p1\n"
      "ld1w_4 z20.s, z21.s, z22.s, z23.s, p1, x15\n"
      "ld1w_4 z16.s, z17.s, z18.s, z19.s, p1, x15\n"
      "move_vector_tile za0h.s, z24.s, z25.s, z26.s, z27.s, w12, p1\n"
      "move_vector_tile za1h.s, z4.s, z5.s, z6.s, z7.s, w12, p1\n"
      "st1w_4 z28.s, z29.s, z30.s, z31.s, p1, x14\n"
      "move_vector_tile za2h.s, z20.s, z21.s, z22.s, z23.s, w12, p1\n"
      "st1w_4 z0.s, z1.s, z2.s, z3.s, p1, x14\n"
      "move_vector_tile za3h.s, z16.s, z17.s, z18.s, z19.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "st1w_4 z8.s, z9.s, z10.s, z11.s, p1, x14\n"
      "cmp x12, x20\n"
      "st1w_4 z12.s, z13.s, z14.s, z15.s, p1, x14\n"
      "blt 11b\n"
      "b 23f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      "move_tile_vector z4.s, z5.s, z6.s, z7.s, za0h.s, w12, p1\n"
      "move_tile_vector z8.s, z9.s, z10.s, z11.s, za1h.s, w12, p1\n"
      "move_tile_vector z0.s, z1.s, z2.s, z3.s, za2h.s, w12, p1\n"
      "move_tile_vector z24.s, z25.s, z26.s, z27.s, za3h.s, w12, p1\n"
      "st1w_4 z4.s, z5.s, z6.s, z7.s, p1, x14\n"
      "add x12, x12, #0x4\n"
      "st1w_4 z8.s, z9.s, z10.s, z11.s, p1, x14\n"
      "cmp x12, x20\n"
      "st1w_4 z0.s, z1.s, z2.s, z3.s, p1, x14\n"
      "st1w_4 z24.s, z25.s, z26.s, z27.s, p1, x14\n"

      "blt 13b\n"
      "b 23f\n"
      "14:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x13, x11\n"
      "cntw x24\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "cmp x25, x24\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x24, LT\n"
      "add x26, x26, x10, LSL #2\n"  // C += n
      "lsr x21, x22, #0x2\n"
      "madd x26, x11, x23, x26\n"  // C += m * ldc
      "and x20, x22, #0x3\n"
      "cbz x21, 16f\n"
      "15:"  // Store to output array: Accumulator row 0 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p1\n"
      "move_tile_vector z24.s, z25.s, z26.s, z27.s, za1h.s, w12, p1\n"
      "st1w_2p z16.s, z24.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "add x12, x12, #0x4\n"
      "st1w_2p z17.s, z25.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "cmp x12, x21, LSL #2\n"
      "st1w_2p z18.s, z26.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "st1w_2p z19.s, z27.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "blt 15b\n"
      "16:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 17f\n"
      "subs x20, x20, #0x1\n"
      "move_tile_vector z4.s, z5.s, z6.s, z7.s, za0h.s, w12, p1\n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za1h.s, w12, p1\n"
      "st1w_2p z4.s, z12.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "beq 17f\n"
      "subs x20, x20, #0x1\n"
      "st1w_2p z5.s, z13.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "beq 17f\n"
      "st1w_2p z6.s, z14.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "17:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 21f\n"
      "cmp x25, x24\n"
      "mov x12, #0x0\n"
      "csel x20, x25, x24, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 19f\n"
      "18:"  // Store to output array: Accumulator row 1 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za2h.s, w12, p1\n"
      "move_tile_vector z24.s, z25.s, z26.s, z27.s, za3h.s, w12, p1\n"
      "st1w_2p z16.s, z24.s , p2, p3, x26\n"
      "add x26, x26, x23\n"
      "add x12, x12, #0x4\n"
      "st1w_2p z17.s, z25.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "cmp x12, x21, LSL #2\n"
      "st1w_2p z18.s, z26.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "st1w_2p z19.s, z27.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "blt 18b\n"
      "19:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 20f\n"
      "subs x20, x20, #0x1\n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za2h.s, w12, p1\n"
      "move_tile_vector z24.s, z25.s, z26.s, z27.s, za3h.s, w12, p1\n"
      "st1w_2p z16.s, z24.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "beq 20f\n"
      "subs x20, x20, #0x1\n"
      "st1w_2p z17.s, z25.s, p2, p3, x26\n"
      "add x26, x26, x23\n"
      "beq 20f\n"
      "st1w_2p z18.s, z26.s, p2, p3, x26\n"
      "20:"  // Store to output array: Accumulator row 1 oddments: End
      "21:"  // Store to output array: End
      "tbz x16, #0, 23f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "22:"  // Store to output array: Refill accumulators: Loop
      "ld1w_4 z8.s, z9.s, z10.s, z11.s, p1, x15\n"
      "ld1w_4 z0.s, z1.s, z2.s, z3.s, p1 x15\n"
      "ld1w_4 z12.s, z13.s, z14.s, z15.s, p1, x15\n"
      "ld1w_4 z4.s, z5.s, z6.s, z7.s, p1, x15\n"
      "move_vector_tile za0h.s, z8.s, z9.s, z10.s, z11.s, w12, p1\n"
      "move_vector_tile za1h.s, z0.s, z1.s, z2.s, z3.s, w12, p1\n"
      "move_vector_tile za2h.s, z12.s, z13.s, z14.s, z15.s, w12, p1\n"
      "move_vector_tile za3h.s, z4.s, z5.s, z6.s, z7.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 22b\n"
      "23:"  // End block
      "incw x10, ALL, MUL #2\n"
      "cmp x10, x9\n"
      "blt 3b\n"
      "incw x11, ALL, MUL #2\n"
      "mov x10, #0x0\n"
      "cmp x11, x13\n"
      "mov x28, x27\n"
      "blt 3b\n"
      "SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME
