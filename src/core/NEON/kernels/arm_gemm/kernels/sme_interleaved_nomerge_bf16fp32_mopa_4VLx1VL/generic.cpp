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

#ifdef __aarch64__
#include <arm_neon.h>
__asm__(
    ".include \"src/core/NEON/kernels/arm_gemm/helper.inc\"\n"
);
#endif

#include "arm_gemm.hpp"

#include "../../bfloat.hpp"
#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme_interleaved_nomerge_bf16fp32_mopa_4VLx1VL(const bfloat16 *const A, const bfloat16 *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const bfloat16 *const A,
      const bfloat16 *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const float *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 2) * sizeof(bfloat16)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        min(-std::numeric_limits<float>::infinity()),
        max(std::numeric_limits<float>::infinity()),
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
      if (act.type == Activation::Type::None)
      {
        flags |= 1 << 2;  // SKIP_ACTIVATION
      }

      // Initialise the activation values
      switch (act.type)
      {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            this->max = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<float>(0);
            break;
      }
    }

    const bfloat16 *const A;
    const bfloat16 *const B;
    const long kstride_bytes;
    float *const C;
    const long ldcb;
    const long M, N, K;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const float *const bias;


    float *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x16, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x16, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      "ld1w_4 z20.s, z21.s, z22.s, z23.s, p1, x15\n"
      "ld1w_4 z24.s, z25.s, z26.s, z27.s, p1, x15\n"
      "ld1w_4 z0.s,  z1.s,  z2.s,  z3.s , p1, x15\n"
      "ld1w_4 z28.s, z29.s, z30.s, z31.s, p1, x15\n"
      "move_vector_tile za0h.s, z20.s, z21.s, z22.s, z23.s,  w12, p1 \n"
      "move_vector_tile za1h.s, z24.s, z25.s, z26.s, z27.s,  w12, p1 \n"
      "move_vector_tile za2h.s, z0.s,  z1.s,  z2.s,  z3.s ,  w12, p1 \n"
      "move_vector_tile za3h.s, z28.s, z29.s, z30.s, z31.s,  w12, p1 \n"
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
      "whilelt p0.s, x10, x9\n"
      "tbnz x16, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      "zero { za }\n"
      "cbz x20, 5f\n"
      "fmov z6.s, #1.0\n"
      "ld1w { z26.s }, p0/Z, [x20, x10, LSL #2]\n"
      ".inst 0x809a24c0  // fmopa za0.s, p1/M, p1/M, z6.s, z26.s\n"
      ".inst 0x809a24c1  // fmopa za1.s, p1/M, p1/M, z6.s, z26.s\n"
      ".inst 0x809a24c2  // fmopa za2.s, p1/M, p1/M, z6.s, z26.s\n"
      ".inst 0x809a24c3  // fmopa za3.s, p1/M, p1/M, z6.s, z26.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x10\n"
      "mov x21, x11\n"
      "incw x20\n"
      "incw x21, ALL, MUL #4\n"
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
      "add x20, x20, #0x1\n"
      "lsr x20, x20, #0x1\n"
      "lsr x21, x20, #0x2\n"
      "madd x23, x10, x22, x23\n"  // bptr = B + n * kstride_bytes
      "and x20, x20, #0x3\n"
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      "ld1h_4 z3.h,  z7.h,  z11.h, z15.h , p1, x27\n"
      "ld1h { z13.h }, p1/Z, [x23]\n"
      "ld1h_4 z18.h, z22.h, z26.h, z30.h , p1, x27\n"
      "ld1h { z21.h }, p1/Z, [x23, #1, MUL VL]\n"
      "ld1h_4 z19.h, z23.h, z27.h, z31.h , p1, x27\n"
      "ld1h { z17.h }, p1/Z, [x23, #2, MUL VL]\n"
      "ld1h_4 z16.h, z20.h, z24.h, z28.h , p1, x27\n"
      "ld1h { z2.h }, p1/Z, [x23, #3, MUL VL]\n"
      "addvl x23, x23, #4\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0x818d2460  // bfmopa za0.s, p1/M, p1/M, z3.h, z13.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x818d24e1  // bfmopa za1.s, p1/M, p1/M, z7.h, z13.h\n"
      ".inst 0x818d2562  // bfmopa za2.s, p1/M, p1/M, z11.h, z13.h\n"
      ".inst 0x818d25e3  // bfmopa za3.s, p1/M, p1/M, z15.h, z13.h\n"
      "ld1h_4 z3.h,  z7.h,  z11.h, z15.h , p1, x27\n"
      ".inst 0x81952640  // bfmopa za0.s, p1/M, p1/M, z18.h, z21.h\n"
      "ld1h { z13.h }, p1/Z, [x23]\n"
      ".inst 0x819526c1  // bfmopa za1.s, p1/M, p1/M, z22.h, z21.h\n"
      ".inst 0x81952742  // bfmopa za2.s, p1/M, p1/M, z26.h, z21.h\n"
      ".inst 0x819527c3  // bfmopa za3.s, p1/M, p1/M, z30.h, z21.h\n"
      "ld1h_4 z18.h, z22.h, z26.h, z30.h , p1, x27\n"
      ".inst 0x81912660  // bfmopa za0.s, p1/M, p1/M, z19.h, z17.h\n"
      "ld1h { z21.h }, p1/Z, [x23, #1, MUL VL]\n"
      ".inst 0x819126e1  // bfmopa za1.s, p1/M, p1/M, z23.h, z17.h\n"
      ".inst 0x81912762  // bfmopa za2.s, p1/M, p1/M, z27.h, z17.h\n"
      ".inst 0x819127e3  // bfmopa za3.s, p1/M, p1/M, z31.h, z17.h\n"
      "ld1h_4 z19.h, z23.h, z27.h, z31.h , p1, x27\n"
      "ld1h { z17.h }, p1/Z, [x23, #2, MUL VL]\n"
      ".inst 0x81822600  // bfmopa za0.s, p1/M, p1/M, z16.h, z2.h\n"
      ".inst 0x81822681  // bfmopa za1.s, p1/M, p1/M, z20.h, z2.h\n"
      ".inst 0x81822702  // bfmopa za2.s, p1/M, p1/M, z24.h, z2.h\n"
      ".inst 0x81822783  // bfmopa za3.s, p1/M, p1/M, z28.h, z2.h\n"
      "ld1h_4 z16.h, z20.h, z24.h, z28.h , p1, x27\n"
      "ld1h { z2.h }, p1/Z, [x23, #3, MUL VL]\n"
      "addvl x23, x23, #4\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0x818d2460  // bfmopa za0.s, p1/M, p1/M, z3.h, z13.h\n"
      ".inst 0x818d24e1  // bfmopa za1.s, p1/M, p1/M, z7.h, z13.h\n"
      ".inst 0x818d2562  // bfmopa za2.s, p1/M, p1/M, z11.h, z13.h\n"
      ".inst 0x818d25e3  // bfmopa za3.s, p1/M, p1/M, z15.h, z13.h\n"
      ".inst 0x81952640  // bfmopa za0.s, p1/M, p1/M, z18.h, z21.h\n"
      ".inst 0x819526c1  // bfmopa za1.s, p1/M, p1/M, z22.h, z21.h\n"
      ".inst 0x81952742  // bfmopa za2.s, p1/M, p1/M, z26.h, z21.h\n"
      ".inst 0x819527c3  // bfmopa za3.s, p1/M, p1/M, z30.h, z21.h\n"
      ".inst 0x81912660  // bfmopa za0.s, p1/M, p1/M, z19.h, z17.h\n"
      ".inst 0x819126e1  // bfmopa za1.s, p1/M, p1/M, z23.h, z17.h\n"
      ".inst 0x81912762  // bfmopa za2.s, p1/M, p1/M, z27.h, z17.h\n"
      ".inst 0x819127e3  // bfmopa za3.s, p1/M, p1/M, z31.h, z17.h\n"
      ".inst 0x81822600  // bfmopa za0.s, p1/M, p1/M, z16.h, z2.h\n"
      ".inst 0x81822681  // bfmopa za1.s, p1/M, p1/M, z20.h, z2.h\n"
      ".inst 0x81822702  // bfmopa za2.s, p1/M, p1/M, z24.h, z2.h\n"
      ".inst 0x81822783  // bfmopa za3.s, p1/M, p1/M, z28.h, z2.h\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      "ld1h_4 z16.h, z20.h, z24.h, z28.h , p1, x27\n"
      "subs x20, x20, #0x1\n"
      "ld1h { z2.h }, p1/Z, [x23]\n"
      "addvl x23, x23, #1\n"
      ".inst 0x81822600  // bfmopa za0.s, p1/M, p1/M, z16.h, z2.h\n"
      ".inst 0x81822681  // bfmopa za1.s, p1/M, p1/M, z20.h, z2.h\n"
      ".inst 0x81822702  // bfmopa za2.s, p1/M, p1/M, z24.h, z2.h\n"
      ".inst 0x81822783  // bfmopa za3.s, p1/M, p1/M, z28.h, z2.h\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x16, #1, 14f\n"
      "tbz x16, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      "ld1w_4 z16.s, z17.s, z18.s, z19.s, p1, x15 \n"
      "move_tile_vector z8.s,  z9.s,  z10.s, z11.s, za0h.s, w12, p1\n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za1h.s, w12, p1\n"
      "ld1w_4 z24.s, z25.s, z26.s, z27.s, p1, x15 \n"
      "move_tile_vector z28.s, z29.s, z30.s, z31.s, za2h.s, w12, p1\n"
      "move_tile_vector z20.s, z21.s, z22.s, z23.s, za3h.s, w12, p1\n"
      "ld1w_4 z4.s,  z5.s,  z6.s,  z7.s , p1, x15 \n"
      "ld1w_4 z0.s,  z1.s,  z2.s,  z3.s , p1, x15 \n"
      "move_vector_tile za0h.s, z16.s, z17.s, z18.s, z19.s, w12, p1\n"
      "move_vector_tile za1h.s, z24.s, z25.s, z26.s, z27.s, w12, p1\n"
      "st1w_4 z8.s,  z9.s,  z10.s, z11.s, p1,  x14 \n"
      "move_vector_tile za2h.s, z4.s,  z5.s,  z6.s,  z7.s , w12, p1\n"
      "st1w_4 z12.s, z13.s, z14.s, z15.s, p1,  x14 \n"
      "move_vector_tile za3h.s, z0.s,  z1.s,  z2.s,  z3.s , w12, p1\n"
      "add x12, x12, #0x4\n"
      "st1w_4 z28.s, z29.s, z30.s, z31.s, p1,  x14 \n"
      "cmp x12, x20\n"
      "st1w_4 z20.s, z21.s, z22.s, z23.s, p1,  x14 \n"
      "blt 11b\n"
      "b 42f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p1\n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za1h.s, w12, p1\n"
      "move_tile_vector z4.s,  z5.s,  z6.s,  z7.s , za2h.s, w12, p1\n"
      "move_tile_vector z8.s,  z9.s,  z10.s, z11.s, za3h.s, w12, p1\n"
      "st1w_4 z16.s, z17.s, z18.s, z19.s, p1,  x14 \n"
      "add x12, x12, #0x4\n"
      "st1w_4 z12.s, z13.s, z14.s, z15.s, p1,  x14 \n"
      "cmp x12, x20\n"
      "st1w_4 z4.s,  z5.s,  z6.s,  z7.s , p1,  x14 \n"
      "st1w_4 z8.s,  z9.s,  z10.s, z11.s, p1,  x14 \n"	
      "blt 13b\n"
      "b 42f\n"
      "14:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x13, x11\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "add x26, x26, x10, LSL #2\n"  // C += n
      "madd x26, x11, x24, x26\n"  // C += m * ldc
      "tbz x16, #2, 27f\n"
      "cntw x23\n"
      "mov x12, #0x0\n"
      "cmp x25, x23\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 16f\n"
      "15:"  // Store to output array: Skip activation: Accumulator row 0 loop
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za0h.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z14.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z15.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 15b\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 17f\n"
      "subs x20, x20, #0x1\n"
      "move_tile_vector z0.s,  z1.s,  z2.s,  z3.s , za0h.s, w12, p1\n"
      "st1w { z0.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 17f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z1.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 17f\n"
      "st1w { z2.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 27f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 19f\n"
      "18:"  // Store to output array: Skip activation: Accumulator row 1 loop
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za1h.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z14.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z15.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 18b\n"
      "19:"  // Store to output array: Skip activation: Accumulator row 1 oddments
      "cbz x20, 20f\n"
      "subs x20, x20, #0x1\n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za1h.s, w12, p1\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 20f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 20f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "20:"  // Store to output array: Skip activation: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 27f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 22f\n"
      "21:"  // Store to output array: Skip activation: Accumulator row 2 loop
      "move_tile_vector z8.s,  z9.s,  z10.s, z11.s, za2h.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "st1w { z8.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z9.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z10.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z11.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 21b\n"
      "22:"  // Store to output array: Skip activation: Accumulator row 2 oddments
      "cbz x20, 23f\n"
      "subs x20, x20, #0x1\n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za2h.s, w12, p1\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 23f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 23f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "23:"  // Store to output array: Skip activation: Accumulator row 2 oddments: End
      "subs x25, x25, x22\n"
      "beq 27f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 25f\n"
      "24:"  // Store to output array: Skip activation: Accumulator row 3 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za3h.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 24b\n"
      "25:"  // Store to output array: Skip activation: Accumulator row 3 oddments
      "cbz x20, 26f\n"
      "subs x20, x20, #0x1\n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za3h.s, w12, p1\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 26f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 26f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "26:"  // Store to output array: Skip activation: Accumulator row 3 oddments: End
      "subs x25, x25, x22\n"
      "beq 27f\n"
      "b 40f\n"
      "27:"  // Store to output array: Skip activation: End
      "cntw x23\n"
      "ld1rw { z21.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0x0\n"
      "cmp x25, x23\n"
      "ld1rw { z20.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 29f\n"
      "28:"  // Store to output array: Accumulator row 0 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p1   \n"
      "add x12, x12, #0x4\n"
      "clamp_float_4    z16.s, z17.s, z18.s, z19.s, z21.s,  z20.s, p1 \n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 28b\n"
      "29:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 30f\n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p1   \n"
      "subs x20, x20, #0x1\n"
      "clamp_float_4    z16.s, z17.s, z18.s, z19.s, z21.s,  z20.s, p1 \n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 30f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 30f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "30:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 40f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 32f\n"
      "31:"  // Store to output array: Accumulator row 1 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za1h.s, w12, p1   \n"
      "add x12, x12, #0x4\n"
      "clamp_float_4    z16.s, z17.s, z18.s, z19.s, z21.s,  z20.s, p1 \n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 31b\n"
      "32:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 33f\n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za1h.s, w12, p1   \n"
      "subs x20, x20, #0x1\n"
      "clamp_float_4    z16.s, z17.s, z18.s, z19.s, z21.s,  z20.s, p1 \n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 33f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 33f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "33:"  // Store to output array: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 40f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 35f\n"
      "34:"  // Store to output array: Accumulator row 2 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za2h.s, w12, p1   \n"
      "add x12, x12, #0x4\n"
      "clamp_float_4    z16.s, z17.s, z18.s, z19.s, z21.s,  z20.s, p1 \n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 34b\n"
      "35:"  // Store to output array: Accumulator row 2 oddments
      "cbz x20, 36f\n"
      "move_tile_vector z28.s, z29.s, z30.s, z31.s, za2h.s, w12, p1   \n"
      "subs x20, x20, #0x1\n"
      "clamp_float_4    z28.s, z29.s, z30.s, z31.s, z21.s,  z20.s, p1 \n"
      "st1w { z28.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 36f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z29.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 36f\n"
      "st1w { z30.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "36:"  // Store to output array: Accumulator row 2 oddments: End
      "subs x25, x25, x22\n"
      "beq 40f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x20, x25, x23, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 38f\n"
      "37:"  // Store to output array: Accumulator row 3 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za3h.s, w12, p1   \n"
      "add x12, x12, #0x4\n"
      "clamp_float_4    z16.s, z17.s, z18.s, z19.s, z21.s,  z20.s, p1 \n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 37b\n"
      "38:"  // Store to output array: Accumulator row 3 oddments
      "cbz x20, 39f\n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p1   \n"
      "subs x20, x20, #0x1\n"
      "clamp_float_4    z12.s, z13.s, z14.s, z15.s, z21.s,  z20.s, p1 \n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 39f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 39f\n"
      "st1w { z14.s }, p0, [x26]\n"
      "39:"  // Store to output array: Accumulator row 3 oddments: End
      "40:"  // Store to output array: End
      "tbz x16, #0, 42f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "41:"  // Store to output array: Refill accumulators: Loop
      "ld1w_4 z12.s, z13.s, z14.s, z15.s, p1, x15 \n"
      "ld1w_4 z28.s, z29.s, z30.s, z31.s, p1, x15 \n"
      "ld1w_4 z4.s,  z5.s,  z6.s,  z7.s , p1, x15 \n"
      "ld1w_4 z8.s,  z9.s,  z10.s, z11.s, p1, x15 \n"
      "move_vector_tile za0h.s, z12.s, z13.s, z14.s, z15.s, w12, p1\n"
      "move_vector_tile za1h.s, z28.s, z29.s, z30.s, z31.s, w12, p1\n"
      "move_vector_tile za2h.s, z4.s,  z5.s,  z6.s,  z7.s , w12, p1\n"
      "move_vector_tile za3h.s, z8.s,  z9.s,  z10.s, z11.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 41b\n"
      "42:"  // End block
      "incw x10\n"
      "cmp x10, x9\n"
      "blt 3b\n"
      "incw x11, ALL, MUL #4\n"
      "mov x10, #0x0\n"
      "cmp x11, x13\n"
      "mov x28, x27\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME
