/*
 * Copyright (c) 2025 Arm Limited.
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


#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL(const __fp16 *const A, const __fp16 *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const __fp16 *const A,
      const __fp16 *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const float *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 2) * sizeof(__fp16)),
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

    const __fp16 *const A;
    const __fp16 *const B;
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
      "ldr x15, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p0.b\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x15, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      "ld1w_4 z28.s, z29.s, z30.s, z31.s,  p0, x14\n"
      "ld1w_4 z24.s, z25.s, z26.s, z27.s,  p0, x14\n"
      "ld1w_4 z20.s, z21.s, z22.s, z23.s,  p0, x14\n"
      "ld1w_4 z12.s, z13.s, z14.s, z15.s,  p0, x14\n"
      "move_vector_tile za0h.s, z28.s, z29.s, z30.s, z31.s, w12, p0\n"
      "move_vector_tile za1h.s, z24.s, z25.s, z26.s, z27.s, w12, p0\n"
      "move_vector_tile za2h.s, z20.s, z21.s, z22.s, z23.s, w12, p0\n"
      "move_vector_tile za3h.s, z12.s, z13.s, z14.s, z15.s, w12, p0\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w11, [%x[args], %[offsetof_M]]\n"
      "mov x10, #0x0\n"
      "mov x9, #0x0\n"
      "ldr w28, [%x[args], %[offsetof_N]]\n"
      "ldr x27, [%x[args], %[offsetof_A]]\n"
      "3:"  // M loop
      "4:"  // N loop
      "mov x26, x27\n"
	  "whilelt p4.s,  x9, x28\n"
	  "incw x9\n"         
	  "whilelt p5.s,  x9, x28\n"
	  "incw x9\n"         
	  "whilelt p6.s,  x9, x28\n"
	  "incw x9\n"         
	  "whilelt p7.s,  x9, x28\n"
	  "decw x9\n"
	  "decw x9\n"
	  "decw x9\n"
      "tbnz x15, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      "zero { za }\n"
      "cbz x20, 6f\n"
      "fmov z15.s, #1.0\n"
	  "add x21, x20, x9, LSL #2\n"
      "ld1w_4p  z0.s, z4.s, z8.s, z12.s , p4,p5,p6,p7, x21\n"      
      ".inst 0x808001e0  // fmopa za0.s, p0/M, p0/M, z15.s, z0.s\n"
      ".inst 0x808401e1  // fmopa za1.s, p0/M, p0/M, z15.s, z4.s\n"
      ".inst 0x808801e2  // fmopa za2.s, p0/M, p0/M, z15.s, z8.s\n"
      ".inst 0x808c01e3  // fmopa za3.s, p0/M, p0/M, z15.s, z12.s\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x9\n"
      "mov x21, x10\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x28\n"
      "mov x20, x15\n"
      "csel x21, x10, x21, LT\n"
      "bfm x15, XZR, #0x0, #0x0  // bfc x15, #0x0, #0x1\n"
      "cmp x21, x11\n"
      "csel x15, x20, x15, LT\n"
      "6:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "ldr x23, [%x[args], %[offsetof_B]]\n"
      "ldr x22, [%x[args], %[offsetof_kstride_bytes]]\n"
      "add x20, x20, #0x1\n"
      "lsr x20, x20, #0x1\n"
      "lsr x21, x20, #0x2\n"
      "madd x23, x9, x22, x23\n"  // bptr = B + n * kstride_bytes
      "and x20, x20, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      "ld1h { z20.h }, p0/Z, [x26]\n"
      "ld1h_4  z19.h, z23.h, z27.h, z31.h , p0, x23 \n"
      "ld1h { z4.h }, p0/Z, [x26, #1, MUL VL]\n"
      "ld1h_4  z12.h, z13.h, z14.h, z15.h,  p0, x23 \n"
      "ld1h { z29.h }, p0/Z, [x26, #2, MUL VL]\n"
      "ld1h_4  z18.h, z22.h, z26.h, z30.h , p0, x23 \n"
      "ld1h { z2.h }, p0/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      "ld1h_4  z8.h, z9.h, z10.h, z11.h   , p0, x23 \n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81b30280  // fmopa za0.s, p0/M, p0/M, z20.h, z19.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81b70281  // fmopa za1.s, p0/M, p0/M, z20.h, z23.h\n"
      ".inst 0x81bb0282  // fmopa za2.s, p0/M, p0/M, z20.h, z27.h\n"
      ".inst 0x81bf0283  // fmopa za3.s, p0/M, p0/M, z20.h, z31.h\n"
      "ld1h { z20.h }, p0/Z, [x26]\n"
      ".inst 0x81ac0080  // fmopa za0.s, p0/M, p0/M, z4.h, z12.h\n"
      "ld1h_4  z19.h, z23.h, z27.h, z31.h , p0, x23 \n"
      ".inst 0x81ad0081  // fmopa za1.s, p0/M, p0/M, z4.h, z13.h\n"
      ".inst 0x81ae0082  // fmopa za2.s, p0/M, p0/M, z4.h, z14.h\n"
      ".inst 0x81af0083  // fmopa za3.s, p0/M, p0/M, z4.h, z15.h\n"
      "ld1h { z4.h }, p0/Z, [x26, #1, MUL VL]\n"
      ".inst 0x81b203a0  // fmopa za0.s, p0/M, p0/M, z29.h, z18.h\n"
      "ld1h_4  z12.h, z13.h, z14.h, z15.h,  p0, x23 \n"
      ".inst 0x81b603a1  // fmopa za1.s, p0/M, p0/M, z29.h, z22.h\n"
      ".inst 0x81ba03a2  // fmopa za2.s, p0/M, p0/M, z29.h, z26.h\n"
      ".inst 0x81be03a3  // fmopa za3.s, p0/M, p0/M, z29.h, z30.h\n"
      "ld1h { z29.h }, p0/Z, [x26, #2, MUL VL]\n"
      "ld1h_4  z18.h, z22.h, z26.h, z30.h , p0, x23 \n"
      ".inst 0x81a80040  // fmopa za0.s, p0/M, p0/M, z2.h, z8.h\n"
      ".inst 0x81a90041  // fmopa za1.s, p0/M, p0/M, z2.h, z9.h\n"
      ".inst 0x81aa0042  // fmopa za2.s, p0/M, p0/M, z2.h, z10.h\n"
      ".inst 0x81ab0043  // fmopa za3.s, p0/M, p0/M, z2.h, z11.h\n"
      "ld1h { z2.h }, p0/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      "ld1h_4  z8.h, z9.h, z10.h, z11.h   , p0, x23 \n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81b30280  // fmopa za0.s, p0/M, p0/M, z20.h, z19.h\n"
      ".inst 0x81b70281  // fmopa za1.s, p0/M, p0/M, z20.h, z23.h\n"
      ".inst 0x81bb0282  // fmopa za2.s, p0/M, p0/M, z20.h, z27.h\n"
      ".inst 0x81bf0283  // fmopa za3.s, p0/M, p0/M, z20.h, z31.h\n"
      ".inst 0x81ac0080  // fmopa za0.s, p0/M, p0/M, z4.h, z12.h\n"
      ".inst 0x81ad0081  // fmopa za1.s, p0/M, p0/M, z4.h, z13.h\n"
      ".inst 0x81ae0082  // fmopa za2.s, p0/M, p0/M, z4.h, z14.h\n"
      ".inst 0x81af0083  // fmopa za3.s, p0/M, p0/M, z4.h, z15.h\n"
      ".inst 0x81b203a0  // fmopa za0.s, p0/M, p0/M, z29.h, z18.h\n"
      ".inst 0x81b603a1  // fmopa za1.s, p0/M, p0/M, z29.h, z22.h\n"
      ".inst 0x81ba03a2  // fmopa za2.s, p0/M, p0/M, z29.h, z26.h\n"
      ".inst 0x81be03a3  // fmopa za3.s, p0/M, p0/M, z29.h, z30.h\n"
      ".inst 0x81a80040  // fmopa za0.s, p0/M, p0/M, z2.h, z8.h\n"
      ".inst 0x81a90041  // fmopa za1.s, p0/M, p0/M, z2.h, z9.h\n"
      ".inst 0x81aa0042  // fmopa za2.s, p0/M, p0/M, z2.h, z10.h\n"
      ".inst 0x81ab0043  // fmopa za3.s, p0/M, p0/M, z2.h, z11.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1h { z26.h }, p0/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #1\n"
      "ld1h_4  z3.h, z7.h, z11.h, z15.h , p0, x23 \n"

      ".inst 0x81a30340  // fmopa za0.s, p0/M, p0/M, z26.h, z3.h\n"
      ".inst 0x81a70341  // fmopa za1.s, p0/M, p0/M, z26.h, z7.h\n"
      ".inst 0x81ab0342  // fmopa za2.s, p0/M, p0/M, z26.h, z11.h\n"
      ".inst 0x81af0343  // fmopa za3.s, p0/M, p0/M, z26.h, z15.h\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x15, #1, 15f\n"
      "tbz x15, #0, 13f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      "ld1w_4 z24.s, z25.s, z26.s, z27.s, p0, x14   \n"
      "move_tile_vector z8.s, z9.s, z10.s, z11.s  , za0h.s, w12, p0 \n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za1h.s, w12, p0 \n"
      "ld1w_4 z4.s, z5.s, z6.s, z7.s    , p0, x14   \n"
      "move_tile_vector z0.s, z1.s, z2.s, z3.s    , za2h.s, w12, p0 \n"
      "move_tile_vector z28.s, z29.s, z30.s, z31.s, za3h.s, w12, p0 \n"
      "ld1w_4 z20.s, z21.s, z22.s, z23.s, p0, x14   \n"
      "ld1w_4 z12.s, z13.s, z14.s, z15.s, p0, x14   \n"
      "move_vector_tile za0h.s, z24.s, z25.s, z26.s, z27.s, w12, p0 \n"
      "move_vector_tile za1h.s, z4.s, z5.s, z6.s, z7.s    , w12, p0 \n"
      "st1w_4 z8.s, z9.s, z10.s, z11.s  , p0,  x13  \n"
      "move_vector_tile za2h.s, z20.s, z21.s, z22.s, z23.s, w12, p0 \n"
      "st1w_4 z16.s, z17.s, z18.s, z19.s, p0,  x13  \n"
      "move_vector_tile za3h.s, z12.s, z13.s, z14.s, z15.s, w12, p0 \n"
      "add x12, x12, #0x4\n"
      "st1w_4 z0.s, z1.s, z2.s, z3.s    , p0,  x13  \n"
      "cmp x12, x20\n"
      "st1w_4 z28.s, z29.s, z30.s, z31.s, p0,  x13  \n"	 
      "blt 12b\n"
      "b 25f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za0h.s, w12, p0 \n"
      "move_tile_vector z4.s, z5.s, z6.s, z7.s    , za1h.s, w12, p0 \n"
      "move_tile_vector z8.s, z9.s, z10.s, z11.s  , za2h.s, w12, p0 \n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za3h.s, w12, p0 \n"	  
      "st1w_4 z12.s, z13.s, z14.s, z15.s, p0,  x13  \n"
      "add x12, x12, #0x4\n"
      "st1w_4 z4.s, z5.s, z6.s, z7.s    , p0,  x13  \n"
      "cmp x12, x20\n"
      "st1w_4 z8.s, z9.s, z10.s, z11.s  , p0,  x13  \n"
      "st1w_4 z16.s, z17.s, z18.s, z19.s, p0,  x13  \n"	
      "blt 14b\n"
      "b 25f\n"
      "15:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x11, x10\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "add x25, x25, x9, LSL #2\n"  // C += n
      "madd x25, x10, x23, x25\n"  // C += m * ldc
      "tbz x15, #2, 19f\n"
      "cntw x20\n"
      "mov x12, #0x0\n"
      "cmp x24, x20\n"
      "csel x22, x24, x20, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
      "move_tile_vector z0.s, z1.s, z2.s, z3.s    , za0h.s, w12, p0 \n"
      "move_tile_vector z4.s, z5.s, z6.s, z7.s    , za1h.s, w12, p0 \n"
      "move_tile_vector z8.s, z9.s, z10.s, z11.s  , za2h.s, w12, p0 \n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p0 \n"  
      "st1w_4pd z0.s , z4.s,z8.s  ,z12.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "add x12, x12, #0x4\n"
      "st1w_4pd z1.s , z5.s,z9.s  ,z13.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "cmp x12, x21, LSL #2\n"
      "st1w_4pd z2.s , z6.s,z10.s ,z14.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "st1w_4pd z3.s , z7.s,z11.s ,z15.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      "move_tile_vector z0.s, z1.s, z2.s, z3.s    , za0h.s, w12, p0 \n"
      "move_tile_vector z4.s, z5.s, z6.s, z7.s    , za1h.s, w12, p0 \n"
      "move_tile_vector z8.s, z9.s, z10.s, z11.s  , za2h.s, w12, p0 \n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p0 \n"  
      "st1w_4pd z0.s , z4.s,z8.s  ,z12.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1w_4pd z1.s , z5.s,z9.s  ,z13.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      "st1w_4pd z2.s , z6.s,z10.s ,z14.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "18:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x24, x24, x22\n"
      "beq 19f\n"
      "b 23f\n"
      "19:"  // Store to output array: Skip activation: End
      "cntw x20\n"
      "ld1rw { z1.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0x0\n"
      "cmp x24, x20\n"
      "ld1rw { z0.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x20, x24, x20, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 21f\n"
      "20:"  // Store to output array: Accumulator row 0 loop
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p0 \n"
      "move_tile_vector z20.s, z21.s, z22.s, z23.s, za1h.s, w12, p0 \n"
      "move_tile_vector z24.s, z25.s, z26.s, z27.s, za2h.s, w12, p0 \n"
      "move_tile_vector z28.s, z29.s, z30.s, z31.s, za3h.s, w12, p0 \n" 
      "clamp_float_4 z16.s, z17.s, z18.s, z19.s, z1.s, z0.s, p0\n"
      "clamp_float_4 z20.s, z21.s, z22.s, z23.s, z1.s, z0.s, p0\n"
      "clamp_float_4 z24.s, z25.s, z26.s, z27.s, z1.s, z0.s, p0\n"
      "clamp_float_4 z28.s, z29.s, z30.s, z31.s, z1.s, z0.s, p0\n"	
      "add x12, x12, #0x4\n"
      "cmp x12, x21, LSL #2\n"
      "st1w_4pd z16.s, z20.s,z24.s,z28.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "st1w_4pd z17.s, z21.s,z25.s,z29.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "st1w_4pd z18.s, z22.s,z26.s,z30.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "st1w_4pd z19.s, z23.s,z27.s,z31.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "blt 20b\n"
      "21:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 22f\n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p0 \n"
      "move_tile_vector z20.s, z21.s, z22.s, z23.s, za1h.s, w12, p0 \n"
      "move_tile_vector z24.s, z25.s, z26.s, z27.s, za2h.s, w12, p0 \n"
      "move_tile_vector z28.s, z29.s, z30.s, z31.s, za3h.s, w12, p0 \n" 
      "clamp_float_4 z16.s, z17.s, z18.s, z19.s, z1.s, z0.s, p0\n"
      "clamp_float_4 z20.s, z21.s, z22.s, z23.s, z1.s, z0.s, p0\n"
      "subs x20, x20, #0x1\n"
      "clamp_float_4 z24.s, z25.s, z26.s, z27.s, z1.s, z0.s, p0\n"
      "clamp_float_4 z28.s, z29.s, z30.s, z31.s, z1.s, z0.s, p0\n"	  
      "st1w_4pd z16.s, z20.s,z24.s,z28.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "beq 22f\n"
      "subs x20, x20, #0x1\n"
      "st1w_4pd z17.s, z21.s,z25.s,z29.s, p4,p5,p6,p7,  x25\n"
      "add x25, x25, x23\n"
      "beq 22f\n"
      "st1w_4pd z18.s, z22.s,z26.s,z30.s, p4,p5,p6,p7,  x25\n"
      "22:"  // Store to output array: Accumulator row 0 oddments: End
      "23:"  // Store to output array: End
      "tbz x15, #0, 25f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "24:"  // Store to output array: Refill accumulators: Loop
      "ld1w_4 z0.s, z1.s, z2.s, z3.s    , p0, x14 \n"  
      "ld1w_4 z12.s, z13.s, z14.s, z15.s, p0, x14 \n"  
      "ld1w_4 z28.s, z29.s, z30.s, z31.s, p0, x14 \n"  
      "ld1w_4 z4.s, z5.s, z6.s, z7.s    , p0, x14 \n"  
      "move_vector_tile za0h.s, z0.s, z1.s, z2.s, z3.s    , w12, p0 \n"
      "move_vector_tile za1h.s, z12.s, z13.s, z14.s, z15.s, w12, p0 \n"
      "move_vector_tile za2h.s, z28.s, z29.s, z30.s, z31.s, w12, p0 \n"
      "move_vector_tile za3h.s, z4.s, z5.s, z6.s, z7.s    , w12, p0 \n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 24b\n"
      "25:"  // End block
      "incw x9, ALL, MUL #4\n"
      "cmp x9, x28\n"
      "blt 4b\n"
      "incw x10\n"
      "mov x9, #0x0\n"
      "cmp x10, x11\n"
      "mov x27, x26\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME
