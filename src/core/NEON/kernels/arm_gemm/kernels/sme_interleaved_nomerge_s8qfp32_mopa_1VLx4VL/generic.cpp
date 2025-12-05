/*
 * Copyright (c) 2024 Arm Limited.
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

void sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, float *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const DequantizeFloat &dq, const float *const late_bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,
      const DequantizeFloat &, const float *const late_bias, const Activation act,
      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        min(-std::numeric_limits<float>::infinity()),
        max(std::numeric_limits<float>::infinity()),
        bias(bias), late_bias(late_bias),
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

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    float *const C;
    const long ldcb;
    const long M, N, K;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const int32_t *const bias;
    const float *const late_bias;

    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, dq, late_bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x14, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART \n"
      "ptrue p0.b\n"
      "ptrue p1.b\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x11, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x14, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      "ld1w_4 z28.s, z29.s, z30.s, z31.s, p1, x13\n"
      "ld1w_4 z0.s,  z1.s,  z2.s,  z3.s , p1, x13\n"
      "ld1w_4 z24.s, z25.s, z26.s, z27.s, p1, x13\n"
      "ld1w_4 z12.s, z13.s, z14.s, z15.s, p1, x13\n"
      "move_vector_tile za0h.s, z28.s, z29.s, z30.s, z31.s, w12, p1\n"
      "move_vector_tile za1h.s, z0.s,  z1.s,  z2.s,  z3.s , w12, p1\n"
      "move_vector_tile za2h.s, z24.s, z25.s, z26.s, z27.s, w12, p1\n"
      "move_vector_tile za3h.s, z12.s, z13.s, z14.s, z15.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w10, [%x[args], %[offsetof_M]]\n"
      "mov x9, #0x0\n"
      "mov x28, #0x0\n"
      "ldr w27, [%x[args], %[offsetof_N]]\n"
      "ldr x26, [%x[args], %[offsetof_A]]\n"
      "3:"  // M loop
      "ldr x25, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x24, x26\n"
	  "whilelt p4.s, x28, x27\n"
	  "incw x28\n"
	  "whilelt p5.s, x28, x27\n"
	  "incw x28\n"
	  "whilelt p6.s, x28, x27\n"
	  "incw x28\n"
	  "whilelt p7.s, x28, x27\n"
	  "decw x28\n"
	  "decw x28\n"
	  "decw x28\n"    
      "tbnz x14, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      "zero { za }\n"
      "cbz x20, 6f\n"
      "add x21, x20, x28, LSL #2 \n"
      

      "ld1w_4p  z8.s,  z9.s,  z10.s, z11.s , p4,p5,p6,p7, x21 \n"
      ".inst 0xc0900100  // addha za0.s, p0/M, p0/M, z8.s\n"
      ".inst 0xc0900121  // addha za1.s, p0/M, p0/M, z9.s\n"
      ".inst 0xc0900142  // addha za2.s, p0/M, p0/M, z10.s\n"
      ".inst 0xc0900163  // addha za3.s, p0/M, p0/M, z11.s\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x28\n"
      "mov x21, x9\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x27\n"
      "mov x20, x14\n"
      "csel x21, x9, x21, LT\n"
      "bfm x14, XZR, #0x0, #0x0  // bfc x14, #0x0, #0x1\n"
      "cmp x21, x10\n"
      "csel x14, x20, x14, LT\n"
      "6:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "add x20, x20, #0x3\n"
      "lsr x20, x20, #0x2\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      "ld1b { z31.b }, p0/Z, [x24]\n"
      "ld1b_4  z8.b,  z9.b,  z10.b, z11.b, p1, x25  \n"
      "ld1b { z1.b }, p0/Z, [x24, #1, MUL VL]\n"
      "ld1b_4  z4.b,  z5.b,  z6.b,  z7.b , p1, x25  \n"
      "ld1b { z0.b }, p0/Z, [x24, #2, MUL VL]\n"
      "ld1b_4  z12.b, z13.b, z14.b, z15.b, p1, x25  \n"
      "ld1b { z3.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      "ld1b_4  z16.b, z17.b, z18.b, z19.b, p1, x25  \n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0xa08803e0  // smopa za0.s, p0/M, p0/M, z31.b, z8.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa08903e1  // smopa za1.s, p0/M, p0/M, z31.b, z9.b\n"
      ".inst 0xa08a03e2  // smopa za2.s, p0/M, p0/M, z31.b, z10.b\n"
      ".inst 0xa08b03e3  // smopa za3.s, p0/M, p0/M, z31.b, z11.b\n"
      "ld1b { z31.b }, p0/Z, [x24]\n"
      ".inst 0xa0840020  // smopa za0.s, p0/M, p0/M, z1.b, z4.b\n"
      "ld1b_4  z8.b,  z9.b,  z10.b, z11.b, p1, x25  \n"
      ".inst 0xa0850021  // smopa za1.s, p0/M, p0/M, z1.b, z5.b\n"
      ".inst 0xa0860022  // smopa za2.s, p0/M, p0/M, z1.b, z6.b\n"
      ".inst 0xa0870023  // smopa za3.s, p0/M, p0/M, z1.b, z7.b\n"
      "ld1b { z1.b }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa08c0000  // smopa za0.s, p0/M, p0/M, z0.b, z12.b\n"
      "ld1b_4  z4.b,  z5.b,  z6.b,  z7.b , p1, x25  \n"
      ".inst 0xa08d0001  // smopa za1.s, p0/M, p0/M, z0.b, z13.b\n"
      ".inst 0xa08e0002  // smopa za2.s, p0/M, p0/M, z0.b, z14.b\n"
      ".inst 0xa08f0003  // smopa za3.s, p0/M, p0/M, z0.b, z15.b\n"
      "ld1b { z0.b }, p0/Z, [x24, #2, MUL VL]\n"
      "ld1b_4  z12.b, z13.b, z14.b, z15.b, p1, x25  \n"
      ".inst 0xa0900060  // smopa za0.s, p0/M, p0/M, z3.b, z16.b\n"
      ".inst 0xa0910061  // smopa za1.s, p0/M, p0/M, z3.b, z17.b\n"
      ".inst 0xa0920062  // smopa za2.s, p0/M, p0/M, z3.b, z18.b\n"
      ".inst 0xa0930063  // smopa za3.s, p0/M, p0/M, z3.b, z19.b\n"
      "ld1b { z3.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      "ld1b_4  z16.b, z17.b, z18.b, z19.b, p1, x25  \n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0xa08803e0  // smopa za0.s, p0/M, p0/M, z31.b, z8.b\n"
      ".inst 0xa08903e1  // smopa za1.s, p0/M, p0/M, z31.b, z9.b\n"
      ".inst 0xa08a03e2  // smopa za2.s, p0/M, p0/M, z31.b, z10.b\n"
      ".inst 0xa08b03e3  // smopa za3.s, p0/M, p0/M, z31.b, z11.b\n"
      ".inst 0xa0840020  // smopa za0.s, p0/M, p0/M, z1.b, z4.b\n"
      ".inst 0xa0850021  // smopa za1.s, p0/M, p0/M, z1.b, z5.b\n"
      ".inst 0xa0860022  // smopa za2.s, p0/M, p0/M, z1.b, z6.b\n"
      ".inst 0xa0870023  // smopa za3.s, p0/M, p0/M, z1.b, z7.b\n"
      ".inst 0xa08c0000  // smopa za0.s, p0/M, p0/M, z0.b, z12.b\n"
      ".inst 0xa08d0001  // smopa za1.s, p0/M, p0/M, z0.b, z13.b\n"
      ".inst 0xa08e0002  // smopa za2.s, p0/M, p0/M, z0.b, z14.b\n"
      ".inst 0xa08f0003  // smopa za3.s, p0/M, p0/M, z0.b, z15.b\n"
      ".inst 0xa0900060  // smopa za0.s, p0/M, p0/M, z3.b, z16.b\n"
      ".inst 0xa0910061  // smopa za1.s, p0/M, p0/M, z3.b, z17.b\n"
      ".inst 0xa0920062  // smopa za2.s, p0/M, p0/M, z3.b, z18.b\n"
      ".inst 0xa0930063  // smopa za3.s, p0/M, p0/M, z3.b, z19.b\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      "ld1b { z18.b }, p0/Z, [x24]\n"
      "subs x20, x20, #0x1\n"
      "addvl x24, x24, #1\n"
      "ld1b_4  z28.b, z29.b, z30.b, z31.b, p1, x25  \n"

      ".inst 0xa09c0240  // smopa za0.s, p0/M, p0/M, z18.b, z28.b\n"
      ".inst 0xa09d0241  // smopa za1.s, p0/M, p0/M, z18.b, z29.b\n"
      ".inst 0xa09e0242  // smopa za2.s, p0/M, p0/M, z18.b, z30.b\n"
      ".inst 0xa09f0243  // smopa za3.s, p0/M, p0/M, z18.b, z31.b\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x14, #1, 15f\n"
      "tbz x14, #0, 13f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      "ld1w_4 z0.s,  z1.s,  z2.s,  z3.s , p1, x13   \n"
      "move_tile_vector z8.s,  z9.s,  z10.s, z11.s, za0h.s, w12, p1 \n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za1h.s, w12, p1 \n"
      "ld1w_4 z28.s, z29.s, z30.s, z31.s, p1, x13   \n"
      "move_tile_vector z4.s,  z5.s,  z6.s,  z7.s , za2h.s, w12, p1 \n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za3h.s, w12, p1 \n"
      "ld1w_4 z24.s, z25.s, z26.s, z27.s, p1, x13   \n"
      "ld1w_4 z20.s, z21.s, z22.s, z23.s, p1, x13   \n"
      "move_vector_tile za0h.s, z0.s,  z1.s,  z2.s,  z3.s , w12, p1 \n"
      "move_vector_tile za1h.s, z28.s, z29.s, z30.s, z31.s, w12, p1 \n"
      "st1w_4 z8.s,  z9.s,  z10.s, z11.s, p1,  x11  \n"
      "move_vector_tile za2h.s, z24.s, z25.s, z26.s, z27.s, w12, p1 \n"
      "st1w_4 z12.s, z13.s, z14.s, z15.s, p1,  x11  \n"
      "move_vector_tile za3h.s, z20.s, z21.s, z22.s, z23.s, w12, p1 \n"
      "add x12, x12, #0x4\n"
      "st1w_4 z4.s,  z5.s,  z6.s,  z7.s , p1,  x11  \n"
      "cmp x12, x20\n"
      "st1w_4 z16.s, z17.s, z18.s, z19.s, p1,  x11  \n"	  
      "blt 12b\n"
      "b 22f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      "move_tile_vector z4.s,  z5.s,  z6.s,  z7.s , za0h.s, w12, p1 \n"
      "move_tile_vector z16.s, z17.s, z18.s, z19.s, za1h.s, w12, p1 \n"
      "move_tile_vector z8.s,  z9.s,  z10.s, z11.s, za2h.s, w12, p1 \n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p1 \n"	  
      "st1w_4 z4.s,  z5.s,  z6.s,  z7.s , p1,  x11  \n"
      "add x12, x12, #0x4\n"
      "st1w_4 z16.s, z17.s, z18.s, z19.s, p1,  x11  \n"
      "cmp x12, x20\n"
      "st1w_4 z8.s,  z9.s,  z10.s, z11.s, p1,  x11  \n"
      "st1w_4 z12.s, z13.s, z14.s, z15.s, p1,  x11  \n"	
      "addvl x11, x11, #16\n"
      "blt 14b\n"
      "b 22f\n"
      "15:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x21, x10, x9\n"
      "ld1rw { z18.s }, p0/Z, [%x[dq], %[offset_DequantizeFloat_scale]]\n"
      "mov z20.s, #0x0\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "mov z21.s, #0x0\n"
      "mov z22.s, #0x0\n"
      "ldr x20, [%x[args], %[offsetof_late_bias]]\n"
      "mov z23.s, #0x0\n"
      "add x23, x23, x28, LSL #2\n"  // C += n
      "madd x23, x9, x22, x23\n"  // C += m * ldc
      "cbz x20, 16f\n"
      "add x20, x20, x28, LSL #2\n"
      "ld1w_4p  z20.s, z21.s, z22.s, z23.s, p4,p5,p6,p7, x20 \n"
      "16:"  // Store to output array: no late bias
      "cntw x20\n"
      "ld1rw { z17.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0x0\n"
      "cmp x21, x20\n"
      "ld1rw { z16.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x20, x21, x20, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 18f\n"
      "17:"  // Store to output array: Accumulator row 0 loop
      "move_tile_vector z0.s,  z1.s,  z2.s,  z3.s , za0h.s, w12, p1 \n"
      "move_tile_vector z4.s,  z5.s,  z6.s,  z7.s , za1h.s, w12, p1 \n"
      "move_tile_vector z8.s,  z9.s,  z10.s, z11.s, za2h.s, w12, p1 \n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p1 \n"
      "scvtf_convert    z0.s,  z1.s,  z2.s,  z3.s ,  p1 \n"
      "scvtf_convert    z4.s,  z5.s,  z6.s,  z7.s ,  p1 \n"
      "scvtf_convert    z8.s,  z9.s,  z10.s, z11.s,  p1 \n"
      "scvtf_convert    z12.s, z13.s, z14.s, z15.s,  p1 \n"
      "fmad z0.s, p0/M, z18.s, z20.s\n"
      "fmad z1.s, p0/M, z18.s, z20.s\n"
      "fmad z2.s, p0/M, z18.s, z20.s\n"
      "fmad z3.s, p0/M, z18.s, z20.s\n"
      "add x12, x12, #0x4\n"
      "fmad z4.s, p0/M, z18.s, z21.s\n"
      "fmad z5.s, p0/M, z18.s, z21.s\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z6.s, p0/M, z18.s, z21.s\n"
      "fmad z7.s, p0/M, z18.s, z21.s\n"
      "fmad z8.s, p0/M, z18.s, z22.s\n"
      "fmad z9.s, p0/M, z18.s, z22.s\n"
      "fmad z10.s, p0/M, z18.s, z22.s\n"
      "fmad z11.s, p0/M, z18.s, z22.s\n"
      "fmad z12.s, p0/M, z18.s, z23.s\n"
      "fmad z13.s, p0/M, z18.s, z23.s\n"
      "fmad z14.s, p0/M, z18.s, z23.s\n"
      "fmad z15.s, p0/M, z18.s, z23.s\n"
      "clamp_float_4    z0.s,  z1.s,  z2.s,  z3.s , z17.s, z16.s, p1  \n"
      "clamp_float_4    z4.s,  z5.s,  z6.s,  z7.s , z17.s, z16.s, p1  \n"
      "clamp_float_4    z8.s,  z9.s,  z10.s, z11.s, z17.s, z16.s, p1  \n"
      "clamp_float_4    z12.s, z13.s, z14.s, z15.s, z17.s, z16.s, p1  \n"
      "st1w_4pd  z0.s, z4.s, z8.s, z12.s  ,p4,p5,p6,p7,  x23\n"
      "add x23, x23, x22\n"
      "st1w_4pd  z1.s, z5.s, z9.s, z13.s  ,p4,p5,p6,p7,  x23\n"
      "add x23, x23, x22\n"
      "st1w_4pd  z2.s, z6.s, z10.s, z14.s ,p4,p5,p6,p7,  x23\n"
      "add x23, x23, x22\n"
      "st1w_4pd  z3.s, z7.s, z11.s, z15.s ,p4,p5,p6,p7,  x23\n"
      "add x23, x23, x22\n"
      "blt 17b\n"
      "18:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 19f\n"
      "move_tile_vector z0.s,  z1.s,  z2.s,  z3.s , za0h.s, w12, p1 \n"
      "move_tile_vector z4.s,  z5.s,  z6.s,  z7.s , za1h.s, w12, p1 \n"
      "move_tile_vector z8.s,  z9.s,  z10.s, z11.s, za2h.s, w12, p1 \n"
      "move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p1 \n"
      "scvtf_convert    z0.s,  z1.s,  z2.s,  z3.s ,  p1 \n"
      "scvtf_convert    z4.s,  z5.s,  z6.s,  z7.s ,  p1 \n"
      "scvtf_convert    z8.s,  z9.s,  z10.s, z11.s,  p1 \n"
      "scvtf_convert    z12.s, z13.s, z14.s, z15.s,  p1 \n"
      "fmad z0.s, p0/M, z18.s, z20.s\n"
      "fmad z1.s, p0/M, z18.s, z20.s\n"
      "fmad z2.s, p0/M, z18.s, z20.s\n"
      "fmad z3.s, p0/M, z18.s, z20.s\n"
      "subs x20, x20, #0x1\n"
      "fmad z4.s, p0/M, z18.s, z21.s\n"
      "fmad z5.s, p0/M, z18.s, z21.s\n"
      "fmad z6.s, p0/M, z18.s, z21.s\n"
      "fmad z7.s, p0/M, z18.s, z21.s\n"
      "fmad z8.s, p0/M, z18.s, z22.s\n"
      "fmad z9.s, p0/M, z18.s, z22.s\n"
      "fmad z10.s, p0/M, z18.s, z22.s\n"
      "fmad z11.s, p0/M, z18.s, z22.s\n"
      "fmad z12.s, p0/M, z18.s, z23.s\n"
      "fmad z13.s, p0/M, z18.s, z23.s\n"
      "fmad z14.s, p0/M, z18.s, z23.s\n"
      "fmad z15.s, p0/M, z18.s, z23.s\n"
      "clamp_float_4    z0.s,  z1.s,  z2.s,  z3.s , z17.s, z16.s, p1  \n"
      "clamp_float_4    z4.s,  z5.s,  z6.s,  z7.s , z17.s, z16.s, p1  \n"
      "clamp_float_4    z8.s,  z9.s,  z10.s, z11.s, z17.s, z16.s, p1  \n"
      "clamp_float_4    z12.s, z13.s, z14.s, z15.s, z17.s, z16.s, p1  \n"
      "st1w_4pd  z0.s, z4.s, z8.s, z12.s  ,p4,p5,p6,p7,  x23\n"
      "add x23, x23, x22\n"
      "beq 19f\n"
      "subs x20, x20, #0x1\n"
      "st1w_4pd  z1.s, z5.s, z9.s, z13.s  ,p4,p5,p6,p7,  x23\n"
      "add x23, x23, x22\n"
      "beq 19f\n"
      "st1w_4pd  z2.s, z6.s, z10.s, z14.s ,p4,p5,p6,p7,  x23\n"
      "19:"  // Store to output array: Accumulator row 0 oddments: End
      "tbz x14, #0, 22f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "21:"  // Store to output array: Refill accumulators: Loop

      "ld1w_4 z20.s, z21.s, z22.s, z23.s, p1, x13\n"
      "ld1w_4 z12.s, z13.s, z14.s, z15.s, p1, x13\n"
      "ld1w_4 z0.s,  z1.s,  z2.s,  z3.s , p1, x13\n"
      "ld1w_4 z8.s,  z9.s,  z10.s, z11.s, p1, x13\n"
      "move_vector_tile za0h.s, z20.s, z21.s, z22.s, z23.s, w12, p1\n"
      "move_vector_tile za1h.s, z12.s, z13.s, z14.s, z15.s, w12, p1\n"
      "move_vector_tile za2h.s, z0.s,  z1.s,  z2.s,  z3.s , w12, p1\n"
      "move_vector_tile za3h.s, z8.s,  z9.s,  z10.s, z11.s, w12, p1\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 21b \n"
      "22:"  // End block
      "incw x28, ALL, MUL #4\n"
      "cmp x28, x27\n"
      "blt 4b\n"
      "incw x9\n"
      "mov x28, #0x0\n"
      "cmp x9, x10\n"
      "mov x26, x24\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [dq] "r" (&dq), [offset_DequantizeFloat_scale] "I" (offsetof(DequantizeFloat, scale)), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_late_bias] "I" (offsetof(KernelArgs, late_bias)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME
