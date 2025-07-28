/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2025-04-25 18:28:12
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2025-04-25 18:28:12
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#pragma once
#include <cstdint>
#include <immintrin.h>


template <typename T>
T* offset_pointer(T* ptr, std::size_t byte_offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + byte_offset);
}

template <typename T>
const T* offset_pointer(const T* ptr, std::size_t byte_offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) + byte_offset);
}

template <typename T>
T* offset_pointer_row_major(T* t, int row, int col, std::size_t ld) {
  return offset_pointer(t, row * ld) + col;
}

template <typename T>
T* offset_pointer_col_major(T* t, int row, int col, std::size_t ld) {
  return offset_pointer(t, col * ld) + row;
}

static inline void avx512_copy_32xbf16(__m512i* src, __m512i* dst) {
  _mm512_storeu_si512(dst, _mm512_loadu_si512(src));
}

static inline void avx512_32xfp32_to_32xbf16(__m512* src0, __m512* src1, __m512i* dst) {
#ifdef __AVX512BF16__
  _mm512_storeu_si512(dst, __m512i(_mm512_cvtne2ps_pbh(*src1, *src0)));
#else
  // Fallback implementation without AVX512BF16
  // Convert 32 FP32 values to 32 BF16 values by truncating mantissa
  
  // Process src0 (first 16 FP32 -> first 16 BF16)
  __m512i src0_int = _mm512_castps_si512(*src0);
  // Round by adding 0x7FFF to account for truncation
  __m512i rounded0 = _mm512_add_epi32(src0_int, _mm512_set1_epi32(0x7FFF));
  // Extract upper 16 bits
  __m512i bf16_low = _mm512_srli_epi32(rounded0, 16);
  
  // Process src1 (next 16 FP32 -> next 16 BF16)
  __m512i src1_int = _mm512_castps_si512(*src1);
  __m512i rounded1 = _mm512_add_epi32(src1_int, _mm512_set1_epi32(0x7FFF));
  __m512i bf16_high = _mm512_srli_epi32(rounded1, 16);
  
  // Pack the results using AVX512BW instructions
  // Convert 32-bit integers to 16-bit integers and pack
  __m256i low_packed = _mm512_cvtepi32_epi16(bf16_low);
  __m256i high_packed = _mm512_cvtepi32_epi16(bf16_high);
  
  // Combine into a single 512-bit register
  __m512i result = _mm512_castsi256_si512(low_packed);
  result = _mm512_inserti64x4(result, high_packed, 1);
  
  _mm512_storeu_si512(dst, result);
#endif
}

static inline void avx512_32xbf16_to_32xfp32(__m512i* src, __m512* dst0, __m512* dst1) {
  _mm512_storeu_ps(dst0, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src))), 16)));
  _mm512_storeu_ps(dst1, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src) + 1)), 16)));
}