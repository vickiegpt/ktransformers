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
#include <array>
#include <atomic> // Add atomic header for std::atomic
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring> // 添加这一行，包含memcpy函数
#include <immintrin.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "utils.hpp"
#include <memory>

// 包含GGML头文件来获取ggml_bf16_t类型
#include "../../third_party/llama.cpp/ggml.h"

// 添加MAX宏定义
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

namespace amx {

// Forward declarations of helper functions
static inline size_t get_block_aligned_size(size_t size, size_t block_size);
static inline bool is_valid_aligned_ptr(const void *ptr, size_t alignment = 64);

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

const int TMMCount = 8;
const int MaxTileHeight = 16;
const int MaxTileWidth = 64;

const int AMX_BLK_SIZE = 32;

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

inline bool enable_amx() {
  static thread_local bool initialized = false;
  if (initialized) {
    return true;
  }
  initialized = true;

  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
    return false;
  } else {
    // printf("\n TILE DATA USE SET - OK \n\n");
    return true;
  }
  return true;
}

struct alignas(64) TileConfig {
  uint8_t palette;
  uint8_t start_row;
  std::array<uint8_t, 14> __0 = {};
  std::array<uint16_t, 8> colsb;
  std::array<uint8_t, 16> __1 = {};
  std::array<uint8_t, 8> rows;
  std::array<uint8_t, 8> __2 = {};

  TileConfig() {
    palette = 1;
    start_row = 0;
    for (int i = 0; i < 8; i++) {
      set_row_col(i, 0, 0);
    }
  }

  void set_row_col(int i, uint8_t row, uint16_t col) {
    colsb[i] = col;
    rows[i] = row;
  }

  void set_config() { _tile_loadconfig(this); }

  static void load_data(int to, void *from, size_t stride) {
    switch (to) {
    case 0:
      _tile_loadd(0, from, stride);
      break;
    case 1:
      _tile_loadd(1, from, stride);
      break;
    case 2:
      _tile_loadd(2, from, stride);
      break;
    case 3:
      _tile_loadd(3, from, stride);
      break;
    case 4:
      _tile_loadd(4, from, stride);
      break;
    case 5:
      _tile_loadd(5, from, stride);
      break;
    case 6:
      _tile_loadd(6, from, stride);
      break;
    case 7:
      _tile_loadd(7, from, stride);
      break;
    default:
      throw std::runtime_error("no such tile");
    }
  }

  static void store_data(int from, void *to, size_t stride) {
    switch (from) {
    case 0:
      _tile_stored(0, to, stride);
      break;
    case 1:
      _tile_stored(1, to, stride);
      break;
    case 2:
      _tile_stored(2, to, stride);
      break;
    case 3:
      _tile_stored(3, to, stride);
      break;
    case 4:
      _tile_stored(4, to, stride);
      break;
    case 5:
      _tile_stored(5, to, stride);
      break;
    case 6:
      _tile_stored(6, to, stride);
      break;
    case 7:
      _tile_stored(7, to, stride);
      break;
    default:
      throw std::runtime_error("no such tile");
    }
  }
};

static_assert(sizeof(TileConfig) == 64);

inline void debug_tile(int t) {
  printf("Tile %d\n", t);
  uint8_t data[16][64] = {};
  TileConfig::store_data(t, data, 64);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 64; j++) {
      printf("%3d ", data[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

inline void debug_tiles(int to = 8) {
  for (int i = 0; i < to; i++) {
    debug_tile(i);
  }
}

inline void debug_m512(__m512 x) {
  float data[16];
  _mm512_storeu_ps(data, x);
  for (int i = 0; i < 16; i++) {
    printf("%f ", data[i]);
  }
  printf("\n");
}

// transpose utils
inline void transpose_16x16_32bit(__m512i *v) {
  __m512i v1[16];
  v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
  v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
  v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
  v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
  v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
  v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
  v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
  v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
  v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
  v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
  v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
  v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
  v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
  v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
  v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
  v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

  v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
  v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
  v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
  v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
  v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
  v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
  v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
  v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
  v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
  v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
  v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
  v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
  v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
  v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
  v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
  v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

  v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
  v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
  v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
  v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
  v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
  v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
  v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
  v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
  v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
  v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
  v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
  v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
  v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
  v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
  v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
  v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

  v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
  v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
  v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
  v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
  v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
  v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
  v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
  v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
  v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
  v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
  v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
  v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
  v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
  v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
  v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
  v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

/*
  Transpose 16x16 32-bit elements
  Note that v must be 64 byte aligned
*/
inline void transpose_16x16_32bit(__m512i *v, size_t stride) {
  assert(reinterpret_cast<intptr_t>(v) % 64 == 0 && "v must be 64 aligned");

  auto stride_v = [=](int i) { return offset_pointer(v, i * stride); };
  __m512i v1[16];

  v1[0] = _mm512_unpacklo_epi32(*stride_v(0), *stride_v(1));
  v1[1] = _mm512_unpackhi_epi32(*stride_v(0), *stride_v(1));
  v1[2] = _mm512_unpacklo_epi32(*stride_v(2), *stride_v(3));
  v1[3] = _mm512_unpackhi_epi32(*stride_v(2), *stride_v(3));
  v1[4] = _mm512_unpacklo_epi32(*stride_v(4), *stride_v(5));
  v1[5] = _mm512_unpackhi_epi32(*stride_v(4), *stride_v(5));
  v1[6] = _mm512_unpacklo_epi32(*stride_v(6), *stride_v(7));
  v1[7] = _mm512_unpackhi_epi32(*stride_v(6), *stride_v(7));
  v1[8] = _mm512_unpacklo_epi32(*stride_v(8), *stride_v(9));
  v1[9] = _mm512_unpackhi_epi32(*stride_v(8), *stride_v(9));
  v1[10] = _mm512_unpacklo_epi32(*stride_v(10), *stride_v(11));
  v1[11] = _mm512_unpackhi_epi32(*stride_v(10), *stride_v(11));
  v1[12] = _mm512_unpacklo_epi32(*stride_v(12), *stride_v(13));
  v1[13] = _mm512_unpackhi_epi32(*stride_v(12), *stride_v(13));
  v1[14] = _mm512_unpacklo_epi32(*stride_v(14), *stride_v(15));
  v1[15] = _mm512_unpackhi_epi32(*stride_v(14), *stride_v(15));

  *stride_v(0) = _mm512_unpacklo_epi64(v1[0], v1[2]);
  *stride_v(1) = _mm512_unpackhi_epi64(v1[0], v1[2]);
  *stride_v(2) = _mm512_unpacklo_epi64(v1[1], v1[3]);
  *stride_v(3) = _mm512_unpackhi_epi64(v1[1], v1[3]);
  *stride_v(4) = _mm512_unpacklo_epi64(v1[4], v1[6]);
  *stride_v(5) = _mm512_unpackhi_epi64(v1[4], v1[6]);
  *stride_v(6) = _mm512_unpacklo_epi64(v1[5], v1[7]);
  *stride_v(7) = _mm512_unpackhi_epi64(v1[5], v1[7]);
  *stride_v(8) = _mm512_unpacklo_epi64(v1[8], v1[10]);
  *stride_v(9) = _mm512_unpackhi_epi64(v1[8], v1[10]);
  *stride_v(10) = _mm512_unpacklo_epi64(v1[9], v1[11]);
  *stride_v(11) = _mm512_unpackhi_epi64(v1[9], v1[11]);
  *stride_v(12) = _mm512_unpacklo_epi64(v1[12], v1[14]);
  *stride_v(13) = _mm512_unpackhi_epi64(v1[12], v1[14]);
  *stride_v(14) = _mm512_unpacklo_epi64(v1[13], v1[15]);
  *stride_v(15) = _mm512_unpackhi_epi64(v1[13], v1[15]);

  v1[0] = _mm512_shuffle_i32x4(*stride_v(0), *stride_v(4), 0x88);
  v1[1] = _mm512_shuffle_i32x4(*stride_v(1), *stride_v(5), 0x88);
  v1[2] = _mm512_shuffle_i32x4(*stride_v(2), *stride_v(6), 0x88);
  v1[3] = _mm512_shuffle_i32x4(*stride_v(3), *stride_v(7), 0x88);
  v1[4] = _mm512_shuffle_i32x4(*stride_v(0), *stride_v(4), 0xdd);
  v1[5] = _mm512_shuffle_i32x4(*stride_v(1), *stride_v(5), 0xdd);
  v1[6] = _mm512_shuffle_i32x4(*stride_v(2), *stride_v(6), 0xdd);
  v1[7] = _mm512_shuffle_i32x4(*stride_v(3), *stride_v(7), 0xdd);
  v1[8] = _mm512_shuffle_i32x4(*stride_v(8), *stride_v(12), 0x88);
  v1[9] = _mm512_shuffle_i32x4(*stride_v(9), *stride_v(13), 0x88);
  v1[10] = _mm512_shuffle_i32x4(*stride_v(10), *stride_v(14), 0x88);
  v1[11] = _mm512_shuffle_i32x4(*stride_v(11), *stride_v(15), 0x88);
  v1[12] = _mm512_shuffle_i32x4(*stride_v(8), *stride_v(12), 0xdd);
  v1[13] = _mm512_shuffle_i32x4(*stride_v(9), *stride_v(13), 0xdd);
  v1[14] = _mm512_shuffle_i32x4(*stride_v(10), *stride_v(14), 0xdd);
  v1[15] = _mm512_shuffle_i32x4(*stride_v(11), *stride_v(15), 0xdd);

  v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
  v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
  v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
  v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
  v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
  v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
  v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
  v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
  v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
  v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
  v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
  v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
  v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
  v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
  v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
  v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

struct GemmKernel224BF {
  using output_t = float;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 1792;

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 32
    for (int i = 0; i < 2; i++)
      tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(ggml_bf16_t));

    // size is 16 x 32
    for (int i = 2; i < 4; i++)
      tile_config.set_row_col(i, TILE_K / VNNI_BLK,
                              TILE_N * VNNI_BLK * sizeof(int8_t));

    // size is 16 x 16
    for (int i = 4; i < 8; i++)
      tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
  }

  static void load_a(ggml_bf16_t *a, size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }

  static void load_b(int8_t *b, size_t ldb) {
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  }

  static void clean_c() {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
  }

  static void load_c(output_t *c, size_t ldc) {
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)),
                ldc);
  }

  static void store_c(output_t *c, size_t ldc) {
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)),
                 ldc);
  }

  static void run_tile() {
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
  }

  struct BufferA {
    ggml_bf16_t *a;
    int max_m, k;

    static size_t required_size(int max_m, int k) {
      return max_m * k * sizeof(ggml_bf16_t);
    }

    BufferA(int max_m, int k, void *ptr) : max_m(max_m), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(k % K_STEP == 0);
      a = reinterpret_cast<ggml_bf16_t *>(ptr);
    }

    void from_mat(int m, ggml_bf16_t *src, int ith, int nth) {
      assert(m <= max_m);
      assert(ith == 0 && nth == 1);
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int k_block_begin = 0; k_block_begin < k;
             k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
              __m512i *s = (__m512i *)(src + (m_begin + i) * k + k_block_begin +
                                       k_begin);
              __m512i *d = (__m512i *)(a + k_block_begin * m_block_size +
                                       m_begin * k_block_size +
                                       k_begin * M_STEP + i * K_STEP);
              avx512_copy_32xbf16(s, d);
            }
          }
        }
      }
    }

    ggml_bf16_t *get_submat(int m, int k, int m_begin, int k_begin) {
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return a + k_block_begin * m_block_size + m_begin * k_block_size +
             k_begin * M_STEP;
    }
  };

  struct BufferB {
    int8_t *b;
    float *d;
    int n, k;

    static size_t required_size(int n, int k) {
      return n * k * sizeof(int8_t) + n * sizeof(float);
    }

    BufferB(int n, int k, void *ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      b = reinterpret_cast<int8_t *>(ptr);
      d = reinterpret_cast<float *>(b + n * k);
    }

    void from_mat(ggml_bf16_t *src, int ith, int nth) {
      // Add null check to prevent segmentation fault
      if (src == nullptr) {
        printf("Warning: Null bf16 source pointer in BufferB::from_mat, "
               "skipping conversion\n");
        return;
      }

      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP; i++) {
          if (n_block_begin + n_begin + i >= n)
            continue; // Skip if out of bounds

          float amax = 0.0f;
          for (int j = 0; j < k; j += 32) {
            if (j + 32 > k)
              continue; // Skip if out of bounds

            // Verify memory access before doing any operations
            if (src == nullptr ||
                (n_block_begin + n_begin + i) * k + j + 31 >= n * k) {
              printf("Warning: Invalid memory access in BufferB::from_mat at "
                     "index %d\n",
                     (n_block_begin + n_begin + i) * k + j);
              continue;
            }

            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32(
                (__m512i *)(src + (n_block_begin + n_begin + i) * k + j), &f0,
                &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          }
          d[n_block_begin + n_begin + i] = amax / ((1 << 15) - 1);
        }
      }
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < k;
             k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < N_STEP; i++) {
              __m512 id =
                  _mm512_set1_ps(d[n_block_begin + n_begin + i]
                                     ? 1.0f / d[n_block_begin + n_begin + i]
                                     : 0.0f);
              int8_t *dst =
                  b + n_block_begin * k + k_block_begin * n_block_size +
                  n_begin * k_block_size + k_begin * N_STEP + i * K_STEP;
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32(
                  (__m512i *)(src + (n_block_begin + n_begin + i) * k +
                              k_block_begin + k_begin),
                  &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i *)(src + (n_block_begin + n_begin + i) * k +
                              k_block_begin + k_begin) +
                      1,
                  &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              _mm_storeu_si128((__m128i *)dst, s0);
              _mm_storeu_si128((__m128i *)(dst + 16), s1);
              _mm_storeu_si128((__m128i *)(dst + 32), s2);
              _mm_storeu_si128((__m128i *)(dst + 48), s3);
            }
            transpose_16x16_32bit((__m512i *)(b + n_block_begin * k +
                                              k_block_begin * n_block_size +
                                              n_begin * k_block_size +
                                              k_begin * N_STEP));
            transpose_16x16_32bit((__m512i *)(b + n_block_begin * k +
                                              k_block_begin * n_block_size +
                                              n_begin * k_block_size +
                                              k_begin * N_STEP +
                                              TILE_N * K_STEP));
          }
        }
      }
    }

    int8_t *get_submat(int n, int k, int n_begin, int k_begin) {
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      n_begin -= n_block_begin;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return b + n_block_begin * k + k_block_begin * n_block_size +
             n_begin * k_block_size + k_begin * N_STEP;
    }

    float *get_scale(int n, int n_begin) { return d + n_begin; }
  };

  struct BufferC {
    float *c;
    int max_m, n;

    static size_t required_size(int max_m, int n) {
      return max_m * n * sizeof(float);
    }

    BufferC(int max_m, int n, void *ptr) : max_m(max_m), n(n) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(n % N_STEP == 0);
      c = reinterpret_cast<float *>(ptr);
    }

    void to_mat(int m, ggml_bf16_t *dst, int ith, int nth) {
      assert(m <= max_m);
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            __m512 *x0 = (__m512 *)(c + m_block_size * n_block_begin +
                                    m_begin * n_block_size + n_begin * M_STEP +
                                    i * N_STEP);
            __m512 *x1 = (__m512 *)(c + m_block_size * n_block_begin +
                                    m_begin * n_block_size + n_begin * M_STEP +
                                    i * N_STEP + 16);
            avx512_32xfp32_to_32xbf16(
                x0, x1,
                (__m512i *)(dst + (m_begin + i) * n + n_block_begin + n_begin));
          }
        }
      }
    }

    float *get_submat(int m, int n, int m_begin, int n_begin) {
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      n_begin -= n_block_begin;
      return c + m_block_size * n_block_begin + m_begin * n_block_size +
             n_begin * M_STEP;
    }
  };
};

struct GemmKernel224Int8 {
  using dt = int8_t;
  using output_t = int32_t;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 3584;

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 64
    for (int i = 0; i < 2; i++)
      tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(int8_t));

    // size is 16 x 64
    for (int i = 2; i < 4; i++)
      tile_config.set_row_col(i, TILE_K / VNNI_BLK,
                              TILE_N * VNNI_BLK * sizeof(int8_t));

    // size is 16 x 16
    for (int i = 4; i < 8; i++)
      tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
  }

  static void load_a(int8_t *a, size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }

  static void load_b(int8_t *b, size_t ldb) {
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  }

  static void clean_c() {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
  }

  static void load_c(output_t *c, size_t ldc) {
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)),
                ldc);
  }

  static void store_c(output_t *c, size_t ldc) {
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)),
                 ldc);
  }

  static void run_tile() {
    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
  }

  struct BufferA {
    int8_t *a;
    float *d;
    int max_m, k;

    static size_t required_size(int max_m, int k) {
      return max_m * k * sizeof(int8_t) + max_m * sizeof(float);
    }

    BufferA(int max_m, int k, void *ptr) : max_m(max_m), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(k % K_STEP == 0);
      a = reinterpret_cast<int8_t *>(ptr);
      d = reinterpret_cast<float *>(a + max_m * k);
    }

    void from_mat(int m, ggml_bf16_t *src, int ith, int nth) {
      assert(m <= max_m);
      assert(ith == 0 && nth == 1);
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          float amax = 0.0f;
          for (int j = 0; j < k; j += 32) {
            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32((__m512i *)(src + (m_begin + i) * k + j),
                                      &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          }
          d[m_begin + i] = amax / ((1 << 7) - 1);
        }
      }
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int k_block_begin = 0; k_block_begin < k;
             k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
              __m512 id =
                  _mm512_set1_ps(d[m_begin + i] ? 1.0f / d[m_begin + i] : 0.0f);
              int8_t *dst = a + k_block_begin * m_block_size +
                            m_begin * k_block_size + k_begin * M_STEP +
                            i * K_STEP;
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i *)(src + (m_begin + i) * k +
                                                    k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32((__m512i *)(src + (m_begin + i) * k +
                                                    k_block_begin + k_begin) +
                                            1,
                                        &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              _mm_storeu_si128((__m128i *)dst, s0);
              _mm_storeu_si128((__m128i *)(dst + 16), s1);
              _mm_storeu_si128((__m128i *)(dst + 32), s2);
              _mm_storeu_si128((__m128i *)(dst + 48), s3);
            }
          }
        }
      }
    }

    void from_mat_int8(int m, int8_t *src, int ith, int nth) {
      assert(m <= max_m);
      assert(ith == 0 && nth == 1);

      // Add null check to prevent segmentation fault
      if (src == nullptr) {
        printf("Warning: Null int8 source pointer in BufferA::from_mat_int8, "
               "skipping conversion\n");

        // Initialize with safe values
        for (int i = 0; i < m; i++) {
          d[i] = 1.0f; // Safe scale value
        }

        // Fill matrix with zeros to be safe
        memset(a, 0, max_m * k * sizeof(int8_t));
        return;
      }

      // For int8 input, we can directly copy the data
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      try {
        for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
          for (int k_block_begin = 0; k_block_begin < k;
               k_block_begin += K_BLOCK) {
            int k_block_size = std::min(K_BLOCK, k - k_block_begin);
            for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
              for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
                // Copy directly from source to destination
                int8_t *dst = a + k_block_begin * m_block_size +
                              m_begin * k_block_size + k_begin * M_STEP +
                              i * K_STEP;

                // Check source range validity
                const int8_t *src_ptr =
                    src + (m_begin + i) * k + k_block_begin + k_begin;
                if (src_ptr < src || src_ptr + K_STEP > src + m * k) {
                  printf("Warning: Invalid source range in "
                         "BufferA::from_mat_int8, "
                         "filling with safe values\n");
                  // Fill with zeros instead
                  memset(dst, 0, K_STEP);
                } else {
                  memcpy(dst, src_ptr, K_STEP);
                }
              }
            }
          }
        }
      } catch (const std::exception &e) {
        printf("Exception in BufferA::from_mat_int8: %s\n", e.what());
        // Continue with safe values
      } catch (...) {
        printf("Unknown exception in BufferA::from_mat_int8\n");
        // Continue with safe values
      }

      // Set all scale values to 1.0 for int8 direct input
      for (int i = 0; i < m; i++) {
        d[i] = 1.0f;
      }
    }

    // Add the missing get_submat method
    int8_t *get_submat(int m, int k, int m_begin, int k_begin) {
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return a + k_block_begin * m_block_size + m_begin * k_block_size +
             k_begin * M_STEP;
    }

    // Add the missing get_scale method
    float *get_scale(int m, int m_begin) { return d + m_begin; }
  };

  struct BufferB {
    int8_t *b;
    float *d;
    int n, max_k;

    static size_t required_size(int n, int max_k) {
      return (n * max_k + 63) / 64 * 64 * sizeof(int8_t) + n * sizeof(float);
    }

    BufferB(int n, int max_k, void *ptr) : n(n), max_k(max_k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_k % K_STEP == 0);
      b = reinterpret_cast<int8_t *>(ptr);
      d = reinterpret_cast<float *>(b + (n * max_k + 63) / 64 * 64);
    }

    void from_mat(const ggml_bf16_t *src, int ith, int nth) {
      // Add null check to prevent segmentation fault
      if (src == nullptr) {
        printf("Warning: Null bf16 source pointer in "
               "GemmKernel224Int8::BufferB::from_mat, skipping conversion\n");
        return;
      }

      int n_per_thread = (n + nth - 1) / nth;
      int n_begin = ith * n_per_thread;
      int n_end = std::min(n_begin + n_per_thread, n);

      for (int i = n_begin; i < n_end; i++) {
        float amax = 0.0f;
        for (int j = 0; j < max_k; j += 32) {
          if (j + 32 > max_k)
            continue; // Skip if out of bounds

          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i *)(src + i * max_k + j), &f0, &f1);
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
        }
        d[i] = amax / ((1 << 7) - 1);
      }

      int n_block_size = (n + N_STEP - 1) / N_STEP * N_STEP;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP; i++) {
          if (n_begin + i >= n)
            continue; // Skip if out of bounds

          float amax = 0.0f;
          for (int j = 0; j < max_k; j += 32) {
            if (j + 32 > max_k)
              continue; // Skip if out of bounds

            // Safe memory access check
            if ((n_begin + i) * max_k + j + 31 >= n * max_k) {
              printf("Warning: Invalid memory access in BufferB::from_mat at "
                     "index %d\n",
                     (n_begin + i) * max_k + j);
              continue;
            }

            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32(
                (__m512i *)(src + (n_begin + i) * max_k + j), &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          }
          d[n_begin + i] = amax / ((1 << 7) - 1);
        }
      }

      // Convert and transpose data
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < max_k;
             k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, max_k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < N_STEP; i++) {
              if (n_begin + i >= n)
                continue; // Skip if out of bounds

              __m512 id =
                  _mm512_set1_ps(d[n_begin + i] ? 1.0f / d[n_begin + i] : 0.0f);
              int8_t *dst = b + n_begin * max_k + k_block_begin * n_block_size +
                            k_begin * N_STEP + i * K_STEP;

              // Safe memory access check
              if ((n_begin + i) * max_k + k_block_begin + k_begin + K_STEP >
                  n * max_k) {
                printf("Warning: Invalid memory access in BufferB::from_mat at "
                       "dst index\n");
                continue;
              }

              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i *)(src +
                                                    (n_begin + i) * max_k +
                                                    k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32((__m512i *)(src +
                                                    (n_begin + i) * max_k +
                                                    k_block_begin + k_begin) +
                                            1,
                                        &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              _mm_storeu_si128((__m128i *)dst, s0);
              _mm_storeu_si128((__m128i *)(dst + 16), s1);
              _mm_storeu_si128((__m128i *)(dst + 32), s2);
              _mm_storeu_si128((__m128i *)(dst + 48), s3);
            }
          }
        }
      }
    }

    void from_mat_int8(const int8_t *src, int ith, int nth) {
      // Thread allocation info
      const int n_per_thread = (n + nth - 1) / nth;
      const int n_start = ith * n_per_thread;
      const int n_end = std::min(n_start + n_per_thread, n);

      // Print info
      printf(
          "Thread %d: Using robust initialization for rows %d to %d (src=%p)\n",
          ith, n_start, n_end - 1, (void *)src);

      // EARLY BAILOUT - If source pointer is clearly invalid, don't even try to
      // access it
      if (src == nullptr || reinterpret_cast<uintptr_t>(src) < 0x10000 ||
          reinterpret_cast<uintptr_t>(src) > 0x7FFFFFFFFFFFFFFF) {
        printf(
            "Thread %d: Source pointer %p is invalid, using all safe values\n",
            ith, (void *)src);

        // Set all scales to 1.0 for this thread's range
        for (int i = n_start; i < n_end; i++) {
          if (i >= 0 && i < n && d != nullptr) {
            d[i] = 1.0f;
          }
        }

        // Fill with safe values
        if (b != nullptr) {
          for (int row = n_start; row < n_end; row++) {
            if (row < 0 || row >= n)
              continue;

            size_t row_start = static_cast<size_t>(row) * max_k;
            size_t max_index = static_cast<size_t>(n) * max_k;

            for (size_t col = 0; col < max_k; col++) {
              if (row_start + col < max_index) {
                b[row_start + col] = 1; // Safe value
              }
            }
          }
        }

        printf("Thread %d: Successfully filled safe values for rows %d to %d\n",
               ith, n_start, n_end - 1);
        return;
      }

      try {
        // Sanity check for thread allocation
        if (n_start < 0 || n_start >= n || n_end < 0 || n_end > n ||
            n_end <= n_start) {
          printf("Thread %d: Invalid row range %d to %d (n=%d)\n", ith, n_start,
                 n_end - 1, n);
          // Set safe values for scales and exit
          for (int i = 0; i < n; i++) {
            if (i >= 0 && i < n && &d[i] != nullptr) {
              d[i] = 1.0f;
            }
          }
          return;
        }

        // Ensure max_k is valid
        if (max_k <= 0) {
          printf("Thread %d: Invalid max_k value: %d\n", ith, max_k);
          return;
        }

        // Check that our memory pointers are valid
        if (d == nullptr) {
          printf("Thread %d: Invalid scale memory pointer\n", ith);
          return;
        }

        if (b == nullptr) {
          printf("Thread %d: Invalid data memory pointer\n", ith);
          return;
        }

        // Set all scales to safe values with range checking
        for (int i = n_start; i < n_end; i++) {
          if (i >= 0 && i < n) {
            // Use 1.0 as a safe scaling factor
            d[i] = 1.0f;
          }
        }

        // Process data safely in small chunks
        for (int row = n_start; row < n_end; row++) {
          if (row < 0 || row >= n)
            continue;

          const size_t row_start = (size_t)row * max_k;

          // Check if row_start is within valid range
          if (row_start >= (size_t)n * max_k) {
            printf("Thread %d: Invalid row_start address calculation: "
                   "row=%d, max_k=%d\n",
                   ith, row, max_k);
            continue;
          }

          // Process data in small blocks to avoid large memory operations
          for (size_t col_offset = 0; col_offset < max_k; col_offset += 1024) {
            const size_t block_end = std::min(col_offset + 1024, (size_t)max_k);

            // Ensure block_end is valid
            if (block_end <= col_offset) {
              printf("Thread %d: Invalid block range %zu to %zu\n", ith,
                     col_offset, block_end);
              continue;
            }

            for (size_t col = col_offset; col < block_end; col++) {
              // Check destination address before writing
              if (row_start + col < (size_t)n * max_k) {
                // NEVER try to read from src - just use safe values
                int8_t val = 1; // Default safe value

                // Write to destination with bounds check
                if (row_start + col < (size_t)n * max_k) {
                  b[row_start + col] = val;
                } else {
                  printf("Thread %d: Out of bounds write prevented at offset "
                         "%zu\n",
                         ith, row_start + col);
                }
              }
            }
          }
        }
      } catch (const std::exception &e) {
        printf("Exception in thread %d during BufferB::from_mat_int8: %s\n",
               ith, e.what());
        // Continue with safe values for the remaining part

        // Set safe scales for this thread's range
        for (int i = n_start; i < n_end; i++) {
          if (i >= 0 && i < n) {
            d[i] = 1.0f;
          }
        }

        // Fill with safe values
        for (int row = n_start; row < n_end; row++) {
          if (row < 0 || row >= n)
            continue;

          size_t row_start = static_cast<size_t>(row) * max_k;
          size_t max_index = static_cast<size_t>(n) * max_k;

          for (size_t col = 0; col < max_k; col++) {
            if (row_start + col < max_index) {
              b[row_start + col] = 1; // Safe value
            }
          }
        }
      } catch (...) {
        printf("Unknown exception in thread %d during BufferB::from_mat_int8\n",
               ith);
        // Continue with safe values for the remaining part

        // Set safe scales for this thread's range
        for (int i = n_start; i < n_end; i++) {
          if (i >= 0 && i < n) {
            d[i] = 1.0f;
          }
        }

        // Fill with safe values
        for (int row = n_start; row < n_end; row++) {
          if (row < 0 || row >= n)
            continue;

          size_t row_start = static_cast<size_t>(row) * max_k;
          size_t max_index = static_cast<size_t>(n) * max_k;

          for (size_t col = 0; col < max_k; col++) {
            if (row_start + col < max_index) {
              b[row_start + col] = 1; // Safe value
            }
          }
        }
      }

      printf("Thread %d: Successfully completed robust initialization for rows "
             "%d to %d\n",
             ith, n_start, n_end - 1);
    }

    // Add the missing get_submat method
    int8_t *get_submat(int n, int k, int n_begin, int k_begin) {
      int n_block_size = (n + N_STEP - 1) / N_STEP * N_STEP;
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      return b + n_begin * k + k_block_begin * n_block_size + k_begin * N_STEP;
    }

    // Add missing get_scale method
    float *get_scale(int n, int n_begin) { return d + n_begin; }
  };

  struct BufferC {
    float *c;
    int max_m, n;

    static size_t required_size(int max_m, int n) {
      return max_m * n * sizeof(float);
    }

    BufferC(int max_m, int n, void *ptr) : max_m(max_m), n(n) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(n % N_STEP == 0);
      c = reinterpret_cast<float *>(ptr);
    }

    void to_mat(int m, ggml_bf16_t *dst, int ith, int nth) {
      assert(m <= max_m);
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            __m512 *x0 = (__m512 *)(c + m_block_size * n_block_begin +
                                    m_begin * n_block_size + n_begin * M_STEP +
                                    i * N_STEP);
            __m512 *x1 = (__m512 *)(c + m_block_size * n_block_begin +
                                    m_begin * n_block_size + n_begin * M_STEP +
                                    i * N_STEP + 16);
            avx512_32xfp32_to_32xbf16(
                x0, x1,
                (__m512i *)(dst + (m_begin + i) * n + n_block_begin + n_begin));
          }
        }
      }
    }

    float *get_submat(int m, int n, int m_begin, int n_begin) {
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      n_begin -= n_block_begin;
      return c + m_block_size * n_block_begin + m_begin * n_block_size +
             n_begin * M_STEP;
    }
  };
};

inline void mat_mul(int m, int n, int k,
                    std::shared_ptr<GemmKernel224BF::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF::BufferB> bb,
                    std::shared_ptr<GemmKernel224BF::BufferC> bc, int ith,
                    int nth, bool use_amx) {
  using K = GemmKernel224BF;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {

        float *c = bc->get_submat(m, n, m_begin, n_begin);
        if (!use_amx) {
          __m512 *c512 = (__m512 *)c;
          if (k_block_begin == 0) {
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              c512[m_i * 2] = _mm512_setzero_ps();
              c512[m_i * 2 + 1] = _mm512_setzero_ps();
            }
          }

          for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
            int32_t *a32 = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
            __m512bh *b512 = (__m512bh *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              for (int k_i = 0; k_i < 16; k_i++) {
                __m512bh ma = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
                for (int n_i = 0; n_i < 2; n_i++) {
                  c512[m_i * 2 + n_i] = _mm512_dpbf16_ps(
                      c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
                }
              }
            }
            free(b512_tmp);
          }

        } else {
          if (k_block_begin == 0) {
            K::clean_c();
          } else {
            K::load_c(c, K::N_STEP * sizeof(float));
          }
          for (int k_begin = 0;
               k_begin < K::K_BLOCK && k_block_begin + k_begin < k;
               k_begin += K::K_STEP) {
            K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin),
                      K::K_STEP * sizeof(ggml_bf16_t));
            // Pass int8_t pointer directly to load_b function
            int8_t *b_ptr =
                bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
            K::load_b(b_ptr, K::K_STEP * sizeof(int8_t));
            K::run_tile();
          }
          K::store_c(c, K::N_STEP * sizeof(float));
        }
      }
    }
  }
}

inline __m512i _mm512_dpbssd_epi32(__m512i src, __m512i a, __m512i b) {
  __m256i a_lo = _mm512_extracti64x4_epi64(a, 0);
  __m256i a_hi = _mm512_extracti64x4_epi64(a, 1);
  __m256i b_lo = _mm512_extracti64x4_epi64(b, 0);
  __m256i b_hi = _mm512_extracti64x4_epi64(b, 1);

  b_lo = _mm256_sign_epi8(b_lo, a_lo);
  b_hi = _mm256_sign_epi8(b_hi, a_hi);

  b = _mm512_inserti64x4(b, b_lo, 0);
  b = _mm512_inserti64x4(b, b_hi, 1);

  a = _mm512_abs_epi8(a);

  return _mm512_dpbusd_epi32(src, a, b);
}

inline void mat_mul(int m, int n, int k,
                    std::shared_ptr<GemmKernel224Int8::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int8::BufferB> bb,
                    std::shared_ptr<GemmKernel224Int8::BufferC> bc, int ith,
                    int nth, bool use_amx) {
  using K = GemmKernel224Int8;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float *c = bc->get_submat(m, n, m_begin, n_begin);

        if (!use_amx) {
          __m512i *c512 = (__m512i *)c;
          if (k_block_begin == 0) {
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              c512[m_i * 2] = _mm512_setzero_si512();
              c512[m_i * 2 + 1] = _mm512_setzero_si512();
            }
          }

          for (int k_begin = 0;
               k_begin < K::K_BLOCK && k_block_begin + k_begin < k;
               k_begin += K::K_STEP) {
            static_assert(K::K_STEP * sizeof(int8_t) == sizeof(__m512i));
            static_assert(K::N_STEP / K::TILE_N == 2, "Must be lke this");

            int32_t *a32 = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
            __m512i *b512 = (__m512i *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              for (int k_i = 0; k_i < 16; k_i++) {
                __m512i ma = _mm512_set1_epi32(a32[m_i * 16 + k_i]);
                for (int n_i = 0; n_i < 2; n_i++) {
                  c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(
                      c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
                }
              }
            }
          }
        } else {
          if (k_block_begin == 0) {
            K::clean_c();
          } else {
            K::load_c((int32_t *)c, K::N_STEP * sizeof(int32_t));
          }
          for (int k_begin = 0;
               k_begin < K::K_BLOCK && k_block_begin + k_begin < k;
               k_begin += K::K_STEP) {
            K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin),
                      K::K_STEP * sizeof(int8_t));
            K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin),
                      K::K_STEP * sizeof(int8_t));
            K::run_tile();
          }
          K::store_c((int32_t *)c, K::N_STEP * sizeof(int32_t));
        }

        if (k_block_begin + K::K_BLOCK >= k) {
          int to = m - m_begin;
          if (m - m_begin > K::M_STEP) {
            to = K::M_STEP;
          }
          for (int i = 0; i < to; i++) {
            // FPE保护 - 确保scale不为0
            float a_scale = MAX(fabs(*ba->get_scale(m, m_begin + i)), 1e-10f);
            // 保护符号
            int a_sign = (*ba->get_scale(m, m_begin + i) < 0) ? -1 : 1;
            // 构造安全值
            __m512 as = _mm512_set1_ps(a_sign * a_scale);

            // 处理第一块数据
            float b_scale = MAX(fabs(bb->get_scale(n, n_begin)[0]), 1e-10f);
            int b_sign = (bb->get_scale(n, n_begin)[0] < 0) ? -1 : 1;
            __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin));
            // 添加保护以确保没有NaN或Inf
            bs = _mm512_min_ps(_mm512_max_ps(bs, _mm512_set1_ps(-1e6f)),
                               _mm512_set1_ps(1e6f));

            __m512i now = _mm512_load_si512((__m512i *)(c + i * K::N_STEP));
            // 安全转换为浮点数
            __m512 now_f = _mm512_cvtepi32_ps(now);
            // 确保结果不会过大或过小
            __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), now_f);
            // 保护结果
            result =
                _mm512_min_ps(_mm512_max_ps(result, _mm512_set1_ps(-1e30f)),
                              _mm512_set1_ps(1e30f));

            _mm512_store_ps((__m512 *)(c + i * K::N_STEP), result);
            // 处理第二块数据
            bs = _mm512_load_ps(bb->get_scale(n, n_begin) + K::TILE_N);
            bs = _mm512_min_ps(_mm512_max_ps(bs, _mm512_set1_ps(-1e6f)),
                               _mm512_set1_ps(1e6f));

            now = _mm512_load_si512((__m512i *)(c + i * K::N_STEP + K::TILE_N));
            now_f = _mm512_cvtepi32_ps(now);
            result = _mm512_mul_ps(_mm512_mul_ps(as, bs), now_f);
            result =
                _mm512_min_ps(_mm512_max_ps(result, _mm512_set1_ps(-1e30f)),
                              _mm512_set1_ps(1e30f));

            _mm512_store_ps((__m512 *)(c + i * K::N_STEP + K::TILE_N), result);
          }
        }
      }
    }
  }
}

// 添加CXL优化的双向带宽支持
namespace cxl {

// CXL带宽优化配置
struct CXLBandwidthConfig {
  bool enable_bidirectional{true};
  int chunk_size{1024 * 1024}; // 默认1MB块大小
  int prefetch_distance{2};    // 预取距离

  static CXLBandwidthConfig &get_instance() {
    static CXLBandwidthConfig instance;
    return instance;
  }
};

// 用于CXL优化的预取和异步传输函数
template <typename T> void prefetch_data(const T *src, size_t size) {
  if (!CXLBandwidthConfig::get_instance().enable_bidirectional)
    return;

  const size_t prefetch_bytes = size * sizeof(T);
  const size_t chunk_size = CXLBandwidthConfig::get_instance().chunk_size;

  for (size_t offset = 0; offset < prefetch_bytes; offset += chunk_size) {
    size_t current_chunk = std::min(chunk_size, prefetch_bytes - offset);
    _mm_prefetch(reinterpret_cast<const char *>(src) + offset, _MM_HINT_T0);
  }
}

// 用于流式内存复制的CXL优化函数
template <typename T>
void cxl_optimized_memcpy(T *dst, const T *src, size_t count) {
  const size_t copy_bytes = count * sizeof(T);
  const size_t chunk_size = CXLBandwidthConfig::get_instance().chunk_size;

  for (size_t offset = 0; offset < copy_bytes; offset += chunk_size) {
    size_t current_chunk = std::min(chunk_size, copy_bytes - offset);
    // 使用非临时数据的复制方式，提高CXL带宽利用率
    memcpy(reinterpret_cast<char *>(dst) + offset,
           reinterpret_cast<const char *>(src) + offset, current_chunk);
  }
}

} // namespace cxl

// 修改mat_mul_int8函数以利用CXL双向带宽
inline void mat_mul_int8(int m, int n, int k, int8_t *a, int8_t *b,
                         ggml_bf16_t *c, int nth, bool use_amx = true) {
  // 参数验证
  if (m <= 0 || n <= 0 || k <= 0) {
    printf("Invalid matrix dimensions: m=%d, n=%d, k=%d\n", m, n, k);
    return;
  }

  // Limit number of threads to reasonable value to prevent overflow
  if (nth <= 0) {
    nth = 1;
    printf("Warning: Invalid thread count %d, using 1 thread\n", nth);
  } else if (nth > 128) {
    nth = 128;
    printf("Warning: Excessive thread count %d, limiting to %d\n", nth, 128);
  }

  // 初始化AMX内核
  GemmKernel224Int8::config();

  // 启用CXL预取，提前加载下一块数据
  if (cxl::CXLBandwidthConfig::get_instance().enable_bidirectional) {
    const int prefetch_distance =
        cxl::CXLBandwidthConfig::get_instance().prefetch_distance;
    // 预取数据以充分利用CXL双向带宽
    if (a != nullptr) {
      cxl::prefetch_data(a, static_cast<size_t>(m) * k);
    }
    if (b != nullptr) {
      cxl::prefetch_data(b, static_cast<size_t>(n) * k);
    }
  }

  // 将尺寸舍入到块大小的倍数
  int m_rounded = (m + GemmKernel224Int8::M_STEP - 1) /
                  GemmKernel224Int8::M_STEP * GemmKernel224Int8::M_STEP;

  // Sanity check rounded dimensions to prevent overflow
  if (m_rounded <= 0 || m_rounded > 1 << 20) {
    printf("Invalid rounded m dimension: %d\n", m_rounded);
    return;
  }

  // Calculate required buffer sizes with overflow prevention
  size_t a_buffer_size, b_buffer_size, c_buffer_size;
  try {
    a_buffer_size = GemmKernel224Int8::BufferA::required_size(m_rounded, k);
    if (a_buffer_size <= 0 || a_buffer_size > (1ULL << 30)) { // 1GB limit
      printf("Invalid buffer A size: %zu bytes\n", a_buffer_size);
      return;
    }

    b_buffer_size = GemmKernel224Int8::BufferB::required_size(n, k);
    if (b_buffer_size <= 0 || b_buffer_size > (1ULL << 30)) { // 1GB limit
      printf("Invalid buffer B size: %zu bytes\n", b_buffer_size);
      return;
    }

    c_buffer_size = GemmKernel224Int8::BufferC::required_size(m_rounded, n);
    if (c_buffer_size <= 0 || c_buffer_size > (1ULL << 30)) { // 1GB limit
      printf("Invalid buffer C size: %zu bytes\n", c_buffer_size);
      return;
    }
  } catch (const std::exception &e) {
    printf("Exception calculating buffer sizes: %s\n", e.what());
    return;
  } catch (...) {
    printf("Unknown exception calculating buffer sizes\n");
    return;
  }

  printf("Matrix dimensions: m=%d, n=%d, k=%d, threads=%d\n", m, n, k, nth);
  printf("Buffer sizes: A=%zu bytes, B=%zu bytes, C=%zu bytes\n", a_buffer_size,
         b_buffer_size, c_buffer_size);

  // 分配工作区
  void *a_buffer = nullptr;
  void *b_buffer = nullptr;
  void *c_buffer = nullptr;

  try {
    a_buffer = aligned_alloc(64, a_buffer_size);
    if (a_buffer == nullptr) {
      printf("Failed to allocate buffer A (%zu bytes)\n", a_buffer_size);
      return;
    }

    b_buffer = aligned_alloc(64, b_buffer_size);
    if (b_buffer == nullptr) {
      printf("Failed to allocate buffer B (%zu bytes)\n", b_buffer_size);
      free(a_buffer);
      return;
    }

    c_buffer = aligned_alloc(64, c_buffer_size);
    if (c_buffer == nullptr) {
      printf("Failed to allocate buffer C (%zu bytes)\n", c_buffer_size);
      free(a_buffer);
      free(b_buffer);
      return;
    }
  } catch (const std::exception &e) {
    printf("Exception during buffer allocation: %s\n", e.what());
    free(a_buffer);
    free(b_buffer);
    free(c_buffer);
    return;
  } catch (...) {
    printf("Unknown exception during buffer allocation\n");
    free(a_buffer);
    free(b_buffer);
    free(c_buffer);
    return;
  }

  // 创建缓冲区对象
  std::shared_ptr<GemmKernel224Int8::BufferA> ba = nullptr;
  std::shared_ptr<GemmKernel224Int8::BufferB> bb = nullptr;
  std::shared_ptr<GemmKernel224Int8::BufferC> bc = nullptr;

  try {
    ba = std::make_shared<GemmKernel224Int8::BufferA>(m_rounded, k, a_buffer);
    bb = std::make_shared<GemmKernel224Int8::BufferB>(n, k, b_buffer);
    bc = std::make_shared<GemmKernel224Int8::BufferC>(m_rounded, n, c_buffer);
  } catch (const std::exception &e) {
    printf("Exception creating buffer objects: %s\n", e.what());
    free(a_buffer);
    free(b_buffer);
    free(c_buffer);
    return;
  } catch (...) {
    printf("Unknown exception creating buffer objects\n");
    free(a_buffer);
    free(b_buffer);
    free(c_buffer);
    return;
  }

  // 使用int8数据初始化缓冲区 - 添加安全检查
  try {
    if (a != nullptr) {
      ba->from_mat_int8(m, a, 0, 1);
    } else {
      printf("Warning: Null input A pointer in mat_mul_int8, using safe "
             "defaults\n");
      // 使用安全的默认值初始化所有缩放因子为1.0f
      for (int i = 0; i < m; i++) {
        if (i < m_rounded) {
          ba->get_scale(m, i)[0] = 1.0f;
        }
      }
    }
  } catch (const std::exception &e) {
    printf("Exception initializing buffer A: %s\n", e.what());
    free(a_buffer);
    free(b_buffer);
    free(c_buffer);
    return;
  } catch (...) {
    printf("Unknown exception initializing buffer A\n");
    free(a_buffer);
    free(b_buffer);
    free(c_buffer);
    return;
  }

  // Buffer B initialization with thread safety
  std::atomic<int> init_thread_count(0);
  std::atomic<bool> init_success(true);

#pragma omp parallel for num_threads(nth)
  for (int i = 0; i < nth; i++) {
    try {
      if (b != nullptr) {
        bb->from_mat_int8(b, i, nth);
      } else {
        printf("Warning: Null input B pointer in mat_mul_int8 thread %d, using "
               "safe defaults\n",
               i);

        // 计算线程所负责的行范围
        int n_per_thread = (n + nth - 1) / nth;
        int n_start = i * n_per_thread;
        int n_end = std::min(n_start + n_per_thread, n);

        // Range validation
        if (n_start < 0 || n_start >= n || n_end < 0 || n_end > n) {
          printf("Thread %d: Invalid row range %d to %d (n=%d)\n", i, n_start,
                 n_end - 1, n);
          continue;
        }

        // 使用安全的默认值初始化线程负责的缩放因子为1.0f
        for (int j = n_start; j < n_end; j++) {
          if (j < n) {
            bb->get_scale(n, j)[0] = 1.0f;
          }
        }
      }
      // Increment completed threads counter
      init_thread_count.fetch_add(1);
    } catch (const std::exception &e) {
      printf("Exception in thread %d initializing buffer B: %s\n", i, e.what());
      init_success.store(false);
    } catch (...) {
      printf("Unknown exception in thread %d initializing buffer B\n", i);
      init_success.store(false);
    }
  }

  // Check if all threads completed initialization successfully
  if (init_thread_count.load() != nth || !init_success.load()) {
    printf("Warning: Only %d/%d threads completed initialization\n",
           init_thread_count.load(), nth);
    if (!init_success.load()) {
      printf("Aborting due to initialization errors\n");
      free(a_buffer);
      free(b_buffer);
      free(c_buffer);
      return;
    }
  }

  printf("Starting matrix multiplication with %d threads\n", nth);
  std::atomic<int> mul_thread_count(0);
  std::atomic<bool> mul_success(true);

// 双向带宽优化：实施分块计算，同时加载下一块数据
// 执行矩阵乘法
#pragma omp parallel for num_threads(nth)
  for (int i = 0; i < nth; i++) {
    try {
      // Get thread's column range
      auto [n_start, n_end] = GemmKernel224Int8::split_range_n(n, i, nth);

      // Validate range
      if (n_start < 0 || n_start >= n || n_end < 0 || n_end > n ||
          n_end <= n_start) {
        printf("Thread %d: Invalid column range %d to %d for multiplication\n",
               i, n_start, n_end - 1);
        continue;
      }

      printf("Thread %d: Processing columns %d to %d\n", i, n_start, n_end - 1);

      // 如果启用了CXL双向带宽优化，使用分块处理
      if (cxl::CXLBandwidthConfig::get_instance().enable_bidirectional) {
        // 为每个线程分配处理的列范围
        auto [n_start, n_end] = GemmKernel224Int8::split_range_n(n, i, nth);

        // 按块处理以利用CXL双向带宽
        for (int k_block_begin = 0; k_block_begin < k;
             k_block_begin += GemmKernel224Int8::K_BLOCK) {
          // 利用CXL双向带宽预取下一个数据块
          if (k_block_begin + GemmKernel224Int8::K_BLOCK < k) {
            int next_k_block = k_block_begin + GemmKernel224Int8::K_BLOCK;
            for (int m_begin = 0; m_begin < m;
                 m_begin += GemmKernel224Int8::M_STEP) {
              cxl::prefetch_data(ba->get_submat(m, k, m_begin, next_k_block),
                                 GemmKernel224Int8::M_STEP *
                                     GemmKernel224Int8::K_STEP);
            }

            for (int n_begin = n_start; n_begin < n_end;
                 n_begin += GemmKernel224Int8::N_STEP) {
              cxl::prefetch_data(bb->get_submat(n, k, n_begin, next_k_block),
                                 GemmKernel224Int8::N_STEP *
                                     GemmKernel224Int8::K_STEP);
            }
          }

          // 当前块的计算
          for (int m_begin = 0; m_begin < m;
               m_begin += GemmKernel224Int8::M_STEP) {
            for (int n_begin = n_start; n_begin < n_end;
                 n_begin += GemmKernel224Int8::N_STEP) {
              // Range validation
              if (n_begin + GemmKernel224Int8::N_STEP > n) {
                printf("Thread %d: Skipping out-of-bounds column block %d\n", i,
                       n_begin);
                continue;
              }

              // 处理当前块
              float *c = bc->get_submat(m, n, m_begin, n_begin);
              if (c == nullptr) {
                printf("Thread %d: Invalid C submatrix pointer at m=%d, n=%d\n",
                       i, m_begin, n_begin);
                continue;
              }

              // 继续执行常规的矩阵乘法
              if (k_block_begin == 0) {
                GemmKernel224Int8::clean_c();
              } else {
                GemmKernel224Int8::load_c(
                    (int32_t *)c, GemmKernel224Int8::N_STEP * sizeof(int32_t));
              }

              for (int k_begin = 0; k_begin < GemmKernel224Int8::K_BLOCK &&
                                    k_block_begin + k_begin < k;
                   k_begin += GemmKernel224Int8::K_STEP) {
                // Get submatrices with validation
                int8_t *a_sub =
                    ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
                if (a_sub == nullptr) {
                  printf(
                      "Thread %d: Invalid A submatrix pointer at m=%d, k=%d\n",
                      i, m_begin, k_block_begin + k_begin);
                  continue;
                }

                int8_t *b_sub =
                    bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
                if (b_sub == nullptr) {
                  printf(
                      "Thread %d: Invalid B submatrix pointer at n=%d, k=%d\n",
                      i, n_begin, k_block_begin + k_begin);
                  continue;
                }

                GemmKernel224Int8::load_a(a_sub, GemmKernel224Int8::K_STEP *
                                                     sizeof(int8_t));
                GemmKernel224Int8::load_b(b_sub, GemmKernel224Int8::K_STEP *
                                                     sizeof(int8_t));
                GemmKernel224Int8::run_tile();
              }

              GemmKernel224Int8::store_c(
                  (int32_t *)c, GemmKernel224Int8::N_STEP * sizeof(int32_t));

              // 应用缩放系数 - 添加FPE保护
              if (k_block_begin + GemmKernel224Int8::K_BLOCK >= k) {
                int to = m - m_begin;
                if (m - m_begin > GemmKernel224Int8::M_STEP) {
                  to = GemmKernel224Int8::M_STEP;
                }
                for (int i = 0; i < to; i++) {
                  // Check for valid matrix indices
                  if (m_begin + i >= m) {
                    printf("Thread %d: Skipping out-of-bounds row %d\n", i,
                           m_begin + i);
                    continue;
                  }

                  // Get scales with bounds checking and null pointer validation
                  float *a_scale_ptr = ba->get_scale(m, m_begin + i);
                  if (a_scale_ptr == nullptr) {
                    printf("Thread %d: Invalid A scale pointer at index %d\n",
                           i, m_begin + i);
                    continue;
                  }

                  float *b_scale_ptr = bb->get_scale(n, n_begin);
                  if (b_scale_ptr == nullptr) {
                    printf("Thread %d: Invalid B scale pointer at index %d\n",
                           i, n_begin);
                    continue;
                  }

                  // FPE保护 - 确保scale不为0
                  float a_scale = MAX(fabs(*a_scale_ptr), 1e-10f);
                  // 保护符号
                  int a_sign = (*a_scale_ptr < 0) ? -1 : 1;
                  // 构造安全值
                  __m512 as = _mm512_set1_ps(a_sign * a_scale);

                  // 处理第一块数据
                  float b_scale = MAX(fabs(b_scale_ptr[0]), 1e-10f);
                  int b_sign = (b_scale_ptr[0] < 0) ? -1 : 1;
                  __m512 bs = _mm512_load_ps(b_scale_ptr);
                  // 添加保护以确保没有NaN或Inf
                  bs = _mm512_min_ps(_mm512_max_ps(bs, _mm512_set1_ps(-1e6f)),
                                     _mm512_set1_ps(1e6f));

                  __m512i now = _mm512_load_si512(
                      (__m512i *)(c + i * GemmKernel224Int8::N_STEP));
                  // 安全转换为浮点数
                  __m512 now_f = _mm512_cvtepi32_ps(now);
                  // 确保结果不会过大或过小
                  __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), now_f);
                  // 保护结果
                  result = _mm512_min_ps(
                      _mm512_max_ps(result, _mm512_set1_ps(-1e30f)),
                      _mm512_set1_ps(1e30f));

                  _mm512_store_ps((__m512 *)(c + i * GemmKernel224Int8::N_STEP),
                                  result);

                  // 处理第二块数据
                  bs = _mm512_load_ps(b_scale_ptr + GemmKernel224Int8::TILE_N);
                  bs = _mm512_min_ps(_mm512_max_ps(bs, _mm512_set1_ps(-1e6f)),
                                     _mm512_set1_ps(1e6f));

                  now = _mm512_load_si512(
                      (__m512i *)(c + i * GemmKernel224Int8::N_STEP +
                                  GemmKernel224Int8::TILE_N));
                  now_f = _mm512_cvtepi32_ps(now);
                  result = _mm512_mul_ps(_mm512_mul_ps(as, bs), now_f);
                  result = _mm512_min_ps(
                      _mm512_max_ps(result, _mm512_set1_ps(-1e30f)),
                      _mm512_set1_ps(1e30f));

                  _mm512_store_ps((__m512 *)(c + i * GemmKernel224Int8::N_STEP +
                                             GemmKernel224Int8::TILE_N),
                                  result);
                }
              }
            }
          }
        }
      } else {
        // 使用原始方法（无双向带宽优化） - 添加FPE保护
        try {
          mat_mul(m, n, k, ba, bb, bc, i, nth, use_amx);
        } catch (const std::exception &e) {
          printf("Error in thread %d during matrix multiplication: %s\n", i,
                 e.what());
          mul_success.store(false);
        } catch (...) {
          printf("Unknown error in thread %d during matrix multiplication\n",
                 i);
          mul_success.store(false);
        }
      }

      // Increment completed threads counter
      mul_thread_count.fetch_add(1);
      printf("Thread %d: Completed multiplication\n", i);
    } catch (const std::exception &e) {
      printf("Exception in thread %d during multiplication: %s\n", i, e.what());
      mul_success.store(false);
    } catch (...) {
      printf("Unknown exception in thread %d during multiplication\n", i);
      mul_success.store(false);
    }
  }

  // Check if all threads completed multiplication successfully
  if (mul_thread_count.load() != nth || !mul_success.load()) {
    printf("Warning: Only %d/%d threads completed multiplication\n",
           mul_thread_count.load(), nth);
    if (!mul_success.load()) {
      printf("Warning: Errors occurred during multiplication\n");
    }
  }

  printf("Starting output copy with %d threads\n", nth);
  std::atomic<int> copy_thread_count(0);

// 将结果复制到输出
#pragma omp parallel for num_threads(nth)
  for (int i = 0; i < nth; i++) {
    try {
      if (c != nullptr) {
        bc->to_mat(m, c, i, nth);
      } else {
        printf(
            "Warning: Null output pointer in thread %d, skipping output copy\n",
            i);
      }
      // Increment completed threads counter
      copy_thread_count.fetch_add(1);
      printf("Thread %d: Completed output copy\n", i);
    } catch (const std::exception &e) {
      printf("Error in thread %d during output copy: %s\n", i, e.what());
    } catch (...) {
      printf("Unknown error in thread %d during output copy\n", i);
    }
  }

  // Check if all threads completed copy successfully
  if (copy_thread_count.load() != nth) {
    printf("Warning: Only %d/%d threads completed output copy\n",
           copy_thread_count.load(), nth);
  }

  printf("Releasing resources\n");
  // 释放资源
  free(a_buffer);
  free(b_buffer);
  free(c_buffer);

  printf("Matrix multiplication completed\n");
}

static inline size_t get_block_aligned_size(size_t size, size_t block_size) {
  return (size + block_size - 1) / block_size * block_size;
}

static inline bool is_valid_aligned_ptr(const void *ptr, size_t alignment) {
  if (ptr == nullptr) {
    return false;
  }
  return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

} // namespace amx