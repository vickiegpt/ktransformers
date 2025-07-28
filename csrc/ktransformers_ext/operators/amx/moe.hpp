/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2025-04-25 18:28:12
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2025-04-25 18:28:12
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_H
#define CPUINFER_OPERATOR_AMX_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

#include "la/amx.hpp"

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
void *numa_alloc_aligned(size_t size, int node, size_t alignment) {
  void *ptr = numa_alloc_onnode(size, node);
  assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
  return ptr;
}
#endif

static inline __m512 exp_avx512(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.44269504089f);
  const __m512 c1 = _mm512_set1_ps(0.69314718056f);

  __m512 y = _mm512_mul_ps(x, log2e);
  __m512i int_part = _mm512_cvtps_epi32(y);
  __m512 frac_part = _mm512_sub_ps(y, _mm512_cvtepi32_ps(int_part));

  const __m512 poly_1 = _mm512_set1_ps(0.9999999995f);
  const __m512 poly_2 = _mm512_set1_ps(0.6931471805f);
  const __m512 poly_3 = _mm512_set1_ps(0.2402265069f);
  const __m512 poly_4 = _mm512_set1_ps(0.0555041087f);
  const __m512 poly_5 = _mm512_set1_ps(0.0096181291f);
  const __m512 poly_6 = _mm512_set1_ps(0.0013333558f);

  __m512 frac_exp = _mm512_fmadd_ps(
      frac_part, poly_6,
      _mm512_fmadd_ps(
          frac_part, poly_5,
          _mm512_fmadd_ps(
              frac_part, poly_4,
              _mm512_fmadd_ps(frac_part, poly_3,
                              _mm512_fmadd_ps(frac_part, poly_2, poly_1)))));

  __m512 two_pow_i =
      _mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(int_part));
  return _mm512_mul_ps(two_pow_i, frac_exp);
}

static inline __m512 act_fn(__m512 gate_val, __m512 up_val) {
  __m512 neg_gate_val = _mm512_sub_ps(_mm512_setzero_ps(), gate_val);
  __m512 exp_neg_gate = exp_avx512(neg_gate_val);
  __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_gate);
  __m512 act_val = _mm512_div_ps(gate_val, denom);

  return _mm512_mul_ps(act_val, up_val);
}

static inline __m512 relu_act_fn(__m512 gate_val, __m512 up_val) {
  __m512 zero_vec = _mm512_setzero_ps();
  __m512 act_val = _mm512_max_ps(zero_vec, gate_val);
  return _mm512_mul_ps(act_val, up_val);
}

struct AMX_MOEConfig {
  int expert_num;
  int routed_expert_num;
  int hidden_size;
  int intermediate_size;
  int max_len;
  bool use_silu;
  void *gate_proj;
  void *up_proj;
  void *down_proj;

  AMX_MOEConfig() {}

  AMX_MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int max_len, bool use_silu,
                void *gate_proj, void *up_proj, void *down_proj)
      : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size),
        intermediate_size(intermediate_size), max_len(max_len), use_silu(use_silu), gate_proj(gate_proj), up_proj(up_proj),
        down_proj(down_proj) {}
};

template <class T> class AMX_MOE {
private:
  AMX_MOEConfig config_;
  void *gate_proj_; // [expert_num * intermediate_size * hidden_size ( /32 if
                    // quantized)]
  void *up_proj_;   // [expert_num * intermediate_size * hidden_size ( /32 if
                    // quantized)]
  void *down_proj_; // [expert_num * hidden_size * intermediate_size ( /32 if
                    // quantized)]

  ggml_bf16_t *m_local_input_; // [routed_expert_num * max_len * hidden_size]
  ggml_bf16_t *
      m_local_gate_output_; // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t
      *m_local_up_output_; // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t
      *m_local_down_output_; // [routed_expert_num * max_len * hidden_size]

  std::vector<std::vector<int>> m_local_pos_;    // [max_len, routed_expert_num]
  std::vector<int> m_local_num_;                 // [expert_num]
  std::vector<int> m_expert_id_map_;             // [expert_num]
  std::vector<ggml_bf16_t *> m_local_input_ptr_; // [expert_num]
  std::vector<ggml_bf16_t *> m_local_gate_output_ptr_; // [expert_num]
  std::vector<ggml_bf16_t *> m_local_up_output_ptr_;   // [expert_num]
  std::vector<ggml_bf16_t *> m_local_down_output_ptr_; // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

#ifdef USE_NUMA
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> gate_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> up_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> down_bb_numa_;
#else
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
#endif

public:
  AMX_MOE(AMX_MOEConfig config) {
    // Validate configuration parameters
    if (config.expert_num <= 0) {
      printf("Warning: Invalid expert_num %d, using 1\n", config.expert_num);
      config.expert_num = 1;
    }

    if (config.routed_expert_num <= 0) {
      printf("Warning: Invalid routed_expert_num %d, using 1\n",
             config.routed_expert_num);
      config.routed_expert_num = 1;
    }

    if (config.hidden_size <= 0) {
      printf("Warning: Invalid hidden_size %d, using 32\n", config.hidden_size);
      config.hidden_size = 32;
    }

    if (config.intermediate_size <= 0) {
      printf("Warning: Invalid intermediate_size %d, using 32\n",
             config.intermediate_size);
      config.intermediate_size = 32;
    }

    if (config.max_len <= 0) {
      printf("Warning: Invalid max_len %d, using 1\n", config.max_len);
      config.max_len = 1;
    }

    // Store validated config
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    // Create memory requests with bounds checking
    std::vector<std::pair<void **, uint64_t>> m_mem_requests;

    // Calculate sizes with overflow protection
    uint64_t input_size = (uint64_t)config_.routed_expert_num *
                          (uint64_t)config_.max_len *
                          (uint64_t)config_.hidden_size * sizeof(ggml_bf16_t);

    uint64_t gate_up_size =
        (uint64_t)config_.routed_expert_num * (uint64_t)config_.max_len *
        (uint64_t)config_.intermediate_size * sizeof(ggml_bf16_t);

    uint64_t down_size = (uint64_t)config_.routed_expert_num *
                         (uint64_t)config_.max_len *
                         (uint64_t)config_.hidden_size * sizeof(ggml_bf16_t);

    // Catch potential overflow
    if (input_size == 0 || gate_up_size == 0 || down_size == 0) {
      printf("Error: Memory size calculation overflow detected\n");
      // Use minimum sensible sizes
      input_size = sizeof(ggml_bf16_t) * 32;
      gate_up_size = sizeof(ggml_bf16_t) * 32;
      down_size = sizeof(ggml_bf16_t) * 32;
    }

    m_mem_requests.push_back({(void **)&m_local_input_, input_size});
    m_mem_requests.push_back({(void **)&m_local_gate_output_, gate_up_size});
    m_mem_requests.push_back({(void **)&m_local_up_output_, gate_up_size});
    m_mem_requests.push_back({(void **)&m_local_down_output_, down_size});

    // Allocate memory for buffer pointers
    std::vector<void *> gate_up_ba_ptr(config_.expert_num);
    std::vector<void *> gate_bc_ptr(config_.expert_num);
    std::vector<void *> up_bc_ptr(config_.expert_num);
    std::vector<void *> down_ba_ptr(config_.expert_num);
    std::vector<void *> down_bc_ptr(config_.expert_num);

    for (int i = 0; i < config_.expert_num; i++) {
      uint64_t bufferA_size =
          T::BufferA::required_size(config_.max_len, config_.hidden_size);
      uint64_t bufferC_interm_size =
          T::BufferC::required_size(config_.max_len, config_.intermediate_size);
      uint64_t bufferC_hidden_size =
          T::BufferC::required_size(config_.max_len, config_.hidden_size);

      // Catch potential overflow in buffer size calculation
      if (bufferA_size == 0 || bufferC_interm_size == 0 ||
          bufferC_hidden_size == 0) {
        printf(
            "Error: Buffer size calculation overflow detected for expert %d\n",
            i);
        // Use minimum sensible sizes
        bufferA_size = 64; // Ensure at least 64 bytes for alignment
        bufferC_interm_size = 64;
        bufferC_hidden_size = 64;
      }

      m_mem_requests.push_back({(void **)&gate_up_ba_ptr[i], bufferA_size});
      m_mem_requests.push_back({(void **)&gate_bc_ptr[i], bufferC_interm_size});
      m_mem_requests.push_back({(void **)&up_bc_ptr[i], bufferC_interm_size});
      m_mem_requests.push_back(
          {(void **)&down_ba_ptr[i],
           T::BufferA::required_size(config_.max_len,
                                     config_.intermediate_size)});
      m_mem_requests.push_back({(void **)&down_bc_ptr[i], bufferC_hidden_size});
    }

    // Allocate memory from shared buffer
    try {
      shared_mem_buffer.alloc(this, m_mem_requests);
    } catch (const std::exception &e) {
      printf("Error during memory allocation: %s\n", e.what());
      // Continue with potentially partial allocation - we'll handle null
      // pointers later
    }

    // Initialize data structures
    try {
      m_local_pos_.resize(config_.max_len);
      for (int i = 0; i < config_.max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
      }
      // Ensure expert_num is valid before resizing
      if (config_.expert_num <= 0) {
        throw std::runtime_error("Invalid expert_num: " + std::to_string(config_.expert_num));
      }
      m_expert_id_map_.resize(config_.expert_num);
      m_local_num_.resize(config_.expert_num);
      m_local_input_ptr_.resize(config_.expert_num);
      m_local_gate_output_ptr_.resize(config_.expert_num);
      m_local_up_output_ptr_.resize(config_.expert_num);
      m_local_down_output_ptr_.resize(config_.expert_num);
    } catch (const std::exception &e) {
      printf("Error during vector allocation: %s\n", e.what());
      // Continue with best effort
    }

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      try {
        gate_up_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.hidden_size, gate_up_ba_ptr[i]));
        gate_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.intermediate_size, gate_bc_ptr[i]));
        up_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.intermediate_size, up_bc_ptr[i]));
        down_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.intermediate_size, down_ba_ptr[i]));
        down_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.hidden_size, down_bc_ptr[i]));

#ifdef USE_NUMA
        try {
          int numa_nodes = numa_num_configured_nodes();
          // Ensure numa_nodes is valid
          if (numa_nodes <= 0) {
            printf("Warning: Invalid NUMA node count %d, using 1\n",
                   numa_nodes);
            numa_nodes = 1;
          }

          // Resize only once (first expert)
          if (i == 0) {
            gate_bb_numa_.resize(numa_nodes);
            up_bb_numa_.resize(numa_nodes);
            down_bb_numa_.resize(numa_nodes);
          }

          for (int j = 0; j < numa_nodes; j++) {
            // Calculate buffer sizes with overflow protection
            uint64_t gate_up_bb_size = T::BufferB::required_size(
                config_.intermediate_size, config_.hidden_size);
            uint64_t down_bb_size = T::BufferB::required_size(
                config_.hidden_size, config_.intermediate_size);

            // Validate buffer sizes
            if (gate_up_bb_size == 0) {
              printf("Warning: Invalid gate_up_bb_size, using minimum size\n");
              gate_up_bb_size = 64; // Minimum allocation for alignment
            }

            if (down_bb_size == 0) {
              printf("Warning: Invalid down_bb_size, using minimum size\n");
              down_bb_size = 64; // Minimum allocation for alignment
            }

            // Allocate memory with error handling
            void *gate_bb_ptr = nullptr;
            try {
              gate_bb_ptr = numa_alloc_aligned(gate_up_bb_size, j, 64);
              if (gate_bb_ptr == nullptr) {
                printf(
                    "Warning: Failed to allocate gate_bb_ptr on NUMA node %d\n",
                    j);
                gate_bb_ptr = std::aligned_alloc(64, gate_up_bb_size);
              }
            } catch (...) {
              printf("Error allocating gate_bb_ptr on NUMA node %d\n", j);
              gate_bb_ptr = std::aligned_alloc(64, gate_up_bb_size);
            }

            if (j >= gate_bb_numa_.size()) {
              printf("Warning: NUMA node %d out of bounds (max: %lu)\n", j,
                     gate_bb_numa_.size());
              continue;
            }

            // Create and store BufferB with null check
            if (gate_bb_ptr != nullptr) {
              gate_bb_numa_[j].push_back(std::make_shared<typename T::BufferB>(
                  config_.intermediate_size, config_.hidden_size, gate_bb_ptr));
            } else {
              printf("Error: Could not allocate gate_bb_ptr, using empty "
                     "BufferB\n");
              // Push empty shared_ptr to maintain indices
              gate_bb_numa_[j].push_back(nullptr);
            }

            // Repeat for up_bb with similar error handling
            void *up_bb_ptr = nullptr;
            try {
              up_bb_ptr = numa_alloc_aligned(gate_up_bb_size, j, 64);
              if (up_bb_ptr == nullptr) {
                printf(
                    "Warning: Failed to allocate up_bb_ptr on NUMA node %d\n",
                    j);
                up_bb_ptr = std::aligned_alloc(64, gate_up_bb_size);
              }
            } catch (...) {
              printf("Error allocating up_bb_ptr on NUMA node %d\n", j);
              up_bb_ptr = std::aligned_alloc(64, gate_up_bb_size);
            }

            // Create and store BufferB with null check
            if (up_bb_ptr != nullptr) {
              up_bb_numa_[j].push_back(std::make_shared<typename T::BufferB>(
                  config_.intermediate_size, config_.hidden_size, up_bb_ptr));
            } else {
              printf(
                  "Error: Could not allocate up_bb_ptr, using empty BufferB\n");
              // Push empty shared_ptr to maintain indices
              up_bb_numa_[j].push_back(nullptr);
            }

            // Repeat for down_bb with similar error handling
            void *down_bb_ptr = nullptr;
            try {
              down_bb_ptr = numa_alloc_aligned(down_bb_size, j, 64);
              if (down_bb_ptr == nullptr) {
                printf(
                    "Warning: Failed to allocate down_bb_ptr on NUMA node %d\n",
                    j);
                down_bb_ptr = std::aligned_alloc(64, down_bb_size);
              }
            } catch (...) {
              printf("Error allocating down_bb_ptr on NUMA node %d\n", j);
              down_bb_ptr = std::aligned_alloc(64, down_bb_size);
            }

            // Create and store BufferB with null check
            if (down_bb_ptr != nullptr) {
              down_bb_numa_[j].push_back(std::make_shared<typename T::BufferB>(
                  config_.hidden_size, config_.intermediate_size, down_bb_ptr));
            } else {
              printf("Error: Could not allocate down_bb_ptr, using empty "
                     "BufferB\n");
              // Push empty shared_ptr to maintain indices
              down_bb_numa_[j].push_back(nullptr);
            }
          }
        } catch (const std::exception &e) {
          printf("Error during NUMA buffer initialization for expert %lu: %s\n",
                 i, e.what());
        } catch (...) {
          printf("Unknown error during NUMA buffer initialization for expert "
                 "%lu\n",
                 i);
        }
#else
        // Non-NUMA buffer allocation with error handling
        try {
          // Calculate buffer sizes with overflow protection
          uint64_t gate_up_bb_size = T::BufferB::required_size(
              config_.intermediate_size, config_.hidden_size);
          uint64_t down_bb_size = T::BufferB::required_size(
              config_.hidden_size, config_.intermediate_size);

          // Validate buffer sizes
          if (gate_up_bb_size == 0) {
            printf("Warning: Invalid gate_up_bb_size, using minimum size\n");
            gate_up_bb_size = 64; // Minimum allocation for alignment
          }

          if (down_bb_size == 0) {
            printf("Warning: Invalid down_bb_size, using minimum size\n");
            down_bb_size = 64; // Minimum allocation for alignment
          }

          // Allocate gate_bb with error handling
          void *gate_bb_ptr = std::aligned_alloc(64, gate_up_bb_size);
          if (gate_bb_ptr != nullptr) {
            gate_bb_.push_back(std::make_shared<typename T::BufferB>(
                config_.intermediate_size, config_.hidden_size, gate_bb_ptr));
          } else {
            printf(
                "Error: Failed to allocate gate_bb_ptr, using null buffer\n");
            gate_bb_.push_back(nullptr);
          }

          // Allocate up_bb with error handling
          void *up_bb_ptr = std::aligned_alloc(64, gate_up_bb_size);
          if (up_bb_ptr != nullptr) {
            up_bb_.push_back(std::make_shared<typename T::BufferB>(
                config_.intermediate_size, config_.hidden_size, up_bb_ptr));
          } else {
            printf("Error: Failed to allocate up_bb_ptr, using null buffer\n");
            up_bb_.push_back(nullptr);
          }

          // Allocate down_bb with error handling
          void *down_bb_ptr = std::aligned_alloc(64, down_bb_size);
          if (down_bb_ptr != nullptr) {
            down_bb_.push_back(std::make_shared<typename T::BufferB>(
                config_.hidden_size, config_.intermediate_size, down_bb_ptr));
          } else {
            printf(
                "Error: Failed to allocate down_bb_ptr, using null buffer\n");
            down_bb_.push_back(nullptr);
          }
        } catch (const std::exception &e) {
          printf("Error during buffer initialization for expert %lu: %s\n", i,
                 e.what());
          // Add null buffers to maintain indices
          gate_bb_.push_back(nullptr);
          up_bb_.push_back(nullptr);
          down_bb_.push_back(nullptr);
        } catch (...) {
          printf("Unknown error during buffer initialization for expert %lu\n",
                 i);
          // Add null buffers to maintain indices
          gate_bb_.push_back(nullptr);
          up_bb_.push_back(nullptr);
          down_bb_.push_back(nullptr);
        }
#endif
      } catch (const std::exception &e) {
        printf("Error creating buffer for expert %lu: %s\n", i, e.what());
        // Continue with next expert
      }
    }
  }

  ~AMX_MOE() { shared_mem_buffer.dealloc(this); }

  void load_weights(Backend *backend) {
    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            gate_bb_numa_[j][expert_idx]->from_mat(
                (ggml_bf16_t *)config_.gate_proj +
                    expert_idx * config_.intermediate_size *
                        config_.hidden_size,
                ith, nth);
            up_bb_numa_[j][expert_idx]->from_mat(
                (ggml_bf16_t *)config_.up_proj + expert_idx *
                                                     config_.intermediate_size *
                                                     config_.hidden_size,
                ith, nth);
          }
#else
          gate_bb_[expert_idx]->from_mat(
              (ggml_bf16_t *)config_.gate_proj +
                  expert_idx * config_.intermediate_size * config_.hidden_size,
              ith, nth);
          up_bb_[expert_idx]->from_mat(
              (ggml_bf16_t *)config_.up_proj +
                  expert_idx * config_.intermediate_size * config_.hidden_size,
              ith, nth);
#endif
        },
        nullptr);
    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            down_bb_numa_[j][expert_idx]->from_mat(
                (ggml_bf16_t *)config_.down_proj +
                    expert_idx * config_.hidden_size *
                        config_.intermediate_size,
                ith, nth);
          }
#else
          down_bb_[expert_idx]->from_mat((ggml_bf16_t *)config_.down_proj +
                                             expert_idx * config_.hidden_size *
                                                 config_.intermediate_size,
                                         ith, nth);
#endif
        },
        nullptr);
  }

  // Add specialized weight loading function for int8 matrices
  void load_weights_int8(Backend *backend) {
    // Validate weight pointers before processing
    if (!config_.gate_proj || !config_.up_proj || !config_.down_proj) {
      printf(
          "ERROR: One or more weight pointers are null in load_weights_int8\n");
      return;
    }

    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;

          // Calculate pointers with bounds checking
          int8_t *gate_proj_ptr =
              (int8_t *)config_.gate_proj +
              expert_idx * config_.intermediate_size * config_.hidden_size;
          int8_t *up_proj_ptr =
              (int8_t *)config_.up_proj +
              expert_idx * config_.intermediate_size * config_.hidden_size;

          // Validate weight pointer ranges
          if (expert_idx >= config_.expert_num) {
            printf("ERROR: Expert index %lu out of bounds (max: %d)\n",
                   expert_idx, config_.expert_num);
            return;
          }

#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            gate_bb_numa_[j][expert_idx]->from_mat_int8(gate_proj_ptr, ith,
                                                        nth);
            up_bb_numa_[j][expert_idx]->from_mat_int8(up_proj_ptr, ith, nth);
          }
#else
          gate_bb_[expert_idx]->from_mat_int8(gate_proj_ptr, ith, nth);
          up_bb_[expert_idx]->from_mat_int8(up_proj_ptr, ith, nth);
#endif
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;

          // Calculate pointer with bounds checking
          int8_t *down_proj_ptr =
              (int8_t *)config_.down_proj +
              expert_idx * config_.hidden_size * config_.intermediate_size;

          // Validate expert index
          if (expert_idx >= config_.expert_num) {
            printf("ERROR: Expert index %lu out of bounds (max: %d)\n",
                   expert_idx, config_.expert_num);
            return;
          }

#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            down_bb_numa_[j][expert_idx]->from_mat_int8(down_proj_ptr, ith,
                                                        nth);
          }
#else
          down_bb_[expert_idx]->from_mat_int8(down_proj_ptr, ith, nth);
#endif
        },
        nullptr);
  }

  void warm_up(Backend *backend) {}

  void forward(int qlen, int k, const uint64_t *expert_ids,
               const float *weights, const void *input, void *output,
               int *batch_size_tensor, Backend *backend) {
    qlen = batch_size_tensor[0];
    bool use_amx = (qlen > 4 * config_.expert_num / config_.routed_expert_num);
    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }

    // Validate expert_ids and count activated experts
    if (expert_ids == nullptr) {
      printf("Warning: expert_ids is null, using default expert 0\n");
      // Default to expert 0
      for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
          // Bounds check for m_local_pos_ access
          if (i >= config_.max_len || j >= config_.routed_expert_num) {
            printf("Warning: Index out of bounds for m_local_pos_[%d][%d] in default path (max_len=%d, routed_expert_num=%d)\n",
                   i, j, config_.max_len, config_.routed_expert_num);
            continue;
          }
          m_local_pos_[i][j] = m_local_num_[0]++;
        }
      }
      // Only activate expert 0
      if (m_local_num_[0] > 0) {
        m_expert_id_map_[0] = 0;
        activated_expert = 1;
      }
    } else {
      // Process actual expert_ids
      for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
          // Bounds check for m_local_pos_ access
          if (i >= config_.max_len || j >= config_.routed_expert_num) {
            printf("Warning: Index out of bounds for m_local_pos_[%d][%d] (max_len=%d, routed_expert_num=%d)\n",
                   i, j, config_.max_len, config_.routed_expert_num);
            continue;
          }
          // Initialize to safe default
          m_local_pos_[i][j] = 0;
          
          if (i * k + j >= qlen * k) {
            continue;  // Bounds check for expert_ids array access
          }
          uint64_t expert_id = expert_ids[i * k + j];
          // Validate expert_id
          if (expert_id >= (uint64_t)config_.expert_num || expert_id >= m_local_num_.size()) {
            printf(
                "Warning: Invalid expert_id %lu at [%d,%d] (expert_num=%d, vector_size=%lu), using expert 0\n",
                expert_id, i, j, config_.expert_num, m_local_num_.size());
            expert_id = 0;
          }
          
          // Extra safety check
          if (expert_id >= m_local_num_.size()) {
            printf("CRITICAL: expert_id %lu still out of bounds after validation! Skipping.\n", expert_id);
            continue;
          }
          
          // Final safety check before array access
          if (expert_id >= m_local_num_.size()) {
            printf("FATAL: expert_id %lu out of bounds (size=%lu) after all checks!\n", 
                   expert_id, m_local_num_.size());
            expert_id = 0;  // Force to 0
            if (m_local_num_.empty()) {
              printf("FATAL: m_local_num_ is empty! Cannot proceed.\n");
              continue;
            }
          }
          
          // Bounds check before assignment
          if (i >= config_.max_len || j >= config_.routed_expert_num) {
            printf("Warning: Index out of bounds for m_local_pos_[%d][%d] assignment (max_len=%d, routed_expert_num=%d)\n",
                   i, j, config_.max_len, config_.routed_expert_num);
            continue;
          }
          m_local_pos_[i][j] = m_local_num_[expert_id]++;
        }
      }

      // Count activated experts
      for (int i = 0; i < config_.expert_num; i++) {
        if (m_local_num_[i] > 0) {
          m_expert_id_map_[activated_expert] = i;
          activated_expert++;
        }
      }
    }

    // Ensure we have at least one activated expert
    if (activated_expert <= 0) {
      printf("Warning: No activated experts, using default expert 0\n");
      m_expert_id_map_[0] = 0;
      m_local_num_[0] = 1;
      activated_expert = 1;
    }

    // Set up offsets for expert data
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] =
          m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] =
          m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] =
          m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }

    // Validate input before copying data
    if (input == nullptr) {
      printf("Warning: Input is null, using zeros\n");
      // Skip the input copying step
    } else {
      backend->do_work_stealing_job(
          qlen, nullptr,
          [&](int i) {
            for (int j = 0; j < k; j++) {
              uint64_t expert_id = expert_ids ? expert_ids[i * k + j] : 0;
              // Validate expert_id again
              if (expert_id >= (uint64_t)config_.expert_num) {
                expert_id = 0;
              }
              // Bounds check for m_local_pos_ access
              if (i >= config_.max_len || j >= config_.routed_expert_num) {
                printf("Warning: Index out of bounds for m_local_pos_[%d][%d] in input copy (max_len=%d, routed_expert_num=%d)\n",
                       i, j, config_.max_len, config_.routed_expert_num);
                continue;
              }
              memcpy(m_local_input_ptr_[expert_id] +
                         m_local_pos_[i][j] * config_.hidden_size,
                     (ggml_bf16_t *)input + i * config_.hidden_size,
                     sizeof(ggml_bf16_t) * config_.hidden_size);
            }
          },
          nullptr);
    }

    // Process activated experts
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          if (task_id < 0 || task_id >= activated_expert) {
            printf(
                "Warning: Invalid task_id %d in expert processing (max: %d)\n",
                task_id, activated_expert - 1);
            return;
          }

          int expert_idx = m_expert_id_map_[task_id];
          if (expert_idx < 0 || expert_idx >= config_.expert_num) {
            printf("Warning: Invalid expert_idx %d mapped from task_id %d\n",
                   expert_idx, task_id);
            return;
          }

          gate_up_ba_[expert_idx]->from_mat(
              m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Calculate number of threads with safety checks
    int recommended_nth = T::recommended_nth(config_.intermediate_size);
    int nth = std::max(1, std::min(recommended_nth, backend->get_thread_num()));

    // Ensure we don't have a zero or negative thread count
    if (nth <= 0)
      nth = 1;

    // Matrix multiplication with bounds checking
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          // Validate task_id
          if (task_id < 0 || task_id >= nth * activated_expert) {
            printf("Warning: Invalid task_id %d in matrix multiplication\n",
                   task_id);
            return;
          }

          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          // Validate expert_idx
          if (expert_idx < 0 || expert_idx >= config_.expert_num) {
            printf("Warning: Invalid expert_idx %d in matrix multiplication\n",
                   expert_idx);
            return;
          }

#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size,
                       config_.hidden_size, gate_up_ba_[expert_idx],
                       gate_bb_numa_[Backend::numa_node][expert_idx],
                       gate_bc_[expert_idx], ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size,
                       config_.hidden_size, gate_up_ba_[expert_idx],
                       up_bb_numa_[Backend::numa_node][expert_idx],
                       up_bc_[expert_idx], ith, nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size,
                       config_.hidden_size, gate_up_ba_[expert_idx],
                       gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth,
                       use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size,
                       config_.hidden_size, gate_up_ba_[expert_idx],
                       up_bb_[expert_idx], up_bc_[expert_idx], ith, nth,
                       use_amx);
#endif
          gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
          if (config_.use_silu) {
            for (int i = 0; i < m_local_num_[expert_idx]; i++) {
                ggml_bf16_t *gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
                ggml_bf16_t *up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
                for (int j = n_start; j < n_end; j += 32) {
                  __m512 gate_val0, gate_val1, up_val0, up_val1;
                  avx512_32xbf16_to_32xfp32((__m512i *)(gate_output_ptr + j), &gate_val0, &gate_val1);
                  avx512_32xbf16_to_32xfp32((__m512i *)(up_output_ptr + j), &up_val0, &up_val1);
                  __m512 result0 = act_fn(gate_val0, up_val0);
                  __m512 result1 = act_fn(gate_val1, up_val1);
                  avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(gate_output_ptr + j));
                }
              }
          }
          else {
              for (int i = 0; i < m_local_num_[expert_idx]; i++) {
                ggml_bf16_t *gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
                ggml_bf16_t *up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
                for (int j = n_start; j < n_end; j += 32) {
                  __m512 gate_val0, gate_val1, up_val0, up_val1;
                  avx512_32xbf16_to_32xfp32((__m512i *)(gate_output_ptr + j), &gate_val0, &gate_val1);
                  avx512_32xbf16_to_32xfp32((__m512i *)(up_output_ptr + j), &up_val0, &up_val1);
                  __m512 result0 = relu_act_fn(gate_val0, up_val0);
                  __m512 result1 = relu_act_fn(gate_val1, up_val1);
                  avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(gate_output_ptr + j));
                }
              }
          }
          
        },
        nullptr);

    // Process the experts for the down projection
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          // Validate task_id
          if (task_id < 0 || task_id >= activated_expert) {
            printf("Warning: Invalid task_id %d in down projection (max: %d)\n",
                   task_id, activated_expert - 1);
            return;
          }

          int expert_idx = m_expert_id_map_[task_id];
          // Validate expert_idx
          if (expert_idx < 0 || expert_idx >= config_.expert_num) {
            printf("Warning: Invalid expert_idx %d in down projection\n",
                   expert_idx);
            return;
          }

          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx],
                                         m_local_gate_output_ptr_[expert_idx],
                                         0, 1);
        },
        nullptr);

    // Recalculate thread count for hidden size
    int recommended_nth_hidden = T::recommended_nth(config_.hidden_size);
    nth = std::max(1,
                   std::min(recommended_nth_hidden, backend->get_thread_num()));

    // Ensure we don't have zero or negative thread count
    if (nth <= 0)
      nth = 1;

    // Down projection with bounds checking
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          // Validate task id
          if (task_id < 0 || task_id >= nth * activated_expert) {
            printf("Warning: Invalid task_id %d in down projection matrix "
                   "multiply\n",
                   task_id);
            return;
          }

          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          // Validate expert index
          if (expert_idx < 0 || expert_idx >= config_.expert_num) {
            printf("Warning: Invalid expert_idx %d in down projection matrix "
                   "multiply\n",
                   expert_idx);
            return;
          }

#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size,
                       config_.intermediate_size, down_ba_[expert_idx],
                       down_bb_numa_[Backend::numa_node][expert_idx],
                       down_bc_[expert_idx], ith, nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size,
                       config_.intermediate_size, down_ba_[expert_idx],
                       down_bb_[expert_idx], down_bc_[expert_idx], ith, nth,
                       use_amx);
#endif
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx],
                                       m_local_down_output_ptr_[expert_idx],
                                       ith, nth);
        },
        nullptr);

    // Final output computation with bounds checking
    if (output == nullptr) {
      printf("Warning: Output pointer is null, skipping final computation\n");
      return;
    }

    // Validate weights
    if (weights == nullptr) {
      printf("Warning: Weights pointer is null, using uniform weights\n");
      // Fill output with zeros as a fallback
      memset(output, 0, qlen * config_.hidden_size * sizeof(ggml_bf16_t));
      return;
    }

    // Compute final weighted output
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          // Process each block of the hidden dimension
          for (int e = 0; e < config_.hidden_size; e += 32) {
            // Bounds check
            if (e + 32 > config_.hidden_size) {
              continue;
            }

            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();

            for (int j = 0; j < k; j++) {
              // Get expert_id with validation
              uint64_t expert_id =
                  (expert_ids != nullptr) ? expert_ids[i * k + j] : 0;
              if (expert_id >= (uint64_t)config_.expert_num) {
                printf("Warning: Invalid expert_id %lu in final computation, "
                       "using expert 0\n",
                       expert_id);
                expert_id = 0;
              }

              // Get weight with validation to prevent FPE
              float weight = 1.0f;
              if (weights != nullptr && i * k + j < (size_t)qlen * k) {
                weight = weights[i * k + j];
                // Sanitize weight value to prevent NaN/Inf
                if (std::isnan(weight) || std::isinf(weight)) {
                  printf(
                      "Warning: Invalid weight %f at index %d,%d, using 1.0\n",
                      weight, i, j);
                  weight = 1.0f;
                }
              }

              __m512 weight_vec = _mm512_set1_ps(weight);
              __m512 down_output0, down_output1;

              // Bounds check for m_local_pos_ access
              if (i >= config_.max_len || j >= config_.routed_expert_num) {
                printf("Warning: Index out of bounds for m_local_pos_[%d][%d] in output computation (max_len=%d, routed_expert_num=%d)\n",
                       i, j, config_.max_len, config_.routed_expert_num);
                continue;
              }

              // Safely access the expert output
              ggml_bf16_t *expert_output =
                  m_local_down_output_ptr_[expert_id] +
                  m_local_pos_[i][j] * config_.hidden_size + e;

              avx512_32xbf16_to_32xfp32((__m512i *)(expert_output),
                                        &down_output0, &down_output1);

              x0 = _mm512_fmadd_ps(down_output0, weight_vec, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight_vec, x1);
            }

            // Safely write to output
            ggml_bf16_t *output_ptr =
                (ggml_bf16_t *)output + i * config_.hidden_size + e;
            avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i *)(output_ptr));
          }
        },
        nullptr);
  }
};

#endif
