#ifndef CPUINFER_OPERATOR_MOE_HPP
#define CPUINFER_OPERATOR_MOE_HPP

// #define CHECK

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../cpu_backend/shared_mem_buffer.h"
#include "common.hpp"

// Forward declaration for Llamafile backend type checking
class LLAMA_MOE_TP;

template <typename T>
concept MOE_TP_PART = requires(T t, int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                               void* output, GeneralMOEConfig config, int tp_idx) {
  typename T::output_t;
  { new T(config, tp_idx) } -> std::same_as<T*>;
  { t.forward(qlen, k, expert_ids, weights, input, output) } -> std::same_as<void>;
  // { t.load_weights() } -> std::same_as<void>;
};

template <MOE_TP_PART T>
class TP_MOE_Common : public MoE_Interface {
 protected:
  std::vector<GeneralMOEConfig> tp_configs;
  std::vector<int> tp_intermediate_offsets;  // Starting offset for each TP's intermediate slice
  int tp_count;
  int me_numa_id;
  std::vector<std::unique_ptr<T>> tps;

  std::vector<typename T::output_t*> local_output_numa;
  T::output_t* local_output = nullptr;

  bool weights_loaded = false;

#ifdef FORWARD_TIME_REPORT
  size_t forward_time_sum_ns = 0;
  size_t forward_count = 0;
#endif
 public:
  GeneralMOEConfig config;
  using input_t = typename T::input_t;
  TP_MOE_Common(GeneralMOEConfig config) : config(config) {
    printf("TP MOE layer %d, pool: 0x%lx, expert num: %d, num_experts_per_tok: %d\n", config.layer_idx,
           (intptr_t)config.pool, config.expert_num, config.num_experts_per_tok);
    printf("intermediate_size %d, tp count %d\n", config.intermediate_size, config.pool->config.subpool_count);
    if (config.pool == nullptr) {
      printf("TP MOE layer %d, no worker pool\n", config.layer_idx);
      throw std::runtime_error("no worker pool");
    }

    this->config = config;
    tp_count = config.pool->config.subpool_count;

    // Check if this is Llamafile backend using compile-time type checking
    constexpr bool is_llamafile = std::is_same<T, LLAMA_MOE_TP>::value;
#ifndef QK_K
#define QK_K 256
#endif

    if (is_llamafile) {
      // For Llamafile backend: use QK_K-aligned TP splitting
      if (config.intermediate_size % QK_K != 0) {
        printf("intermediate_size %d must be divisible by QK_K %d for Llamafile backend\n", config.intermediate_size,
               QK_K);
        throw std::runtime_error("intermediate_size must be divisible by QK_K (256) for Llamafile backend");
      }

      int num_blocks = config.intermediate_size / QK_K;
      int base_blocks = num_blocks / tp_count;
      int extra_blocks = num_blocks % tp_count;

      if (base_blocks == 0) {
        printf("intermediate_size %d is too small for tp_count %d (num_blocks=%d)\n", config.intermediate_size,
               tp_count, num_blocks);
        throw std::runtime_error("intermediate_size too small: cannot distribute blocks to all TP instances");
      }

      printf("Llamafile TP splitting: intermediate_size=%d, tp_count=%d, QK_K=%d\n", config.intermediate_size, tp_count,
             QK_K);
      printf("  num_blocks=%d, base_blocks=%d, extra_blocks=%d\n", num_blocks, base_blocks, extra_blocks);

      int current_offset = 0;
      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;

        // First extra_blocks TPs get one more block
        int num_blocks_for_this_tp = base_blocks + (i < extra_blocks ? 1 : 0);
        tp_config.intermediate_size = num_blocks_for_this_tp * QK_K;

        printf("  TP %d: intermediate_size=%d, offset=%d, blocks=%d\n", i, tp_config.intermediate_size, current_offset,
               num_blocks_for_this_tp);

        tp_intermediate_offsets.push_back(current_offset);
        tp_configs.push_back(tp_config);
        current_offset += tp_config.intermediate_size;
      }
    } else {
      // For non-Llamafile backends (AMX): use B_K_STEP-aligned distribution to support CXL/arbitrary NUMA counts
      // AMX BufferA requires k to be a multiple of K_STEP (64), BufferC requires n to be a multiple of N_STEP (32)
      // INT4 BufferB requires k to be a multiple of B_K_STEP (2*K_STEP = 128)
      // Using 128 as block size satisfies all constraints (128 is multiple of 64 and 32)
      constexpr int K_STEP = 128;  // B_K_STEP for INT4 compatibility

      if (config.intermediate_size % K_STEP != 0) {
        printf("intermediate_size %d must be divisible by B_K_STEP %d for AMX INT4 backend\n", config.intermediate_size,
               K_STEP);
        throw std::runtime_error("intermediate_size must be divisible by B_K_STEP (128) for AMX INT4 backend");
      }

      int num_blocks = config.intermediate_size / K_STEP;

      // Check if weight ratios are specified for weighted distribution
      bool use_weighted = config.pool->config.subpool_weight_ratios.size() == (size_t)tp_count;
      std::vector<int> blocks_per_tp(tp_count);

      if (use_weighted) {
        // Weighted distribution: distribute blocks according to weight ratios
        int total_weight = 0;
        for (int i = 0; i < tp_count; i++) {
          total_weight += config.pool->config.subpool_weight_ratios[i];
        }

        printf("AMX TP splitting (weighted): intermediate_size=%d, tp_count=%d, B_K_STEP=%d\n",
               config.intermediate_size, tp_count, K_STEP);
        printf("  num_blocks=%d, total_weight=%d, weight_ratios=[", num_blocks, total_weight);
        for (int i = 0; i < tp_count; i++) {
          printf("%d%s", config.pool->config.subpool_weight_ratios[i], i < tp_count - 1 ? ":" : "]\n");
        }

        // Calculate blocks for each TP based on weight ratio
        int assigned_blocks = 0;
        for (int i = 0; i < tp_count; i++) {
          if (i == tp_count - 1) {
            // Last TP gets remaining blocks to ensure total matches
            blocks_per_tp[i] = num_blocks - assigned_blocks;
          } else {
            // Calculate proportional blocks, rounded to nearest
            blocks_per_tp[i] =
                (num_blocks * config.pool->config.subpool_weight_ratios[i] + total_weight / 2) / total_weight;
            // Ensure at least 1 block per TP
            if (blocks_per_tp[i] < 1) blocks_per_tp[i] = 1;
          }
          assigned_blocks += blocks_per_tp[i];
        }

        // Verify last TP gets at least 1 block
        if (blocks_per_tp[tp_count - 1] < 1) {
          printf("ERROR: Not enough blocks (%d) to satisfy weight ratios for tp_count=%d\n", num_blocks, tp_count);
          throw std::runtime_error("intermediate_size too small: cannot satisfy weight ratios");
        }
      } else {
        // Even distribution: distribute blocks evenly with remainder to first TPs
        int base_blocks = num_blocks / tp_count;
        int extra_blocks = num_blocks % tp_count;

        if (base_blocks == 0) {
          printf("intermediate_size %d is too small for tp_count %d (num_blocks=%d)\n", config.intermediate_size,
                 tp_count, num_blocks);
          throw std::runtime_error("intermediate_size too small: cannot distribute blocks to all TP instances");
        }

        printf("AMX TP splitting (even): intermediate_size=%d, tp_count=%d, B_K_STEP=%d\n", config.intermediate_size,
               tp_count, K_STEP);
        printf("  num_blocks=%d, base_blocks=%d, extra_blocks=%d\n", num_blocks, base_blocks, extra_blocks);

        for (int i = 0; i < tp_count; i++) {
          blocks_per_tp[i] = base_blocks + (i < extra_blocks ? 1 : 0);
        }
      }

      int current_offset = 0;
      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;

        tp_config.intermediate_size = blocks_per_tp[i] * K_STEP;

        printf("  TP %d: intermediate_size=%d, offset=%d, blocks=%d\n", i, tp_config.intermediate_size, current_offset,
               blocks_per_tp[i]);

        tp_intermediate_offsets.push_back(current_offset);
        tp_configs.push_back(tp_config);
        current_offset += tp_config.intermediate_size;
      }
    }

    config.pool->dispense_backend()->do_numa_job(
        [this, config](int i) { tps[i] = std::move(std::unique_ptr<T>(new T(tp_configs[i], i))); });

    local_output_numa.resize(tp_count, nullptr);
    MemoryRequest mem_requests;
    for (auto i = 0; i < tp_count; i++) {
      mem_requests.append_pointer(
          &local_output_numa[i],
          (size_t)sizeof(typename T::output_t) * tp_configs[i].max_possible_qlen() * tp_configs[i].hidden_size);
    }
    mem_requests.append_pointer(
        (void**)&local_output,
        sizeof(typename T::output_t) * tp_configs[0].max_possible_qlen() * tp_configs[0].hidden_size);
    // printf("local output tp, %d,\n", tp_configs[0].max_possible_qlen());
    shared_mem_buffer.alloc(this, mem_requests);
  }

  void warm_up() {
    int qlen = config.max_possible_qlen();
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config.num_experts_per_tok);
    std::vector<float> weights(qlen * config.num_experts_per_tok);
    for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
      expert_ids[i] = i % config.expert_num;
      weights[i] = 0.01;
    }
    forward(&qlen, config.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data(), false);
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental = false) {
    int qlen_local = qlen;
    forward(&qlen_local, k, expert_ids, weights, input, output, incremental);
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    forward(qlen_ptr, k, expert_ids, weights, input, output, false);
  }

  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
#ifdef FORWARD_TIME_REPORT
    auto start = std::chrono::high_resolution_clock::now();
#endif
    int qlen = *qlen_ptr;

    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });

    merge_results(qlen, output, incremental);
#ifdef FORWARD_TIME_REPORT
    auto end = std::chrono::high_resolution_clock::now();
    auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    int unique_experts = 0;
    {
      std::unordered_set<int64_t> expert_set;
      for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
        expert_set.insert(expert_ids[i]);
      }
      unique_experts = expert_set.size();
    }
    auto band_width =
        (1.0 * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) / (1.0 * forward_time / 1e6);
    auto GFLOPS =
        (1.0 * config.hidden_size * config.intermediate_size * qlen * 3 * config.num_experts_per_tok * 2 / 1e9) /
        (1.0 * forward_time / 1e6);
    if (qlen <= 10) {
      forward_time_sum_ns += forward_time;
      forward_count++;
    }
    auto average_bandwidth =
        (1.0 * forward_count * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) /
        (1.0 * forward_time_sum_ns / 1e6);
    printf(
        "forward time %ld, time stamp:%ld, band width %f GElement/s, ave bandwidth %f GElement/s (only "
        "decode), %f GFLOPS, me numa: %d\n",
        forward_time, end.time_since_epoch().count() / 1000 % 100000000, band_width, average_bandwidth, GFLOPS,
        numa_node_of_cpu(sched_getcpu()));
#endif
  }

  virtual void load_weights() = 0;

  virtual void merge_results(int qlen, void* output) = 0;

  virtual void merge_results(int qlen, void* output, bool incremental) {
    if (incremental == false) {
      merge_results(qlen, output);
    } else {
      throw std::runtime_error("Not Implemented");
    }
  };
};

template <MOE_TP_PART T>
class TP_MOE : public TP_MOE_Common<T> {
 public:
  using TP_MOE_Common<T>::TP_MOE_Common;
  void load_weights(const uint64_t* physical_to_logical_map) { throw std::runtime_error("Not Implemented"); }
  // void merge_results(int qlen, void *output, bool incremental) { throw std::runtime_error("Not Implemented"); }
};

#endif
