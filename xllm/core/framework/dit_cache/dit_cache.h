/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "dit_cache_impl.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

namespace xllm {

class DiTCache {
 private:
  struct RegionEPrefetchedKV {
    int64_t block_id = -1;
    torch::Tensor key;
    torch::Tensor value;
#if defined(USE_NPU)
    std::shared_ptr<c10_npu::NPUEvent> ready_event;
#endif
  };

 public:
  DiTCache() = default;
  ~DiTCache() = default;

  DiTCache(const DiTCache&) = delete;
  DiTCache& operator=(const DiTCache&) = delete;
  DiTCache(DiTCache&&) = delete;
  DiTCache& operator=(DiTCache&&) = delete;

  static DiTCache& get_instance() {
    static DiTCache ditcache;
    return ditcache;
  }

  bool init(const DiTCacheConfig& cfg);

  DiTCache(const DiTCacheConfig& cfg) {
    active_cache_ = create_dit_cache(cfg);
    active_cond_cache_ = create_dit_cache(cfg);
    if (!active_cache_ || !active_cond_cache_) {
      LOG(ERROR) << "failed to initialized dit cache, "
                    "please check your config";
    }
    active_cache_->init(cfg);
    active_cond_cache_->init(cfg);
  }

  bool on_before_block(const CacheBlockIn& blockin, bool use_cfg = false);

  CacheBlockOut on_after_block(const CacheBlockIn& blockin,
                               bool use_cfg = false);

  bool on_before_step(const CacheStepIn& stepin, bool use_cfg = false);

  CacheStepOut on_after_step(const CacheStepIn& stepin, bool use_cfg = false);

  virtual void set_infer_steps(const int64_t& infer_steps) {
    regione_infer_steps_ = infer_steps;
    active_cache_->set_infer_steps(infer_steps);
    active_cond_cache_->set_infer_steps(infer_steps);
  }

  virtual void set_num_blocks(const int64_t& num_blocks) {
    regione_num_blocks_ = num_blocks;
    active_cache_->set_num_blocks(num_blocks);
    active_cond_cache_->set_num_blocks(num_blocks);
    if (regione_enabled_) ensure_regione_kv_size(num_blocks);
  }

  bool is_regione_enabled() const { return regione_enabled_; }
  int64_t regione_warmup_steps() const { return config_.regione.warmup_steps; }
  bool regione_has_regions() const { return regione_edited_ids_.defined(); }
  bool regione_is_partial_mode() const { return regione_partial_mode_; }
  bool regione_is_partial_sp_mode() const;
  bool regione_should_compute_velocity(int64_t step) const;
  bool regione_should_run_full_step(int64_t step) const;
  bool regione_should_direct_unedited(int64_t step) const;
  int64_t regione_next_direct_step(int64_t step) const;
  void regione_prepare_inference(const torch::Tensor& latents,
                                 const torch::Tensor& condition_latents,
                                 int64_t grid_h,
                                 int64_t grid_w,
                                 int64_t sp_rank = 0,
                                 int64_t sp_size = 1);
  void regione_select_regions(const torch::Tensor& sample,
                              const torch::Tensor& model_output,
                              const torch::Tensor& sigmas,
                              int64_t step);
  torch::Tensor regione_gather_edited(const torch::Tensor& tensor) const;
  torch::Tensor regione_gather_unedited(const torch::Tensor& tensor) const;
  torch::Tensor regione_scatter_edited(const torch::Tensor& edited,
                                       const torch::Tensor& base) const;
  torch::Tensor regione_scatter_unedited(const torch::Tensor& unedited,
                                         const torch::Tensor& base) const;
  torch::Tensor regione_gather_query_rope(
      const torch::Tensor& image_rope) const;
  torch::Tensor regione_gather_key_rope(const torch::Tensor& image_rope,
                                        int64_t key_len) const;
  torch::Tensor regione_local_update_mask(const torch::Tensor& base) const;
  void regione_update_velocity_cache(const torch::Tensor& value);
  torch::Tensor regione_velocity_cache() const;
  void regione_prefetch_img_kv(int64_t block_id,
                               bool use_cfg,
                               const torch::Tensor& reference);
  void regione_set_current_block(
      int64_t block_id,
      bool use_cfg,
      const torch::Tensor& reference = torch::Tensor());
  void regione_set_current_step(int64_t step);
  void regione_set_partial_mode(bool partial_mode);
  int64_t regione_current_block() const { return regione_current_block_; }
  bool regione_current_use_cfg() const { return regione_current_use_cfg_; }
  bool regione_should_store_kv() const;
  bool regione_should_patch_kv() const;
  void regione_store_img_kv(int64_t block_id,
                            bool use_cfg,
                            const torch::Tensor& key,
                            const torch::Tensor& value);
  std::pair<torch::Tensor, torch::Tensor> regione_patch_img_kv(
      int64_t block_id,
      bool use_cfg,
      const torch::Tensor& key,
      const torch::Tensor& value);
  bool regione_profile_enabled() const;
  void regione_profile_reset_step(int64_t step,
                                  bool partial_step,
                                  bool full_step,
                                  bool velocity_cache,
                                  int64_t step_tokens,
                                  int64_t full_tokens);
  void regione_profile_log_step(double transformer_ms,
                                double arp_ms,
                                double scheduler_ms,
                                double total_ms) const;

 private:
  torch::Tensor get_tensor_or_empty(const TensorMap& m, const std::string& k);
  bool regione_is_refresh_step(int64_t step) const;
  bool regione_is_tail_step(int64_t step) const;
  torch::Tensor regione_normalize_ids(const torch::Tensor& ids,
                                      const torch::Device& device) const;
  torch::Tensor regione_active_edited_ids() const;
  torch::Tensor regione_kv_update_ids() const;
  void regione_update_local_ids();
  torch::Tensor regione_gather_ids(const torch::Tensor& tensor,
                                   const torch::Tensor& ids,
                                   int64_t dim) const;
  torch::Tensor regione_scatter_ids(const torch::Tensor& values,
                                    const torch::Tensor& ids,
                                    const torch::Tensor& base,
                                    int64_t dim) const;
  void ensure_regione_kv_size(int64_t num_blocks);
  std::vector<RegionEPrefetchedKV>& regione_prefetch_slots(bool use_cfg);
  void regione_clear_prefetch_slot(RegionEPrefetchedKV& slot);
  void regione_clear_prefetch_block(bool use_cfg, int64_t block_id);
  void regione_clear_all_prefetch_slots();
  void regione_prefetch_img_kv(int64_t block_id,
                               bool use_cfg,
                               const torch::Device& device,
                               c10::ScalarType dtype);
  bool regione_take_prefetched_img_kv(int64_t block_id,
                                      bool use_cfg,
                                      const torch::Device& device,
                                      c10::ScalarType dtype,
                                      torch::Tensor* key,
                                      torch::Tensor* value);
  void regione_profile_add_kv_store(double ms);
  void regione_profile_add_prefetch_issue(double ms);
  void regione_profile_add_prefetch_hit(double wait_ms);
  void regione_profile_add_prefetch_miss();
  void regione_profile_add_fallback_h2d(double ms);
  void regione_profile_add_patch_scatter(double ms);

  std::unique_ptr<DitCacheImpl> active_cache_;
  std::unique_ptr<DitCacheImpl> active_cond_cache_;
  DiTCacheConfig config_;
  bool regione_enabled_ = false;
  int64_t regione_infer_steps_ = 0;
  int64_t regione_num_blocks_ = 0;
  int64_t regione_current_step_ = 0;
  int64_t regione_current_block_ = -1;
  bool regione_current_use_cfg_ = false;
  bool regione_partial_mode_ = false;
  int64_t regione_target_seq_len_ = 0;
  int64_t regione_grid_h_ = 0;
  int64_t regione_grid_w_ = 0;
  int64_t regione_image_seq_len_ = 0;
  int64_t regione_sp_rank_ = 0;
  int64_t regione_sp_size_ = 1;
  int64_t regione_local_start_ = 0;
  int64_t regione_local_end_ = 0;
  torch::Tensor regione_condition_latents_;
  torch::Tensor regione_edited_ids_;
  torch::Tensor regione_unedited_ids_;
  torch::Tensor regione_local_edited_global_ids_;
  torch::Tensor regione_local_edited_cache_ids_;
  torch::Tensor regione_local_image_global_ids_;
  torch::Tensor regione_velocity_cache_;
  std::vector<torch::Tensor> regione_k_cache_cpu_;
  std::vector<torch::Tensor> regione_v_cache_cpu_;
  std::vector<torch::Tensor> regione_cond_k_cache_cpu_;
  std::vector<torch::Tensor> regione_cond_v_cache_cpu_;
  std::vector<RegionEPrefetchedKV> regione_prefetch_slots_;
  std::vector<RegionEPrefetchedKV> regione_cond_prefetch_slots_;
  int64_t regione_profile_step_ = -1;
  bool regione_profile_partial_step_ = false;
  bool regione_profile_full_step_ = false;
  bool regione_profile_velocity_cache_ = false;
  int64_t regione_profile_step_tokens_ = 0;
  int64_t regione_profile_full_tokens_ = 0;
  int64_t regione_profile_kv_store_count_ = 0;
  int64_t regione_profile_prefetch_issue_count_ = 0;
  int64_t regione_profile_prefetch_hit_count_ = 0;
  int64_t regione_profile_prefetch_miss_count_ = 0;
  int64_t regione_profile_fallback_h2d_count_ = 0;
  int64_t regione_profile_patch_scatter_count_ = 0;
  double regione_profile_kv_store_cpu_ms_ = 0.0;
  double regione_profile_prefetch_issue_ms_ = 0.0;
  double regione_profile_prefetch_wait_ms_ = 0.0;
  double regione_profile_fallback_h2d_ms_ = 0.0;
  double regione_profile_patch_scatter_ms_ = 0.0;
};

}  // namespace xllm
