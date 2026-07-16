#include <cstdlib>
#include <cstring>
#include "core/rs_context.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "utils/rs_log.h"

// Include corresponding backend headers based on macro definitions
#ifdef RS_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef RS_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef RS_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef RS_USE_CANN
#include "ggml-cann.h"
#endif
#ifdef RS_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Global registry for model architectures.
// Stored as a raw pointer so ASYNCIFY rewind cannot reset it — rewind
// re-runs global-object constructors but does not free heap allocations,
// so the map survives across suspend/resume cycles.
static std::unordered_map<std::string, ModelCreator> *s_model_registry = nullptr;

static std::unordered_map<std::string, ModelCreator> &get_model_registry() {
  if (!s_model_registry)
    s_model_registry = new std::unordered_map<std::string, ModelCreator>();
  return *s_model_registry;
}

void rs_register_model_arch(const std::string &arch, ModelCreator creator) {
  get_model_registry()[arch] = creator;
}

rs_context_t::rs_context_t()
    : sched(nullptr), ctx_gguf(nullptr), gguf_data(nullptr) {}

rs_context_t::~rs_context_t() {
  // IMPORTANT: Clear processor and model first.
  // This ensures that any RSState or persistent backend buffers (especially
  // Metal resources) are deallocated while the backends and scheduler are still
  // valid.
  processor.reset();
  model.reset();

  // Free weight buffers explicitly to satisfy Metal residency set assertions
  for (auto buf : weight_buffers) {
    ggml_backend_buffer_free(buf);
  }
  weight_buffers.clear();

  // Now it is safe to free the scheduler as all its managed tensors/buffers are
  // gone
  if (sched_cpu)
    ggml_backend_sched_free(sched_cpu);
  if (sched)
    ggml_backend_sched_free(sched);

  // Free all backend instances
  for (auto b : backends)
    ggml_backend_free(b);

  // Cleanup GGUF related resources
  if (ctx_gguf)
    gguf_free(ctx_gguf);
  if (gguf_data)
    ggml_free(gguf_data);
}

/**
 * Initialize backends: try CUDA -> Metal -> Vulkan -> CANN -> CPU in order of
 * priority
 */
bool rs_context_t::init_backend(bool prefer_cpu) {
  bool gpu_initialized = false;

  // Skip GPU for small models with many fine-grained ops (OpenVoice2)
  // that trigger Metal command-buffer errors on GPU.
  if (prefer_cpu) {
    RS_LOG_INFO("Model prefers CPU — skipping GPU backend init");
  }

  if (params.use_gpu && !prefer_cpu) {
#ifdef RS_USE_CUDA
    if (!gpu_initialized) {
      ggml_backend_t b = ggml_backend_cuda_init(0); // Use device 0 by default
      if (b) {
        backends.push_back(b);
        RS_LOG_INFO("CUDA backend added to scheduler.");
        gpu_initialized = true;
      }
    }
#endif

#ifdef RS_USE_METAL
    if (!gpu_initialized) {
      ggml_backend_reg_t metal_reg = ggml_backend_metal_reg();
      if (metal_reg) {
        int n_dev = ggml_backend_reg_dev_count(metal_reg);
        RS_LOG_INFO("Metal registry found: %d device(s)", n_dev);
        ggml_backend_t b = ggml_backend_metal_init();
        if (b) {
          backends.push_back(b);
          RS_LOG_INFO("Metal backend added to scheduler.");
          gpu_initialized = true;
        } else {
          RS_LOG_WARN("Metal registry OK but ggml_backend_metal_init() returned NULL");
        }
      } else {
        RS_LOG_WARN("ggml_backend_metal_reg() returned NULL — Metal backend not registered");
      }
    }
#endif

#ifdef RS_USE_VULKAN
    if (!gpu_initialized) {
      ggml_backend_t b = ggml_backend_vk_init(0);
      if (b) {
        backends.push_back(b);
        RS_LOG_INFO("Vulkan backend added to scheduler.");
        gpu_initialized = true;
      }
    }
#endif

#ifdef RS_USE_CANN
    if (!gpu_initialized) {
      ggml_backend_t b = ggml_backend_cann_init(0);
      if (b) {
        backends.push_back(b);
        RS_LOG_INFO("CANN backend added to scheduler.");
        gpu_initialized = true;
      }
    }
#endif

#ifdef RS_USE_WEBGPU
    // WebGPU: portable GPU backend usable both natively (via Dawn) and in
    // Emscripten/browser builds. Tried last among GPU backends because the
    // native vendor-specific paths (CUDA/Metal/Vulkan/CANN) are typically
    // faster when available.
    if (!gpu_initialized) {
      ggml_backend_t b = ggml_backend_webgpu_init();
      if (b) {
        backends.push_back(b);
        RS_LOG_INFO("WebGPU backend added to scheduler.");
        gpu_initialized = true;
      } else {
        RS_LOG_WARN("ggml_backend_webgpu_init() returned NULL — no compatible "
                    "WebGPU adapter found");
      }
    }
#endif

    if (!gpu_initialized) {
      RS_LOG_WARN("GPU requested but no supported GPU backend could be "
                  "initialized. Falling back to CPU.");
    }
  }

  // Always add CPU backend as a fallback or for collaborative computing
  ggml_backend_t cpu = ggml_backend_cpu_init();
  if (cpu) {
    ggml_backend_cpu_set_n_threads(cpu, params.n_threads);
    backends.push_back(cpu);
  } else {
    RS_LOG_ERR("Failed to initialize CPU backend.");
    return false;
  }

  // Initialize the scheduler to distribute computation across backends.
  // ggml_backend_sched_new requires the LAST backend to be CPU type.
  // When GPU is available, op_offload=true keeps computation on GPU and only
  // uses CPU fallback for unsupported ops — avoids cross-device copies that
  // hurt small-model throughput.
  if (gpu_initialized) {
    sched = ggml_backend_sched_new(backends.data(), nullptr,
                                   (int)backends.size(), 16384, false, true);
    RS_LOG_INFO("Scheduler: GPU+CPU (op_offload=true, %d backend(s))",
                (int)backends.size());
    // CPU-only scheduler for batch-1 decode loops (Qwen3-ASR). The CPU backend is
    // last in `backends`. Lets the autoregressive decode run on CPU (3.7x faster
    // at batch-1 on Maxwell sm_53) while prefill/encode use the GPU `sched`.
    ggml_backend_t cpu_only[1] = {backends[(int)backends.size() - 1]};
    sched_cpu = ggml_backend_sched_new(cpu_only, nullptr, 1, 16384, false, false);
    RS_LOG_INFO("CPU-only decode scheduler (sched_cpu) created: %s",
                sched_cpu ? "OK" : "FAILED(null)");
  } else {
    int cpu_idx = (int)backends.size() - 1;
    sched = ggml_backend_sched_new(&backends[cpu_idx], nullptr, 1, 16384, false,
                                   false);
    RS_LOG_INFO("Scheduler: CPU-only mode");
  }
  return sched != nullptr;
}

rs_context_t *rs_context_init_internal(rs_init_params_t params) {
  auto ctx = std::make_unique<rs_context_t>();
  ctx->params = params;

  // 1. Load GGUF handle early to detect architecture before backend init.
  //    Small TTS models (OpenVoice2) have many fine-grained ops that cause
  //    Metal GPU command-buffer failures; they need CPU-only backends.
  struct gguf_init_params g_params = {/*.no_alloc =*/true,
                                      /*.ctx      =*/&ctx->gguf_data};

  ctx->ctx_gguf = gguf_init_from_file(params.model_path, g_params);
  if (!ctx->ctx_gguf) {
    RS_LOG_ERR("Failed to load GGUF file: %s", params.model_path);
    return nullptr;
  }

  // 2. Detect architecture before backend init
  int64_t arch_key = gguf_find_key(ctx->ctx_gguf, "general.architecture");
  if (arch_key == -1) {
    RS_LOG_ERR("GGUF file missing 'general.architecture' key.");
    return nullptr;
  }
  std::string arch = gguf_get_val_str(ctx->ctx_gguf, arch_key);
  RS_LOG_INFO("Architecture detected: %s", arch.c_str());

  // Models with many fine-grained GPU-hostile ops (HiFi-GAN vocoder, etc.)
  // run faster on CPU and trigger Metal command-buffer errors on GPU.
  // openvoice2's CPU preference is a METAL workaround (command-buffer errors
  // on many fine-grained ops); CUDA handles the graph fine — keep GPU there.
#ifdef RS_USE_CUDA
  // matcha-tts has a very fine-grained CFM-decoder graph (3-step ODE x UNet); the CPU path is
  // faster for it and avoids the GB10 sm_53->sm_121 PTX-JIT stall on first use. Its forward path
  // is backend-agnostic (ggml_backend_sched) though, so it CAN target CUDA — opt in with
  // MATCHA_USE_CUDA=1 (useful on Ampere+, where ggml's CUDA-graph replay (cc>=800) actually helps).
  bool prefer_cpu = (arch == "matcha-tts" && getenv("MATCHA_USE_CUDA") == nullptr);
#else
  bool prefer_cpu = (arch == "openvoice2" || arch == "matcha-tts");
#endif

  // Qwen3-ASR: by default keep ALL weights on the GPU (encoder + LLM prefill +
  // decode all on the GPU scheduler). Measured on Jetson Nano gen1 (sm_53): all-GPU
  // gives RTF ~1.44, whereas placing weights on CPU to run the decode on a CPU
  // scheduler BACKFIRES here — RapidSpeech uses a single weight buffer, so CPU
  // weights also put the 185M encoder on CPU and op_offload re-uploads it per op
  // (encoder 5s -> 16s). The CPU-decode split only wins with per-component weight
  // buffers (as in the standalone engine). Opt in for experiments with
  // RS_QWEN3ASR_CPU_WEIGHTS=1 (puts weights on CPU + runs decode on sched_cpu).
  bool cpu_weights = (arch == "Qwen3ASR" && getenv("RS_QWEN3ASR_CPU_WEIGHTS") != nullptr);

  // 3. Hardware detection and backend initialization
  if (!ctx->init_backend(prefer_cpu)) {
    RS_LOG_ERR("Failed to initialize backend scheduler.");
    return nullptr;
  }

  // 4. Allocate physical memory buffers on the appropriate backend.
  //    Qwen3-ASR DEFAULT (GPU present): ALL weights on GPU (encoder + LLM). On
  //    Jetson Nano gen1 (sm_53) this is the fastest config measured (RTF ~1.3):
  //    GPU-persistent decode is ~460ms/tok. The PER-COMPONENT split (encoder GPU
  //    + LLM CPU + CPU decode) is correct and the standalone engine reaches
  //    195ms/tok with it, but RapidSpeech's per-token LLM decode path is ~4x
  //    slower on CPU (~790ms/tok, per-token graph-build/alloc overhead), so the
  //    split is currently a LOSS here (RTF ~2.1). Keep it opt-in for when that
  //    decode path is optimized or on hardware where CPU decode wins.
  //    Opt-in: RS_QWEN3ASR_SPLIT=1 (per-component). RS_QWEN3ASR_CPU_WEIGHTS=1 (all-CPU).
  const bool have_gpu = (int)ctx->backends.size() > 1;
  // MOSS-Transcribe-Diarize: smart placement default, measured on Jetson Nano
  // gen1 (sm_53, 12 s clip): all-GPU = 44 s total (prefill 3.0 s, decode
  // 554 ms/tok) vs the per-component split = 81 s (prefill collapses to 19 s —
  // CPU-resident LLM weights get re-uploaded per op — and RapidSpeech's CPU
  // decode is ~890 ms/tok from per-token graph overhead). So ALL-GPU wins
  // 2.5x... but only if the token-embedding type has a CUDA get_rows kernel:
  // the decode's per-token embedding lookup on a q6_K/K-quant embed aborts
  // (unsupported src type). Default: all-GPU when the embed is get_rows-
  // compatible (f16/f32/q4_0/q4_1/q5_0/q5_1/q8_0 — e.g. the *embq8* gguf),
  // per-component split otherwise (correct everywhere, slower). Overrides:
  // RS_MOSS_GPU_WEIGHTS=1 forces all-GPU, RS_MOSS_SPLIT=1 forces the split.
  bool moss_embed_gpu_ok = false;
  if (arch == "MossTD" && have_gpu) {
    ggml_tensor *emb =
        ggml_get_tensor(ctx->gguf_data, "llm.model.embed_tokens.weight");
    if (emb) {
      switch (emb->type) {
      case GGML_TYPE_F16: case GGML_TYPE_F32:
      case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
      case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
      case GGML_TYPE_Q8_0:
        moss_embed_gpu_ok = true; break;
      default:
        RS_LOG_WARN("MossTD: token embed type %s has no CUDA get_rows — "
                    "using per-component split (encoder GPU, LLM/decode CPU). "
                    "For ~2x faster decode use an embq8 gguf "
                    "(rs-quantize --token-embedding-type q8_0).",
                    ggml_type_name(emb->type));
        break;
      }
    }
  }
  const bool split_weights =
      have_gpu &&
      ((arch == "Qwen3ASR" && getenv("RS_QWEN3ASR_SPLIT") != nullptr) ||
       (arch == "MossTD" && getenv("RS_MOSS_GPU_WEIGHTS") == nullptr &&
        (getenv("RS_MOSS_SPLIT") != nullptr ||
         getenv("RS_MOSS_ENC_CPU") != nullptr || !moss_embed_gpu_ok)));
  // RS_MOSS_ENC_CPU=1 INVERTS the split: encoder/adaptor stay on CPU and the
  // LLM goes to GPU. Rationale (Jetson Nano): the encoder's long-sequence
  // attention kernels on a display-driving Tegra exceed the CUDA launch
  // watchdog on >30 s windows, while LLM prefill/decode kernels are small and
  // batch-tiled; this placement dodges the watchdog and keeps the ~2.5x GPU
  // decode win. Requires a get_rows-compatible token embed (embq8 gguf).
  const bool enc_cpu = getenv("RS_MOSS_ENC_CPU") != nullptr;
  auto is_encoder_tensor = [enc_cpu](const char *nm) {
    const bool enc = nm && (strncmp(nm, "a.", 2) == 0 || strncmp(nm, "mm.", 3) == 0);
    return enc_cpu ? !enc : enc;   // inverted: GPU group = everything BUT encoder
  };
  if (split_weights) {
    ggml_backend_t gpu_be = ctx->backends[0];
    ggml_backend_t cpu_be = ctx->backends[(int)ctx->backends.size() - 1];
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_be);
    const size_t align = ggml_backend_buft_get_alignment(gpu_buft);
    // size the GPU (encoder/projector) group
    size_t gpu_size = 0; int gpu_n = 0;
    for (struct ggml_tensor *t = ggml_get_first_tensor(ctx->gguf_data); t;
         t = ggml_get_next_tensor(ctx->gguf_data, t)) {
      if (is_encoder_tensor(t->name)) {
        gpu_size += ((ggml_nbytes(t) + align - 1) / align) * align;
        gpu_n++;
      }
    }
    // Generous headroom: tallocr re-aligns each tensor's offset, so reserve an
    // extra `align` per tensor plus 1 MiB to avoid edge-case overflow.
    ggml_backend_buffer_t gpu_buf = ggml_backend_buft_alloc_buffer(
        gpu_buft, gpu_size + (size_t)(gpu_n + 8) * align + (1u << 20));
    if (!gpu_buf) {
      RS_LOG_ERR("Qwen3ASR: failed to alloc GPU encoder buffer (%zu B)", gpu_size);
      return nullptr;
    }
    struct ggml_tallocr ta = ggml_tallocr_new(gpu_buf);
    for (struct ggml_tensor *t = ggml_get_first_tensor(ctx->gguf_data); t;
         t = ggml_get_next_tensor(ctx->gguf_data, t)) {
      if (is_encoder_tensor(t->name)) ggml_tallocr_alloc(&ta, t);
    }
    ctx->weight_buffers.push_back(gpu_buf);
    // remaining tensors (the LLM, still NULL-buffer) -> CPU
    ggml_backend_buffer_t cpu_buf =
        ggml_backend_alloc_ctx_tensors(ctx->gguf_data, cpu_be);
    if (cpu_buf) ctx->weight_buffers.push_back(cpu_buf);
    RS_LOG_INFO("Qwen3ASR per-component weights: %d encoder/proj tensors on GPU "
                "(%zu MB), LLM on CPU (decode runs on CPU sched)",
                gpu_n, gpu_size >> 20);
  } else {
    // Single-buffer path (all other models; or Qwen3-ASR all-GPU/all-CPU override).
    int weight_backend_idx =
        (prefer_cpu || cpu_weights) ? (int)ctx->backends.size() - 1 : 0;
    ggml_backend_buffer_t weight_buffer =
        ggml_backend_alloc_ctx_tensors(ctx->gguf_data, ctx->backends[weight_backend_idx]);
    if (weight_buffer) ctx->weight_buffers.push_back(weight_buffer);
  }

  // 5. Load tensor data from the binary blob in the file
  FILE *f = fopen(params.model_path, "rb");
  if (!f) {
    RS_LOG_ERR("Failed to open model file for data loading: %s",
               params.model_path);
    return nullptr;
  }

  size_t data_offset = gguf_get_data_offset(ctx->ctx_gguf);
  int64_t n_tensors = gguf_get_n_tensors(ctx->ctx_gguf);
  std::vector<char> read_buf;

  for (int64_t i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx->ctx_gguf, i);
    struct ggml_tensor *t = ggml_get_tensor(ctx->gguf_data, name);

    if (t) {
      size_t t_offset = gguf_get_tensor_offset(ctx->ctx_gguf, i);
      size_t t_size = ggml_nbytes(t);

      if (t_size > 0) {
        if (read_buf.size() < t_size)
          read_buf.resize(t_size);

        fseek(f, data_offset + t_offset, SEEK_SET);
        if (fread(read_buf.data(), 1, t_size, f) != t_size) {
          RS_LOG_ERR("Failed to read data for tensor: %s", name);
          fclose(f);
          return nullptr;
        }

        ggml_backend_tensor_set(t, read_buf.data(), 0, t_size);
      }
    }
  }
  fclose(f);

  // 6. Create model instance from registered architectures
  auto it = get_model_registry().find(arch);
  if (it == get_model_registry().end()) {
    RS_LOG_ERR("Unsupported architecture: %s", arch.c_str());
    return nullptr;
  }

  ctx->model = it->second();

  // 7. Initialize processor and audio frontend
  ctx->processor = std::make_unique<RSProcessor>(ctx->model, ctx->sched);

  // 8. Execute model-specific Load (e.g., mapping pointers, loading CMVN)
  //    Use the correct backend: CPU-preferring models load on CPU.
  ggml_backend_t load_backend = ctx->backends[prefer_cpu ? (int)ctx->backends.size() - 1 : 0];
  if (!ctx->model->Load(ctx, load_backend)) {
    RS_LOG_ERR("Model load failed.");
    return nullptr;
  }

  RS_LOG_INFO("RapidSpeech context core initialized successfully.");
  return ctx.release();
}

/**
 * Helper to safely initialize a ggml context and graph.
 * Prevents 0x0 crashes by checking allocation results.
 */
bool init_compute_ctx(struct ggml_context **ctx, struct ggml_cgraph **gf,
                      int n_nodes) {
  // We add 1MB of buffer to the tensor overhead to be safe
  size_t mem_size = n_nodes * ggml_tensor_overhead() + (1024 * 1024);
  struct ggml_init_params params = {mem_size, nullptr, true};
  *ctx = ggml_init(params);
  if (!(*ctx)) {
    RS_LOG_ERR("ggml_init failed: out of memory for context.");
    return false;
  }
  *gf = ggml_new_graph_custom(*ctx, n_nodes, false);
  if (!(*gf)) {
    RS_LOG_ERR(
        "ggml_new_graph_custom failed: too many nodes or out of memory.");
    return false;
  }
  return true;
}