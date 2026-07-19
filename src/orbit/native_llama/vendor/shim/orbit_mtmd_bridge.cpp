#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef ORBIT_LLAMA_UPSTREAM_COMMIT
#error "ORBIT_LLAMA_UPSTREAM_COMMIT must be provided"
#endif
#ifndef ORBIT_LLAMA_UPSTREAM_TAG
#error "ORBIT_LLAMA_UPSTREAM_TAG must be provided"
#endif
#ifndef ORBIT_LLAMA_SOURCE_TREE_HASH
#error "ORBIT_LLAMA_SOURCE_TREE_HASH must be provided"
#endif
#ifndef ORBIT_LLAMA_PATCHSET_HASH
#error "ORBIT_LLAMA_PATCHSET_HASH must be provided"
#endif
#ifndef ORBIT_LLAMA_BUILD_FLAGS
#error "ORBIT_LLAMA_BUILD_FLAGS must be provided"
#endif

#if defined(_WIN32)
#define ORBIT_EXPORT __declspec(dllexport)
#else
#define ORBIT_EXPORT __attribute__((visibility("default")))
#endif

namespace {

thread_local std::string last_error;

struct orbit_mtmd_context {
    mtmd_context * value = nullptr;
    std::string media_marker;
};

struct orbit_mtmd_bitmap {
    mtmd_bitmap * value = nullptr;
    void * auxiliary = nullptr;
};

template <typename T, typename = void>
struct has_batch_max_tokens : std::false_type {};

template <typename T>
struct has_batch_max_tokens<T, std::void_t<decltype(std::declval<T &>().batch_max_tokens)>> : std::true_type {};

template <typename T, typename = void>
struct has_progress_callback : std::false_type {};

template <typename T>
struct has_progress_callback<T, std::void_t<
    decltype(std::declval<T &>().progress_callback),
    decltype(std::declval<T &>().progress_callback_user_data)>> : std::true_type {};

template <typename T, typename = void>
struct has_text_len : std::false_type {};

template <typename T>
struct has_text_len<T, std::void_t<decltype(std::declval<T &>().text_len)>> : std::true_type {};

template <typename T, typename = void>
struct has_bitmap_wrapper : std::false_type {};

template <typename T>
struct has_bitmap_wrapper<T, std::void_t<
    decltype(std::declval<T &>().bitmap),
    decltype(std::declval<T &>().video_ctx)>> : std::true_type {};

using bitmap_init_result = decltype(mtmd_helper_bitmap_init_from_buf(
    static_cast<mtmd_context *>(nullptr),
    static_cast<const unsigned char *>(nullptr),
    size_t{0},
    false));

template <typename Params>
void initialize_optional_context_fields(Params & params) {
    if constexpr (has_batch_max_tokens<Params>::value) {
        params.batch_max_tokens = 1024;
    }
    if constexpr (has_progress_callback<Params>::value) {
        params.progress_callback = nullptr;
        params.progress_callback_user_data = nullptr;
    }
}

template <typename Text>
void initialize_text_length(Text & text, size_t size) {
    if constexpr (has_text_len<Text>::value) {
        text.text_len = size;
    }
}

template <typename Result>
orbit_mtmd_bitmap * wrap_bitmap_result(Result result) {
    auto handle = std::make_unique<orbit_mtmd_bitmap>();
    if constexpr (std::is_pointer_v<Result>) {
        handle->value = result;
    } else if constexpr (has_bitmap_wrapper<Result>::value) {
        handle->value = result.bitmap;
        handle->auxiliary = result.video_ctx;
    }
    if (!handle->value) {
        return nullptr;
    }
    return handle.release();
}

const char * abi_profile() {
#ifdef ORBIT_MTMD_FORCE_UNSUPPORTED_ABI
    return "unsupported";
#endif
    const bool common_offsets =
        offsetof(mtmd_context_params, use_gpu) == 0 &&
        offsetof(mtmd_context_params, print_timings) == 1 &&
        offsetof(mtmd_context_params, n_threads) == 4 &&
        offsetof(mtmd_context_params, image_marker) == 8 &&
        offsetof(mtmd_context_params, media_marker) == 16 &&
        offsetof(mtmd_context_params, flash_attn_type) == 24 &&
        offsetof(mtmd_context_params, warmup) == 28 &&
        offsetof(mtmd_context_params, image_min_tokens) == 32 &&
        offsetof(mtmd_context_params, image_max_tokens) == 36 &&
        offsetof(mtmd_context_params, cb_eval) == 40 &&
        offsetof(mtmd_context_params, cb_eval_user_data) == 48 &&
        alignof(mtmd_context_params) == 8;
    if (!common_offsets) {
        return "unsupported";
    }
    if (!has_batch_max_tokens<mtmd_context_params>::value &&
        !has_progress_callback<mtmd_context_params>::value &&
        sizeof(mtmd_context_params) == 56) {
        return "mtmd-context-v1";
    }
    if (has_batch_max_tokens<mtmd_context_params>::value &&
        has_progress_callback<mtmd_context_params>::value &&
        sizeof(mtmd_context_params) == 80) {
        return "mtmd-context-v2";
    }
    return "unsupported";
}

template <typename Params>
size_t optional_batch_max_tokens_offset() {
    if constexpr (has_batch_max_tokens<Params>::value) {
        return offsetof(Params, batch_max_tokens);
    }
    return SIZE_MAX;
}

template <typename Params>
size_t optional_progress_callback_offset() {
    if constexpr (has_progress_callback<Params>::value) {
        return offsetof(Params, progress_callback);
    }
    return SIZE_MAX;
}

template <typename Text>
size_t optional_text_len_offset() {
    if constexpr (has_text_len<Text>::value) {
        return offsetof(Text, text_len);
    }
    return SIZE_MAX;
}

template <typename Result>
const char * bitmap_result_profile_for() {
    if constexpr (std::is_pointer_v<Result>) {
        return sizeof(Result) == sizeof(void *) ? "pointer-v1" : "unsupported";
    }
    if constexpr (has_bitmap_wrapper<Result>::value) {
        return sizeof(Result) == 2 * sizeof(void *) &&
                alignof(Result) == alignof(void *) &&
                offsetof(Result, bitmap) == 0 &&
                offsetof(Result, video_ctx) == sizeof(void *)
            ? "wrapper-v1"
            : "unsupported";
    }
    return "unsupported";
}

const char * bitmap_result_profile() {
    return bitmap_result_profile_for<bitmap_init_result>();
}

const char * input_text_profile() {
    const bool common_offsets =
        offsetof(mtmd_input_text, text) == 0 &&
        alignof(mtmd_input_text) == alignof(void *);
    if (!common_offsets) {
        return "unsupported";
    }
    if (!has_text_len<mtmd_input_text>::value &&
        sizeof(mtmd_input_text) == 16 &&
        offsetof(mtmd_input_text, add_special) == 8 &&
        offsetof(mtmd_input_text, parse_special) == 9) {
        return "mtmd-input-text-v1";
    }
    if (has_text_len<mtmd_input_text>::value &&
        sizeof(mtmd_input_text) == 24 &&
        optional_text_len_offset<mtmd_input_text>() == 8 &&
        offsetof(mtmd_input_text, add_special) == 16 &&
        offsetof(mtmd_input_text, parse_special) == 17) {
        return "mtmd-input-text-v2";
    }
    return "unsupported";
}

bool abi_supported() {
    return std::strcmp(abi_profile(), "unsupported") != 0 &&
        std::strcmp(input_text_profile(), "unsupported") != 0 &&
        std::strcmp(bitmap_result_profile(), "unsupported") != 0 &&
        sizeof(mtmd_caps) == 2 &&
        alignof(mtmd_caps) == 1 &&
        offsetof(mtmd_caps, inp_vision) == 0 &&
        offsetof(mtmd_caps, inp_audio) == 1;
}

void free_video_auxiliary(void * ptr) {
    if (!ptr) {
        return;
    }
    using free_fn = void (*)(void *);
    auto fn = reinterpret_cast<free_fn>(dlsym(RTLD_DEFAULT, "mtmd_helper_video_free"));
    if (fn) {
        fn(ptr);
    }
}

template <typename Fn>
auto guarded(Fn && fn, decltype(fn()) failure) -> decltype(fn()) {
    try {
        last_error.clear();
        return fn();
    } catch (const std::exception & exc) {
        last_error = exc.what();
    } catch (...) {
        last_error = "unknown native exception";
    }
    return failure;
}

} // namespace

extern "C" {

ORBIT_EXPORT uint32_t orbit_mtmd_bridge_api_version() {
    return 1;
}

ORBIT_EXPORT bool orbit_mtmd_bridge_abi_supported() {
    return abi_supported();
}

ORBIT_EXPORT const char * orbit_mtmd_bridge_last_error() {
    return last_error.c_str();
}

ORBIT_EXPORT const char * orbit_mtmd_bridge_manifest_json() {
    static char manifest[8192];
    std::snprintf(
        manifest,
        sizeof(manifest),
        "{\"bridge_api\":1,\"abi_profile\":\"%s\","
        "\"llama_batch\":{\"size\":%zu,\"align\":%zu,\"n_tokens\":%zu,"
        "\"token\":%zu,\"embd\":%zu,\"pos\":%zu,\"n_seq_id\":%zu,"
        "\"seq_id\":%zu,\"logits\":%zu},"
        "\"llama_model_params\":{\"size\":%zu,\"align\":%zu,"
        "\"n_gpu_layers\":%zu,\"progress_callback\":%zu,"
        "\"progress_callback_user_data\":%zu,\"kv_overrides\":%zu,"
        "\"vocab_only\":%zu,\"no_alloc\":%zu},"
        "\"llama_context_params\":{\"size\":%zu,\"align\":%zu,"
        "\"n_ctx\":%zu,\"n_outputs_max\":%zu,\"n_threads\":%zu,"
        "\"flash_attn_type\":%zu,\"defrag_thold\":%zu,\"cb_eval\":%zu,"
        "\"type_k\":%zu,\"abort_callback\":%zu,\"embeddings\":%zu,"
        "\"samplers\":%zu,\"ctx_other\":%zu},"
        "\"llama_sampler_chain_params\":{\"size\":%zu,\"align\":%zu,"
        "\"no_perf\":%zu},"
        "\"llama_chat_message\":{\"size\":%zu,\"align\":%zu,"
        "\"role\":%zu,\"content\":%zu},"
        "\"mtmd_context_params\":{\"size\":%zu,\"align\":%zu,"
        "\"use_gpu\":%zu,\"n_threads\":%zu,\"media_marker\":%zu,"
        "\"cb_eval\":%zu,\"cb_eval_user_data\":%zu,"
        "\"batch_max_tokens\":%zu,\"progress_callback\":%zu},"
        "\"mtmd_input_text\":{\"profile\":\"%s\",\"size\":%zu,\"align\":%zu,"
        "\"text\":%zu,\"text_len\":%zu,\"add_special\":%zu,"
        "\"parse_special\":%zu},\"mtmd_caps\":{\"size\":%zu,\"align\":%zu,"
        "\"inp_vision\":%zu,\"inp_audio\":%zu},\"bitmap_result\":\"%s\","
        "\"upstream_commit\":\"%s\",\"upstream_tag\":\"%s\","
        "\"source_tree_hash\":\"%s\",\"patchset_hash\":\"%s\","
        "\"compiler\":\"%s\",\"build_flags\":\"%s\"}",
        abi_profile(),
        sizeof(llama_batch),
        alignof(llama_batch),
        offsetof(llama_batch, n_tokens),
        offsetof(llama_batch, token),
        offsetof(llama_batch, embd),
        offsetof(llama_batch, pos),
        offsetof(llama_batch, n_seq_id),
        offsetof(llama_batch, seq_id),
        offsetof(llama_batch, logits),
        sizeof(llama_model_params),
        alignof(llama_model_params),
        offsetof(llama_model_params, n_gpu_layers),
        offsetof(llama_model_params, progress_callback),
        offsetof(llama_model_params, progress_callback_user_data),
        offsetof(llama_model_params, kv_overrides),
        offsetof(llama_model_params, vocab_only),
        offsetof(llama_model_params, no_alloc),
        sizeof(llama_context_params),
        alignof(llama_context_params),
        offsetof(llama_context_params, n_ctx),
        offsetof(llama_context_params, n_outputs_max),
        offsetof(llama_context_params, n_threads),
        offsetof(llama_context_params, flash_attn_type),
        offsetof(llama_context_params, defrag_thold),
        offsetof(llama_context_params, cb_eval),
        offsetof(llama_context_params, type_k),
        offsetof(llama_context_params, abort_callback),
        offsetof(llama_context_params, embeddings),
        offsetof(llama_context_params, samplers),
        offsetof(llama_context_params, ctx_other),
        sizeof(llama_sampler_chain_params),
        alignof(llama_sampler_chain_params),
        offsetof(llama_sampler_chain_params, no_perf),
        sizeof(llama_chat_message),
        alignof(llama_chat_message),
        offsetof(llama_chat_message, role),
        offsetof(llama_chat_message, content),
        sizeof(mtmd_context_params),
        alignof(mtmd_context_params),
        offsetof(mtmd_context_params, use_gpu),
        offsetof(mtmd_context_params, n_threads),
        offsetof(mtmd_context_params, media_marker),
        offsetof(mtmd_context_params, cb_eval),
        offsetof(mtmd_context_params, cb_eval_user_data),
        optional_batch_max_tokens_offset<mtmd_context_params>(),
        optional_progress_callback_offset<mtmd_context_params>(),
        input_text_profile(),
        sizeof(mtmd_input_text),
        alignof(mtmd_input_text),
        offsetof(mtmd_input_text, text),
        optional_text_len_offset<mtmd_input_text>(),
        offsetof(mtmd_input_text, add_special),
        offsetof(mtmd_input_text, parse_special),
        sizeof(mtmd_caps),
        alignof(mtmd_caps),
        offsetof(mtmd_caps, inp_vision),
        offsetof(mtmd_caps, inp_audio),
        bitmap_result_profile(),
        ORBIT_LLAMA_UPSTREAM_COMMIT,
        ORBIT_LLAMA_UPSTREAM_TAG,
        ORBIT_LLAMA_SOURCE_TREE_HASH,
        ORBIT_LLAMA_PATCHSET_HASH,
        __VERSION__,
        ORBIT_LLAMA_BUILD_FLAGS);
    return manifest;
}

ORBIT_EXPORT const char * orbit_mtmd_default_marker() {
    return mtmd_default_marker();
}

ORBIT_EXPORT orbit_mtmd_context * orbit_mtmd_context_create(
        const char * mmproj_path,
        llama_model * model,
        bool use_gpu,
        bool print_timings,
        int32_t n_threads,
        const char * media_marker) {
    return guarded([&]() -> orbit_mtmd_context * {
        if (!abi_supported()) {
            last_error = "unsupported mtmd ABI profile";
            return nullptr;
        }
        if (!mmproj_path || !model || !media_marker || n_threads <= 0) {
            last_error = "invalid mtmd bridge arguments";
            return nullptr;
        }
        auto handle = std::make_unique<orbit_mtmd_context>();
        handle->media_marker = media_marker;
        auto params = mtmd_context_params_default();
        params.use_gpu = use_gpu;
        params.print_timings = print_timings;
        params.n_threads = n_threads;
        params.image_marker = nullptr;
        params.media_marker = handle->media_marker.c_str();
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        params.warmup = true;
        params.image_min_tokens = -1;
        params.image_max_tokens = -1;
        params.cb_eval = nullptr;
        params.cb_eval_user_data = nullptr;
        initialize_optional_context_fields(params);
        handle->value = mtmd_init_from_file(mmproj_path, model, params);
        if (!handle->value) {
            last_error = "mtmd_init_from_file failed";
            return nullptr;
        }
        return handle.release();
    }, static_cast<orbit_mtmd_context *>(nullptr));
}

ORBIT_EXPORT void orbit_mtmd_context_free(orbit_mtmd_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->value) {
        mtmd_free(ctx->value);
    }
    delete ctx;
}

ORBIT_EXPORT bool orbit_mtmd_support_vision(const orbit_mtmd_context * ctx) {
    return ctx && ctx->value && mtmd_support_vision(ctx->value);
}

ORBIT_EXPORT bool orbit_mtmd_support_audio(const orbit_mtmd_context * ctx) {
    return ctx && ctx->value && mtmd_support_audio(ctx->value);
}

ORBIT_EXPORT bool orbit_mtmd_get_cap_from_file(const char * path, bool * vision, bool * audio) {
    if (!path || !vision || !audio) {
        last_error = "invalid capability output";
        return false;
    }
    auto caps = mtmd_get_cap_from_file(path);
    *vision = caps.inp_vision;
    *audio = caps.inp_audio;
    return true;
}

ORBIT_EXPORT orbit_mtmd_bitmap * orbit_mtmd_bitmap_init_from_buf(
        orbit_mtmd_context * ctx,
        const unsigned char * data,
        size_t size,
        bool placeholder) {
    return guarded([&]() -> orbit_mtmd_bitmap * {
        if (!ctx || !ctx->value || !data || size == 0) {
            last_error = "invalid bitmap input";
            return nullptr;
        }
        return wrap_bitmap_result(mtmd_helper_bitmap_init_from_buf(ctx->value, data, size, placeholder));
    }, static_cast<orbit_mtmd_bitmap *>(nullptr));
}

ORBIT_EXPORT void orbit_mtmd_bitmap_free(orbit_mtmd_bitmap * bitmap) {
    if (!bitmap) {
        return;
    }
    if (bitmap->value) {
        mtmd_bitmap_free(bitmap->value);
    }
    free_video_auxiliary(bitmap->auxiliary);
    delete bitmap;
}

ORBIT_EXPORT void * orbit_mtmd_chunks_create() {
    return mtmd_input_chunks_init();
}

ORBIT_EXPORT void orbit_mtmd_chunks_free(void * chunks) {
    if (chunks) {
        mtmd_input_chunks_free(static_cast<mtmd_input_chunks *>(chunks));
    }
}

ORBIT_EXPORT size_t orbit_mtmd_chunks_size(const void * chunks) {
    return chunks ? mtmd_input_chunks_size(static_cast<const mtmd_input_chunks *>(chunks)) : 0;
}

ORBIT_EXPORT const void * orbit_mtmd_chunks_get(void * chunks, size_t index) {
    return chunks ? mtmd_input_chunks_get(static_cast<mtmd_input_chunks *>(chunks), index) : nullptr;
}

ORBIT_EXPORT size_t orbit_mtmd_chunk_token_count(const void * chunk) {
    return chunk ? mtmd_input_chunk_get_n_tokens(static_cast<const mtmd_input_chunk *>(chunk)) : 0;
}

ORBIT_EXPORT size_t orbit_mtmd_chunks_token_count(const void * chunks) {
    return chunks ? mtmd_helper_get_n_tokens(static_cast<const mtmd_input_chunks *>(chunks)) : 0;
}

ORBIT_EXPORT int32_t orbit_mtmd_tokenize(
        orbit_mtmd_context * ctx,
        void * chunks,
        const char * text_data,
        size_t text_size,
        bool add_special,
        bool parse_special,
        orbit_mtmd_bitmap * const * bitmaps,
        size_t bitmap_count) {
    return guarded([&]() -> int32_t {
        if (!ctx || !ctx->value || !chunks || !text_data) {
            last_error = "invalid tokenize input";
            return -1;
        }
        mtmd_input_text text{};
        text.text = text_data;
        initialize_text_length(text, text_size);
        text.add_special = add_special;
        text.parse_special = parse_special;
        std::vector<const mtmd_bitmap *> native_bitmaps;
        native_bitmaps.reserve(bitmap_count);
        for (size_t index = 0; index < bitmap_count; ++index) {
            if (!bitmaps[index] || !bitmaps[index]->value) {
                last_error = "invalid bitmap handle";
                return -1;
            }
            native_bitmaps.push_back(bitmaps[index]->value);
        }
        return mtmd_tokenize(
            ctx->value,
            static_cast<mtmd_input_chunks *>(chunks),
            &text,
            native_bitmaps.data(),
            native_bitmaps.size());
    }, int32_t{-1});
}

ORBIT_EXPORT int32_t orbit_mtmd_eval_chunk(
        orbit_mtmd_context * ctx,
        llama_context * llama_ctx,
        const void * chunk,
        int32_t n_past,
        int32_t seq_id,
        int32_t n_batch,
        bool logits_last,
        int32_t * new_n_past) {
    return guarded([&]() -> int32_t {
        if (!ctx || !ctx->value || !llama_ctx || !chunk || !new_n_past || n_batch <= 0) {
            last_error = "invalid eval input";
            return -1;
        }
        return mtmd_helper_eval_chunk_single(
            ctx->value,
            llama_ctx,
            static_cast<const mtmd_input_chunk *>(chunk),
            n_past,
            seq_id,
            n_batch,
            logits_last,
            new_n_past);
    }, int32_t{-1});
}

} // extern "C"
