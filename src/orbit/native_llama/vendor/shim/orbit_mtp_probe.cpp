#include "llama.h"
#include "common/speculative.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

static long rss_kb() {
    FILE * f = std::fopen("/proc/self/status", "r");
    if (!f) {
        return -1;
    }
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "VmRSS: %ld kB", &kb) == 1) {
            break;
        }
    }
    std::fclose(f);
    return kb;
}

static double elapsed_s(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static void print_json(
    bool ok,
    const char * error,
    long rss_before,
    long rss_after,
    long rss_peak,
    double target_load_s,
    double draft_load_s,
    double target_ctx_s,
    double draft_ctx_s,
    double speculative_init_s
) {
    std::printf("{\"ok\":%s,", ok ? "true" : "false");
    if (error) {
        std::printf("\"error\":\"");
        for (const char * p = error; *p; ++p) {
            if (*p == '"' || *p == '\\') {
                std::printf("\\");
            }
            std::printf("%c", *p);
        }
        std::printf("\",");
    } else {
        std::printf("\"error\":null,");
    }
    std::printf(
        "\"rss_before_kb\":%ld,\"rss_after_kb\":%ld,\"rss_peak_kb\":%ld,"
        "\"target_load_s\":%.6f,\"draft_load_s\":%.6f,"
        "\"target_ctx_s\":%.6f,\"draft_ctx_s\":%.6f,\"speculative_init_s\":%.6f}\n",
        rss_before,
        rss_after,
        rss_peak,
        target_load_s,
        draft_load_s,
        target_ctx_s,
        draft_ctx_s,
        speculative_init_s
    );
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s TARGET.gguf DRAFT.gguf\n", argv[0]);
        return 2;
    }

    llama_model * model_tgt = nullptr;
    llama_model * model_dft = nullptr;
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    common_speculative * spec = nullptr;
    const char * error = nullptr;
    llama_context_params mtp_params;
    common_params_speculative spec_params;

    long rss_before = rss_kb();
    long rss_peak = rss_before;
    double target_load_s = 0.0;
    double draft_load_s = 0.0;
    double target_ctx_s = 0.0;
    double draft_ctx_s = 0.0;
    double speculative_init_s = 0.0;

    llama_backend_init();

    auto model_params = llama_model_default_params();
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 128;
    ctx_params.n_threads = 6;
    ctx_params.n_threads_batch = 6;
    ctx_params.n_outputs_max = 1;

    auto t0 = std::chrono::steady_clock::now();
    model_tgt = llama_model_load_from_file(argv[1], model_params);
    target_load_s = elapsed_s(t0);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!model_tgt) {
        error = "failed to load target model";
        goto cleanup;
    }

    t0 = std::chrono::steady_clock::now();
    model_dft = llama_model_load_from_file(argv[2], model_params);
    draft_load_s = elapsed_s(t0);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!model_dft) {
        error = "failed to load draft model";
        goto cleanup;
    }

    t0 = std::chrono::steady_clock::now();
    ctx_tgt = llama_init_from_model(model_tgt, ctx_params);
    target_ctx_s = elapsed_s(t0);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!ctx_tgt) {
        error = "failed to create target context";
        goto cleanup;
    }

    mtp_params = ctx_params;
    mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    mtp_params.n_rs_seq = 0;
    mtp_params.ctx_other = ctx_tgt;

    t0 = std::chrono::steady_clock::now();
    ctx_dft = llama_init_from_model(model_dft, mtp_params);
    draft_ctx_s = elapsed_s(t0);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!ctx_dft) {
        error = "failed to create MTP draft context";
        goto cleanup;
    }

    spec_params.types = common_speculative_types_from_names({"draft-mtp"});
    spec_params.draft.ctx_tgt = ctx_tgt;
    spec_params.draft.ctx_dft = ctx_dft;

    t0 = std::chrono::steady_clock::now();
    spec = common_speculative_init(spec_params, 1);
    speculative_init_s = elapsed_s(t0);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!spec) {
        error = "failed to initialize speculative MTP state";
        goto cleanup;
    }

cleanup:
    if (spec) {
        common_speculative_free(spec);
    }
    if (ctx_dft) {
        llama_free(ctx_dft);
    }
    if (ctx_tgt) {
        llama_free(ctx_tgt);
    }
    if (model_dft) {
        llama_model_free(model_dft);
    }
    if (model_tgt) {
        llama_model_free(model_tgt);
    }
    llama_backend_free();

    print_json(
        error == nullptr,
        error,
        rss_before,
        rss_kb(),
        rss_peak,
        target_load_s,
        draft_load_s,
        target_ctx_s,
        draft_ctx_s,
        speculative_init_s
    );
    return error == nullptr ? 0 : 1;
}
