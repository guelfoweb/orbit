#include "llama.h"
#include "common/speculative.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

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
    int draft_tokens,
    long rss_before,
    long rss_after,
    long rss_peak,
    double target_load_s,
    double draft_load_s,
    double target_ctx_s,
    double draft_ctx_s,
    double speculative_init_s,
    double prompt_decode_s,
    double draft_s
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
        "\"draft_tokens\":%d,"
        "\"rss_before_kb\":%ld,\"rss_after_kb\":%ld,\"rss_peak_kb\":%ld,"
        "\"target_load_s\":%.6f,\"draft_load_s\":%.6f,"
        "\"target_ctx_s\":%.6f,\"draft_ctx_s\":%.6f,"
        "\"speculative_init_s\":%.6f,\"prompt_decode_s\":%.6f,\"draft_s\":%.6f}\n",
        draft_tokens,
        rss_before,
        rss_after,
        rss_peak,
        target_load_s,
        draft_load_s,
        target_ctx_s,
        draft_ctx_s,
        speculative_init_s,
        prompt_decode_s,
        draft_s
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
    llama_sampler * smpl = nullptr;
    llama_batch prompt_batch = {};
    const char * error = nullptr;
    llama_context_params mtp_params;
    common_params_speculative spec_params;
    std::vector<llama_token> prompt;
    std::vector<llama_token> draft;
    int draft_tokens = 0;
    llama_token sampled = 0;

    long rss_before = rss_kb();
    long rss_peak = rss_before;
    double target_load_s = 0.0;
    double draft_load_s = 0.0;
    double target_ctx_s = 0.0;
    double draft_ctx_s = 0.0;
    double speculative_init_s = 0.0;
    double prompt_decode_s = 0.0;
    double draft_s = 0.0;

    llama_backend_init();

    auto model_params = llama_model_default_params();
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 128;
    ctx_params.n_threads = 6;
    ctx_params.n_threads_batch = 6;
    ctx_params.n_outputs_max = 16;

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

    {
        const char * text = "Hello";
        auto * vocab = llama_model_get_vocab(model_tgt);
        const int32_t n_tok = -llama_tokenize(vocab, text, (int32_t) std::strlen(text), nullptr, 0, true, true);
        if (n_tok <= 0) {
            error = "failed to size prompt tokenization";
            goto cleanup;
        }
        prompt.resize((size_t) n_tok);
        const int32_t rc = llama_tokenize(vocab, text, (int32_t) std::strlen(text), prompt.data(), n_tok, true, true);
        if (rc < 0) {
            error = "failed to tokenize prompt";
            goto cleanup;
        }
    }

    {
        prompt_batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
        prompt_batch.n_tokens = (int32_t) prompt.size();
        for (int32_t i = 0; i < prompt_batch.n_tokens; ++i) {
            prompt_batch.token[i] = prompt[(size_t) i];
            prompt_batch.pos[i] = i;
            prompt_batch.n_seq_id[i] = 1;
            prompt_batch.seq_id[i][0] = 0;
            prompt_batch.logits[i] = 1;
        }
        t0 = std::chrono::steady_clock::now();
        const int rc = llama_decode(ctx_tgt, prompt_batch);
        prompt_decode_s = elapsed_s(t0);
        rss_peak = std::max(rss_peak, rss_kb());
        if (rc != 0) {
            error = "failed to decode target prompt";
            goto cleanup;
        }
        if (!common_speculative_process(spec, prompt_batch)) {
            error = "failed to process speculative prompt batch";
            goto cleanup;
        }
    }

    common_speculative_begin(spec, 0, prompt);

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    sampled = llama_sampler_sample(smpl, ctx_tgt, -1);
    llama_sampler_accept(smpl, sampled);

    common_speculative_get_draft_params(spec, 0) = {
        /* .drafting = */ true,
        /* .n_max    = */ 3,
        /* .n_past   = */ (llama_pos) prompt.size(),
        /* .id_last  = */ sampled,
        /* .prompt   = */ &prompt,
        /* .result   = */ &draft,
    };

    t0 = std::chrono::steady_clock::now();
    common_speculative_draft(spec);
    draft_s = elapsed_s(t0);
    rss_peak = std::max(rss_peak, rss_kb());
    draft_tokens = (int) draft.size();

cleanup:
    if (smpl) {
        llama_sampler_free(smpl);
    }
    if (prompt_batch.token || prompt_batch.embd || prompt_batch.pos || prompt_batch.n_seq_id || prompt_batch.seq_id || prompt_batch.logits) {
        llama_batch_free(prompt_batch);
    }
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
        draft_tokens,
        rss_before,
        rss_kb(),
        rss_peak,
        target_load_s,
        draft_load_s,
        target_ctx_s,
        draft_ctx_s,
        speculative_init_s,
        prompt_decode_s,
        draft_s
    );
    return error == nullptr ? 0 : 1;
}
