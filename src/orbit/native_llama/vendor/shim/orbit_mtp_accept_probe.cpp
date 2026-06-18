#include "llama.h"
#include "common/speculative.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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
    int accepted_tokens,
    int rejected_tokens,
    double acceptance_ratio,
    int target_decode_calls,
    int draft_decode_calls,
    double elapsed_ms,
    long rss_before,
    long rss_after,
    long rss_peak
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
        "\"draft_tokens\":%d,\"accepted_tokens\":%d,\"rejected_tokens\":%d,"
        "\"acceptance_ratio\":%.6f,\"target_decode_calls\":%d,\"draft_decode_calls\":%d,"
        "\"elapsed_ms\":%.6f,\"rss_before_kb\":%ld,\"rss_after_kb\":%ld,\"rss_peak_kb\":%ld}\n",
        draft_tokens,
        accepted_tokens,
        rejected_tokens,
        acceptance_ratio,
        target_decode_calls,
        draft_decode_calls,
        elapsed_ms,
        rss_before,
        rss_after,
        rss_peak
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
    llama_batch validate_batch = {};
    const char * error = nullptr;
    llama_context_params mtp_params;
    common_params_speculative spec_params;
    std::vector<llama_token> prompt;
    std::vector<llama_token> draft;
    llama_token sampled = 0;

    int draft_tokens = 0;
    int accepted_tokens = 0;
    int rejected_tokens = 0;
    int target_decode_calls = 0;
    int draft_decode_calls = 0;
    double acceptance_ratio = 0.0;

    const auto total_start = std::chrono::steady_clock::now();
    long rss_before = rss_kb();
    long rss_peak = rss_before;

    llama_backend_init();

    auto model_params = llama_model_default_params();
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 128;
    ctx_params.n_threads = 6;
    ctx_params.n_threads_batch = 6;
    ctx_params.n_outputs_max = 16;

    model_tgt = llama_model_load_from_file(argv[1], model_params);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!model_tgt) {
        error = "failed to load target model";
        goto cleanup;
    }

    model_dft = llama_model_load_from_file(argv[2], model_params);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!model_dft) {
        error = "failed to load draft model";
        goto cleanup;
    }

    ctx_tgt = llama_init_from_model(model_tgt, ctx_params);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!ctx_tgt) {
        error = "failed to create target context";
        goto cleanup;
    }

    mtp_params = ctx_params;
    mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    mtp_params.n_rs_seq = 0;
    mtp_params.ctx_other = ctx_tgt;

    ctx_dft = llama_init_from_model(model_dft, mtp_params);
    rss_peak = std::max(rss_peak, rss_kb());
    if (!ctx_dft) {
        error = "failed to create MTP draft context";
        goto cleanup;
    }

    spec_params.types = common_speculative_types_from_names({"draft-mtp"});
    spec_params.draft.ctx_tgt = ctx_tgt;
    spec_params.draft.ctx_dft = ctx_dft;
    spec = common_speculative_init(spec_params, 1);
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

    prompt_batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
    prompt_batch.n_tokens = (int32_t) prompt.size();
    for (int32_t i = 0; i < prompt_batch.n_tokens; ++i) {
        prompt_batch.token[i] = prompt[(size_t) i];
        prompt_batch.pos[i] = i;
        prompt_batch.n_seq_id[i] = 1;
        prompt_batch.seq_id[i][0] = 0;
        prompt_batch.logits[i] = 1;
    }

    if (llama_decode(ctx_tgt, prompt_batch) != 0) {
        error = "failed to decode target prompt";
        goto cleanup;
    }
    target_decode_calls++;
    rss_peak = std::max(rss_peak, rss_kb());
    if (!common_speculative_process(spec, prompt_batch)) {
        error = "failed to process speculative prompt batch";
        goto cleanup;
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

    common_speculative_draft(spec);
    draft_tokens = (int) draft.size();
    draft_decode_calls = draft_tokens > 0 ? 1 : 0;
    rss_peak = std::max(rss_peak, rss_kb());
    if (draft.empty()) {
        error = "draft probe produced no tokens";
        goto cleanup;
    }

    validate_batch = llama_batch_init((int32_t) draft.size(), 0, 1);
    validate_batch.n_tokens = (int32_t) draft.size();
    for (int32_t i = 0; i < validate_batch.n_tokens; ++i) {
        validate_batch.token[i] = draft[(size_t) i];
        validate_batch.pos[i] = (llama_pos) prompt.size() + i;
        validate_batch.n_seq_id[i] = 1;
        validate_batch.seq_id[i][0] = 0;
        validate_batch.logits[i] = 1;
    }

    if (llama_decode(ctx_tgt, validate_batch) != 0) {
        error = "failed to validate drafted tokens on target";
        goto cleanup;
    }
    target_decode_calls++;
    rss_peak = std::max(rss_peak, rss_kb());

    {
        llama_sampler * validate_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(validate_sampler, llama_sampler_init_greedy());
        for (int i = 0; i < draft_tokens; ++i) {
            const llama_token predicted = llama_sampler_sample(validate_sampler, ctx_tgt, i);
            llama_sampler_accept(validate_sampler, predicted);
            if (predicted != draft[(size_t) i]) {
                break;
            }
            accepted_tokens++;
        }
        llama_sampler_free(validate_sampler);
    }

    rejected_tokens = draft_tokens - accepted_tokens;
    acceptance_ratio = draft_tokens > 0 ? (double) accepted_tokens / (double) draft_tokens : 0.0;
    common_speculative_accept(spec, 0, (uint16_t) accepted_tokens);

cleanup:
    if (validate_batch.token || validate_batch.embd || validate_batch.pos || validate_batch.n_seq_id || validate_batch.seq_id || validate_batch.logits) {
        llama_batch_free(validate_batch);
    }
    if (prompt_batch.token || prompt_batch.embd || prompt_batch.pos || prompt_batch.n_seq_id || prompt_batch.seq_id || prompt_batch.logits) {
        llama_batch_free(prompt_batch);
    }
    if (smpl) {
        llama_sampler_free(smpl);
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
        accepted_tokens,
        rejected_tokens,
        acceptance_ratio,
        target_decode_calls,
        draft_decode_calls,
        elapsed_s(total_start) * 1000.0,
        rss_before,
        rss_kb(),
        rss_peak
    );
    return error == nullptr ? 0 : 1;
}
