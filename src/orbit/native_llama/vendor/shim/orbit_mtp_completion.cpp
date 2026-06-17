#include "llama.h"
#include "common/speculative.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double elapsed_s(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char buf[512];
    const int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n <= 0) {
        return {};
    }
    return std::string(buf, buf + n);
}

static void print_json(
    bool ok,
    const char * error,
    const std::string & content,
    int output_tokens,
    int draft_tokens_total,
    int accepted_tokens_total,
    int rejected_tokens_total,
    double acceptance_ratio,
    int target_decode_calls,
    int draft_decode_calls,
    double elapsed_ms,
    double tokens_per_second
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
    std::printf("\"content\":\"");
    for (const char * p = content.c_str(); *p; ++p) {
        if (*p == '"' || *p == '\\') {
            std::printf("\\");
        }
        if (*p == '\n') {
            std::printf("\\n");
            continue;
        }
        std::printf("%c", *p);
    }
    std::printf("\",");
    std::printf(
        "\"output_tokens\":%d,\"draft_tokens_total\":%d,\"accepted_tokens_total\":%d,"
        "\"rejected_tokens_total\":%d,\"acceptance_ratio\":%.6f,"
        "\"target_decode_calls\":%d,\"draft_decode_calls\":%d,"
        "\"elapsed_ms\":%.6f,\"tokens_per_second\":%.6f}\n",
        output_tokens,
        draft_tokens_total,
        accepted_tokens_total,
        rejected_tokens_total,
        acceptance_ratio,
        target_decode_calls,
        draft_decode_calls,
        elapsed_ms,
        tokens_per_second
    );
}

static bool tokenize_prompt(llama_model * model, const char * text, std::vector<llama_token> & out) {
    auto * vocab = llama_model_get_vocab(model);
    const int32_t n_tok = -llama_tokenize(vocab, text, (int32_t) std::strlen(text), nullptr, 0, true, true);
    if (n_tok <= 0) {
        return false;
    }
    out.resize((size_t) n_tok);
    return llama_tokenize(vocab, text, (int32_t) std::strlen(text), out.data(), n_tok, true, true) >= 0;
}

static void fill_batch(llama_batch & batch, const std::vector<llama_token> & tokens, int32_t pos0) {
    batch.n_tokens = (int32_t) tokens.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[(size_t) i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s TARGET.gguf DRAFT.gguf PROMPT MAXTOKENS\n", argv[0]);
        return 2;
    }

    llama_model * model_tgt = nullptr;
    llama_model * model_dft = nullptr;
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    common_speculative * spec = nullptr;
    llama_sampler * smpl = nullptr;
    const char * error = nullptr;
    std::string content;
    common_params_speculative spec_params;

    int output_tokens = 0;
    int draft_tokens_total = 0;
    int accepted_tokens_total = 0;
    int rejected_tokens_total = 0;
    int target_decode_calls = 0;
    int draft_decode_calls = 0;
    double acceptance_ratio = 0.0;

    std::vector<llama_token> prompt;
    std::vector<llama_token> prompt_tgt;
    std::vector<llama_token> generated;
    llama_token id_last = 0;

    const int max_tokens = std::max(1, std::min(32, std::atoi(argv[4])));
    const auto t0 = std::chrono::steady_clock::now();

    llama_backend_init();

    auto model_params = llama_model_default_params();
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 128;
    ctx_params.n_threads = 6;
    ctx_params.n_threads_batch = 6;
    ctx_params.n_outputs_max = 256;

    model_tgt = llama_model_load_from_file(argv[1], model_params);
    if (!model_tgt) {
        error = "failed to load target model";
        goto cleanup;
    }
    model_dft = llama_model_load_from_file(argv[2], model_params);
    if (!model_dft) {
        error = "failed to load draft model";
        goto cleanup;
    }
    ctx_tgt = llama_init_from_model(model_tgt, ctx_params);
    if (!ctx_tgt) {
        error = "failed to create target context";
        goto cleanup;
    }

    {
        auto mtp_params = ctx_params;
        mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mtp_params.n_rs_seq = 0;
        mtp_params.ctx_other = ctx_tgt;
        ctx_dft = llama_init_from_model(model_dft, mtp_params);
    }
    if (!ctx_dft) {
        error = "failed to create MTP draft context";
        goto cleanup;
    }

    spec_params.types = common_speculative_types_from_names({"draft-mtp"});
    spec_params.draft.ctx_tgt = ctx_tgt;
    spec_params.draft.ctx_dft = ctx_dft;
    spec = common_speculative_init(spec_params, 1);
    if (!spec) {
        error = "failed to initialize speculative MTP state";
        goto cleanup;
    }

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    if (!tokenize_prompt(model_tgt, argv[3], prompt)) {
        error = "failed to tokenize prompt";
        goto cleanup;
    }
    if (prompt.size() < 2) {
        error = "prompt too short for mtp completion";
        goto cleanup;
    }
    id_last = prompt.back();
    prompt_tgt.assign(prompt.begin(), prompt.end() - 1);
    generated.reserve((size_t) max_tokens);

    while ((int) generated.size() < max_tokens) {
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        llama_memory_clear(llama_get_memory(ctx_dft), true);

        if (!prompt_tgt.empty()) {
            llama_batch prefill = llama_batch_init((int32_t) prompt_tgt.size(), 0, 1);
            fill_batch(prefill, prompt_tgt, 0);
            if (llama_decode(ctx_tgt, prefill) != 0) {
                error = "failed to decode target prefill";
                llama_batch_free(prefill);
                goto cleanup;
            }
            target_decode_calls++;
            if (!common_speculative_process(spec, prefill)) {
                error = "failed to process speculative prefill";
                llama_batch_free(prefill);
                goto cleanup;
            }
            common_speculative_begin(spec, 0, prompt_tgt);
            llama_batch_free(prefill);
        } else {
            common_speculative_begin(spec, 0, prompt_tgt);
        }

        std::vector<llama_token> draft;
        common_speculative_get_draft_params(spec, 0) = {
            /* .drafting = */ true,
            /* .n_max    = */ std::min(3, max_tokens - (int) generated.size()),
            /* .n_past   = */ (llama_pos) prompt_tgt.size(),
            /* .id_last  = */ id_last,
            /* .prompt   = */ &prompt_tgt,
            /* .result   = */ &draft,
        };
        common_speculative_draft(spec);
        draft_decode_calls++;
        draft_tokens_total += (int) draft.size();

        std::vector<llama_token> validate_tokens;
        validate_tokens.reserve(draft.size() + 1);
        validate_tokens.push_back(id_last);
        validate_tokens.insert(validate_tokens.end(), draft.begin(), draft.end());

        llama_batch validate = llama_batch_init((int32_t) validate_tokens.size(), 0, 1);
        fill_batch(validate, validate_tokens, (int32_t) prompt_tgt.size());
        if (llama_decode(ctx_tgt, validate) != 0) {
            error = "failed to validate speculative batch on target";
            llama_batch_free(validate);
            goto cleanup;
        }
        target_decode_calls++;

        std::vector<llama_token> ids;
        ids.reserve(draft.size() + 1);
        for (int i = 0; i < (int) draft.size(); ++i) {
            const llama_token predicted = llama_sampler_sample(smpl, ctx_tgt, i);
            llama_sampler_accept(smpl, predicted);
            ids.push_back(predicted);
            if (predicted != draft[(size_t) i]) {
                break;
            }
        }
        if (ids.size() == draft.size()) {
            const llama_token predicted = llama_sampler_sample(smpl, ctx_tgt, (int) draft.size());
            llama_sampler_accept(smpl, predicted);
            ids.push_back(predicted);
        }
        llama_batch_free(validate);

        if (ids.empty()) {
            error = "speculative acceptance produced no ids";
            goto cleanup;
        }

        const int accepted = std::max(0, (int) ids.size() - 1);
        common_speculative_accept(spec, 0, (uint16_t) accepted);
        accepted_tokens_total += accepted;
        rejected_tokens_total += (int) draft.size() - accepted;

        for (size_t i = 0; i < ids.size() && (int) generated.size() < max_tokens; ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];

            if (llama_vocab_is_eog(llama_model_get_vocab(model_tgt), id_last)) {
                goto done;
            }

            generated.push_back(id_last);
            content += token_piece(llama_model_get_vocab(model_tgt), id_last);
            output_tokens++;
        }
    }

done:
    acceptance_ratio = draft_tokens_total > 0 ? (double) accepted_tokens_total / (double) draft_tokens_total : 0.0;

cleanup:
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

    const double elapsed_ms = elapsed_s(t0) * 1000.0;
    const double tps = elapsed_ms > 0.0 ? ((double) output_tokens / elapsed_ms) * 1000.0 : 0.0;
    print_json(
        error == nullptr,
        error,
        content,
        output_tokens,
        draft_tokens_total,
        accepted_tokens_total,
        rejected_tokens_total,
        acceptance_ratio,
        target_decode_calls,
        draft_decode_calls,
        elapsed_ms,
        tps
    );
    return error == nullptr ? 0 : 1;
}
