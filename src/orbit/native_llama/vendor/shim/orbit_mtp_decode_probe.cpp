#include "llama.h"
#include "common/speculative.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct prompt_result {
    std::string name;
    int output_tokens = 0;
    int draft_tokens_total = 0;
    int accepted_tokens_total = 0;
    int rejected_tokens_total = 0;
    double acceptance_ratio = 0.0;
    int target_decode_calls = 0;
    int draft_decode_calls = 0;
    double elapsed_ms = 0.0;
    double tokens_per_second = 0.0;
    std::string error;
};

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

static void print_json(bool ok, const char * error, const std::vector<prompt_result> & prompts, long rss_before, long rss_after, long rss_peak) {
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
    std::printf("\"prompts\":[");
    for (size_t i = 0; i < prompts.size(); ++i) {
        const auto & pr = prompts[i];
        if (i > 0) {
            std::printf(",");
        }
        std::printf(
            "{\"name\":\"%s\",\"output_tokens\":%d,\"draft_tokens_total\":%d,"
            "\"accepted_tokens_total\":%d,\"rejected_tokens_total\":%d,"
            "\"acceptance_ratio\":%.6f,\"target_decode_calls\":%d,\"draft_decode_calls\":%d,"
            "\"elapsed_ms\":%.6f,\"tokens_per_second\":%.6f,",
            pr.name.c_str(),
            pr.output_tokens,
            pr.draft_tokens_total,
            pr.accepted_tokens_total,
            pr.rejected_tokens_total,
            pr.acceptance_ratio,
            pr.target_decode_calls,
            pr.draft_decode_calls,
            pr.elapsed_ms,
            pr.tokens_per_second
        );
        if (pr.error.empty()) {
            std::printf("\"error\":null}");
        } else {
            std::printf("\"error\":\"");
            for (const char * p = pr.error.c_str(); *p; ++p) {
                if (*p == '"' || *p == '\\') {
                    std::printf("\\");
                }
                std::printf("%c", *p);
            }
            std::printf("\"}");
        }
    }
    std::printf("],\"rss_before_kb\":%ld,\"rss_after_kb\":%ld,\"rss_peak_kb\":%ld}\n", rss_before, rss_after, rss_peak);
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

static bool run_prompt(
    llama_model * model_tgt,
    llama_context * ctx_tgt,
    llama_context * ctx_dft,
    common_speculative * spec,
    const char * prompt_text,
    const char * prompt_name,
    prompt_result & out,
    long & rss_peak
) {
    out.name = prompt_name;
    const auto t0 = std::chrono::steady_clock::now();
    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);
    auto * vocab = llama_model_get_vocab(model_tgt);

    std::vector<llama_token> prompt;
    if (!tokenize_prompt(model_tgt, prompt_text, prompt)) {
        out.error = "failed to tokenize prompt";
        return false;
    }
    if (prompt.size() < 2) {
        out.error = "prompt too short for mtp probe";
        return false;
    }

    llama_token id_last = prompt.back();
    std::vector<llama_token> prompt_tgt(prompt.begin(), prompt.end() - 1);
    std::vector<llama_token> generated;
    generated.reserve(16);
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    while ((int) generated.size() < 16) {
        llama_memory_clear(mem_tgt, true);
        llama_memory_clear(mem_dft, true);

        if (!prompt_tgt.empty()) {
            llama_batch prefill = llama_batch_init((int32_t) prompt_tgt.size(), 0, 1);
            fill_batch(prefill, prompt_tgt, 0);
            if (llama_decode(ctx_tgt, prefill) != 0) {
                out.error = "failed to decode target prefill";
                llama_batch_free(prefill);
                llama_sampler_free(smpl);
                return false;
            }
            out.target_decode_calls++;
            rss_peak = std::max(rss_peak, rss_kb());

            if (!common_speculative_process(spec, prefill)) {
                out.error = "failed to process speculative prefill";
                llama_batch_free(prefill);
                llama_sampler_free(smpl);
                return false;
            }
            common_speculative_begin(spec, 0, prompt_tgt);
            llama_batch_free(prefill);
        } else {
            common_speculative_begin(spec, 0, prompt_tgt);
        }

        std::vector<llama_token> draft;
        common_speculative_get_draft_params(spec, 0) = {
            /* .drafting = */ true,
            /* .n_max    = */ std::min(3, 16 - (int) generated.size()),
            /* .n_past   = */ (llama_pos) prompt_tgt.size(),
            /* .id_last  = */ id_last,
            /* .prompt   = */ &prompt_tgt,
            /* .result   = */ &draft,
        };
        common_speculative_draft(spec);
        out.draft_decode_calls++;
        rss_peak = std::max(rss_peak, rss_kb());
        out.draft_tokens_total += (int) draft.size();

        std::vector<llama_token> validate_tokens;
        validate_tokens.reserve(draft.size() + 1);
        validate_tokens.push_back(id_last);
        validate_tokens.insert(validate_tokens.end(), draft.begin(), draft.end());

        llama_batch validate = llama_batch_init((int32_t) validate_tokens.size(), 0, 1);
        fill_batch(validate, validate_tokens, (int32_t) prompt_tgt.size());
        if (llama_decode(ctx_tgt, validate) != 0) {
            out.error = "failed to validate speculative batch on target";
            llama_batch_free(validate);
            llama_sampler_free(smpl);
            return false;
        }
        out.target_decode_calls++;
        rss_peak = std::max(rss_peak, rss_kb());

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
        if (ids.empty()) {
            out.error = "speculative acceptance produced no ids";
            llama_batch_free(validate);
            llama_sampler_free(smpl);
            return false;
        }

        const int accepted = std::max(0, (int) ids.size() - 1);

        common_speculative_accept(spec, 0, (uint16_t) accepted);
        out.accepted_tokens_total += accepted;
        out.rejected_tokens_total += (int) draft.size() - accepted;

        for (size_t i = 0; i < ids.size() && (int) generated.size() < 16; ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                break;
            }
            generated.push_back(id_last);
            out.output_tokens++;
        }

        llama_batch_free(validate);
    }

    llama_sampler_free(smpl);
    out.acceptance_ratio = out.draft_tokens_total > 0 ? (double) out.accepted_tokens_total / (double) out.draft_tokens_total : 0.0;
    out.elapsed_ms = elapsed_s(t0) * 1000.0;
    out.tokens_per_second = out.elapsed_ms > 0.0 ? ((double) out.output_tokens / out.elapsed_ms) * 1000.0 : 0.0;
    return true;
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
    std::vector<prompt_result> prompts;
    llama_context_params mtp_params;
    common_params_speculative spec_params;

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
    ctx_params.n_outputs_max = 256;

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

    prompts.resize(2);
    if (!run_prompt(model_tgt, ctx_tgt, ctx_dft, spec, "Say only: ok.", "prompt_short", prompts[0], rss_peak) && error == nullptr) {
        error = prompts[0].error.c_str();
    }
    if (error == nullptr && !run_prompt(model_tgt, ctx_tgt, ctx_dft, spec, "In one short sentence, explain what a CPU does.", "prompt_medium", prompts[1], rss_peak)) {
        error = prompts[1].error.c_str();
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
    print_json(error == nullptr, error, prompts, rss_before, rss_kb(), rss_peak);
    return error == nullptr ? 0 : 1;
}
