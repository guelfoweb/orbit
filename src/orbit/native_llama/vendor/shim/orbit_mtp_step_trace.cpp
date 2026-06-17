#include "llama.h"
#include "common/common.h"
#include "common/speculative.h"
#include "common/sampling.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

static constexpr int32_t ORBIT_MTP_DRAFT_N_MAX = 3;

struct trace_step {
    int index = 0;
    std::string mode;
    std::vector<llama_token> draft;
    std::vector<llama_token> accepted_ids;
    int accepted_draft = 0;
    int validated_count = 0;
    int checkpoint_total = 0;
    int restore_total = 0;
    bool checkpoint_created = false;
    bool restore_executed = false;
    bool validate_processed_by_spec = false;
    std::string resolution;
    std::string sampler_before;
    std::string sampler_after;
    int32_t kv_tgt_before_min = -1;
    int32_t kv_tgt_before_max = -1;
    int32_t kv_dft_before_min = -1;
    int32_t kv_dft_before_max = -1;
    int32_t kv_tgt_after_min = -1;
    int32_t kv_tgt_after_max = -1;
    int32_t kv_dft_after_min = -1;
    int32_t kv_dft_after_max = -1;
};

struct trace_summary {
    std::string mode;
    int draft_tokens_total = 0;
    int accepted_tokens_total = 0;
    int rejected_tokens_total = 0;
    double acceptance_ratio = 0.0;
    int target_decode_calls = 0;
    int draft_decode_calls = 0;
    int checkpoint_count = 0;
    int restore_count = 0;
    int output_tokens = 0;
    std::string error;
};

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

static bool tokenize_prompt(llama_model * model, const char * text, std::vector<llama_token> & out) {
    auto * vocab = llama_model_get_vocab(model);
    const bool add_special = std::strncmp(text, "<bos>", 5) != 0;
    const int32_t n_tok = -llama_tokenize(vocab, text, (int32_t) std::strlen(text), nullptr, 0, add_special, true);
    if (n_tok <= 0) {
        return false;
    }
    out.resize((size_t) n_tok);
    return llama_tokenize(vocab, text, (int32_t) std::strlen(text), out.data(), n_tok, add_special, true) >= 0;
}

static common_params_sampling make_sampling_params() {
    common_params_sampling params;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.min_p = 0.0f;
    params.typ_p = 1.0f;
    params.temp = 0.0f;
    params.penalty_last_n = 0;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 0.0f;
    params.dry_multiplier = 0.0f;
    params.samplers = {
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    return params;
}

static std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    return out;
}

static std::string token_piece(const llama_context * ctx, llama_token tok) {
    return common_token_to_piece(ctx, tok, true);
}

static void print_token_vector_json(const llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::printf("[");
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf(
            "{\"id\":%d,\"piece\":\"%s\"}",
            (int) tokens[i],
            json_escape(token_piece(ctx, tokens[i])).c_str());
    }
    std::printf("]");
}

static void print_optional_token_json(const llama_context * ctx, const std::vector<llama_token> & draft, int accepted_draft) {
    if (accepted_draft < 0 || accepted_draft >= (int) draft.size()) {
        std::printf("null");
        return;
    }
    const auto tok = draft[(size_t) accepted_draft];
    std::printf(
        "{\"id\":%d,\"piece\":\"%s\"}",
        (int) tok,
        json_escape(token_piece(ctx, tok)).c_str());
}

static void print_trace_json(
    bool ok,
    const trace_summary & summary,
    const std::vector<trace_step> & steps,
    const llama_context * ctx_tgt
) {
    std::printf("{\"ok\":%s,", ok ? "true" : "false");
    std::printf("\"mode\":\"%s\",", summary.mode.c_str());
    if (summary.error.empty()) {
        std::printf("\"error\":null,");
    } else {
        std::printf("\"error\":\"%s\",", json_escape(summary.error).c_str());
    }
    std::printf(
        "\"summary\":{\"draft_tokens_total\":%d,\"accepted_tokens_total\":%d,"
        "\"rejected_tokens_total\":%d,\"acceptance_ratio\":%.6f,"
        "\"target_decode_calls\":%d,\"draft_decode_calls\":%d,"
        "\"checkpoint_count\":%d,\"restore_count\":%d,\"output_tokens\":%d},",
        summary.draft_tokens_total,
        summary.accepted_tokens_total,
        summary.rejected_tokens_total,
        summary.acceptance_ratio,
        summary.target_decode_calls,
        summary.draft_decode_calls,
        summary.checkpoint_count,
        summary.restore_count,
        summary.output_tokens
    );
    std::printf("\"steps\":[");
    for (size_t i = 0; i < steps.size(); ++i) {
        const auto & step = steps[i];
        if (i > 0) {
            std::printf(",");
        }
        std::printf(
            "{\"index\":%d,\"mode\":\"%s\",\"accepted_draft\":%d,"
            "\"validated_count\":%d,\"checkpoint_created\":%s,"
            "\"restore_executed\":%s,\"checkpoint_total\":%d,"
            "\"restore_total\":%d,\"validate_processed_by_spec\":%s,"
            "\"resolution\":\"%s\",\"sampler_before\":\"%s\","
            "\"sampler_after\":\"%s\",\"kv_tgt_before\":{\"min\":%d,\"max\":%d},"
            "\"kv_dft_before\":{\"min\":%d,\"max\":%d},\"kv_tgt_after\":{\"min\":%d,\"max\":%d},"
            "\"kv_dft_after\":{\"min\":%d,\"max\":%d},\"draft\":",
            step.index,
            step.mode.c_str(),
            step.accepted_draft,
            step.validated_count,
            step.checkpoint_created ? "true" : "false",
            step.restore_executed ? "true" : "false",
            step.checkpoint_total,
            step.restore_total,
            step.validate_processed_by_spec ? "true" : "false",
            step.resolution.c_str(),
            json_escape(step.sampler_before).c_str(),
            json_escape(step.sampler_after).c_str(),
            step.kv_tgt_before_min,
            step.kv_tgt_before_max,
            step.kv_dft_before_min,
            step.kv_dft_before_max,
            step.kv_tgt_after_min,
            step.kv_tgt_after_max,
            step.kv_dft_after_min,
            step.kv_dft_after_max
        );
        print_token_vector_json(ctx_tgt, step.draft);
        std::printf(",\"accepted_ids\":");
        print_token_vector_json(ctx_tgt, step.accepted_ids);
        std::printf(",\"first_rejected\":");
        print_optional_token_json(ctx_tgt, step.draft, step.accepted_draft);
        std::printf("}");
    }
    std::printf("]}\n");
}

static bool run_orbit_current(
    llama_model * model_tgt,
    llama_context * ctx_tgt,
    llama_context * ctx_dft,
    common_speculative * spec,
    const char * prompt_text,
    int max_tokens,
    trace_summary & summary,
    std::vector<trace_step> & steps
) {
    summary.mode = "orbit-current";
    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);
    if (!mem_tgt || !mem_dft) {
        summary.error = "missing llama memory";
        return false;
    }

    std::vector<llama_token> prompt;
    if (!tokenize_prompt(model_tgt, prompt_text, prompt) || prompt.size() < 2) {
        summary.error = "failed to tokenize prompt";
        return false;
    }

    std::vector<llama_token> prompt_tgt(prompt);
    llama_token id_last = LLAMA_TOKEN_NULL;
    int32_t n_past = (int32_t) prompt_tgt.size();
    const auto sampling_params = make_sampling_params();
    common_sampler * smpl = common_sampler_init(model_tgt, const_cast<common_params_sampling &>(sampling_params));
    if (!smpl) {
        summary.error = "failed to create sampler";
        return false;
    }

    bool need_replay = true;
    std::vector<llama_token> draft;
    common_prompt_checkpoint ckpt;
    bool have_ckpt = false;
    bool stop = false;
    int step_index = 0;

    while (summary.output_tokens < max_tokens && !stop) {
        if (need_replay) {
            llama_memory_clear(mem_tgt, true);
            llama_memory_clear(mem_dft, true);
            if (!prompt_tgt.empty()) {
                llama_batch prefill_tgt = llama_batch_init((int32_t) prompt_tgt.size(), 0, 1);
                fill_batch(prefill_tgt, prompt_tgt, 0);
                if (llama_decode(ctx_tgt, prefill_tgt) != 0) {
                    llama_batch_free(prefill_tgt);
                    summary.error = "target prefill failed";
                    common_sampler_free(smpl);
                    return false;
                }
                summary.target_decode_calls++;
                llama_batch_free(prefill_tgt);

                llama_batch prefill_spec = llama_batch_init((int32_t) prompt_tgt.size(), 0, 1);
                fill_batch(prefill_spec, prompt_tgt, 0);
                if (!common_speculative_process(spec, prefill_spec)) {
                    llama_batch_free(prefill_spec);
                    summary.error = "speculative prefill failed";
                    common_sampler_free(smpl);
                    return false;
                }
                llama_batch_free(prefill_spec);
            }
            common_speculative_begin(spec, 0, prompt_tgt);
            draft.clear();
            ckpt.clear();
            have_ckpt = false;
            n_past = (int32_t) prompt_tgt.size();
            if (summary.output_tokens == 0) {
                id_last = common_sampler_sample(smpl, ctx_tgt, -1);
                common_sampler_accept(smpl, id_last, true);
            }
            need_replay = false;
        }

        bool checkpoint_created = false;
        if (draft.empty()) {
            ckpt.update_pos(
                (int64_t) prompt_tgt.size(),
                llama_memory_seq_pos_min(mem_tgt, 0),
                llama_memory_seq_pos_max(mem_tgt, 0));
            ckpt.update_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

            common_speculative_get_draft_params(spec, 0) = {
                true,
                std::min(ORBIT_MTP_DRAFT_N_MAX, max_tokens - summary.output_tokens),
                (llama_pos) n_past,
                id_last,
                &prompt_tgt,
                &draft,
            };
            common_speculative_draft(spec);
            summary.draft_decode_calls++;
            summary.draft_tokens_total += (int) draft.size();

            if (!draft.empty()) {
                ckpt.update_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                ckpt.load_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
                have_ckpt = true;
                checkpoint_created = true;
                summary.checkpoint_count++;
            } else {
                have_ckpt = false;
            }
        }

        std::vector<llama_token> validate_tokens;
        validate_tokens.push_back(id_last);
        validate_tokens.insert(validate_tokens.end(), draft.begin(), draft.end());

        trace_step step;
        step.index = ++step_index;
        step.mode = summary.mode;
        step.draft = draft;
        step.validated_count = (int) validate_tokens.size();
        step.checkpoint_created = checkpoint_created;
        step.checkpoint_total = summary.checkpoint_count;
        step.restore_total = summary.restore_count;
        step.validate_processed_by_spec = true;
        step.sampler_before = common_sampler_prev_str(smpl, ctx_tgt, 8);
        step.kv_tgt_before_min = llama_memory_seq_pos_min(mem_tgt, 0);
        step.kv_tgt_before_max = llama_memory_seq_pos_max(mem_tgt, 0);
        step.kv_dft_before_min = llama_memory_seq_pos_min(mem_dft, 0);
        step.kv_dft_before_max = llama_memory_seq_pos_max(mem_dft, 0);

        llama_batch validate = llama_batch_init((int32_t) validate_tokens.size(), 0, 1);
        fill_batch(validate, validate_tokens, n_past);
        if (llama_decode(ctx_tgt, validate) != 0) {
            llama_batch_free(validate);
            summary.error = "target validate failed";
            common_sampler_free(smpl);
            return false;
        }
        summary.target_decode_calls++;
        if (!common_speculative_process(spec, validate)) {
            llama_batch_free(validate);
            summary.error = "speculative validate process failed";
            common_sampler_free(smpl);
            return false;
        }
        std::vector<int> rows(validate_tokens.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            rows[i] = (int) i;
        }
        auto * smpl_save = have_ckpt ? common_sampler_clone(smpl) : nullptr;
        auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, rows, draft);
        llama_batch_free(validate);
        if (ids.empty()) {
            if (smpl_save) {
                common_sampler_free(smpl_save);
            }
            summary.error = "acceptance produced no ids";
            common_sampler_free(smpl);
            return false;
        }

        const int accepted = std::max(0, (int) ids.size() - 1);
        summary.accepted_tokens_total += accepted;
        summary.rejected_tokens_total += (int) draft.size() - accepted;
        step.accepted_draft = accepted;
        step.accepted_ids = ids;
        step.sampler_after = common_sampler_prev_str(smpl, ctx_tgt, 8);

        if (accepted == (int) draft.size()) {
            if (smpl_save) {
                common_sampler_free(smpl_save);
            }
            step.resolution = "full_accept";
            common_speculative_accept(spec, 0, (uint16_t) accepted);
            for (llama_token tok : ids) {
                prompt_tgt.push_back(id_last);
                id_last = tok;
                if (llama_vocab_is_eog(llama_model_get_vocab(model_tgt), id_last)) {
                    stop = true;
                    break;
                }
                summary.output_tokens++;
                if (summary.output_tokens >= max_tokens) {
                    break;
                }
            }
            n_past = (int32_t) prompt_tgt.size();
            llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
            llama_memory_seq_rm(mem_dft, 0, n_past, -1);
            draft.clear();
            have_ckpt = false;
            need_replay = false;
        } else if (have_ckpt) {
            step.resolution = "partial_restore";
            draft = ids;
            ckpt.load_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(mem_tgt, 0, ckpt.pos_max + 1, -1);
            ckpt.load_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
            prompt_tgt.resize((size_t) ckpt.n_tokens);
            n_past = (int32_t) prompt_tgt.size();
            if (smpl_save) {
                common_sampler_free(smpl);
                smpl = smpl_save;
                smpl_save = nullptr;
            }
            summary.restore_count++;
            step.restore_executed = true;
            step.restore_total = summary.restore_count;
        } else {
            if (smpl_save) {
                common_sampler_free(smpl_save);
            }
            step.resolution = "replay_fallback";
            need_replay = true;
            draft.clear();
        }

        step.kv_tgt_after_min = llama_memory_seq_pos_min(mem_tgt, 0);
        step.kv_tgt_after_max = llama_memory_seq_pos_max(mem_tgt, 0);
        step.kv_dft_after_min = llama_memory_seq_pos_min(mem_dft, 0);
        step.kv_dft_after_max = llama_memory_seq_pos_max(mem_dft, 0);
        steps.push_back(step);
    }

    common_sampler_free(smpl);
    summary.acceptance_ratio = summary.draft_tokens_total > 0
        ? (double) summary.accepted_tokens_total / (double) summary.draft_tokens_total
        : 0.0;
    return true;
}

static bool run_server_context_like(
    llama_model * model_tgt,
    llama_context * ctx_tgt,
    llama_context * ctx_dft,
    common_speculative * spec,
    const char * prompt_text,
    int max_tokens,
    trace_summary & summary,
    std::vector<trace_step> & steps
) {
    summary.mode = "server-context-like";
    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);
    if (!mem_tgt || !mem_dft) {
        summary.error = "missing llama memory";
        return false;
    }

    std::vector<llama_token> prompt;
    if (!tokenize_prompt(model_tgt, prompt_text, prompt) || prompt.empty()) {
        summary.error = "failed to tokenize prompt";
        return false;
    }

    const auto sampling_params = make_sampling_params();
    common_sampler * smpl = common_sampler_init(model_tgt, const_cast<common_params_sampling &>(sampling_params));
    if (!smpl) {
        summary.error = "failed to create sampler";
        return false;
    }

    llama_memory_clear(mem_tgt, true);
    llama_memory_clear(mem_dft, true);
    llama_batch prompt_batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
    fill_batch(prompt_batch, prompt, 0);
    if (llama_decode(ctx_tgt, prompt_batch) != 0) {
        llama_batch_free(prompt_batch);
        summary.error = "target prompt prefill failed";
        common_sampler_free(smpl);
        return false;
    }
    summary.target_decode_calls++;
    if (!common_speculative_process(spec, prompt_batch)) {
        llama_batch_free(prompt_batch);
        summary.error = "speculative prompt process failed";
        common_sampler_free(smpl);
        return false;
    }
    llama_batch_free(prompt_batch);

    std::vector<llama_token> prompt_tgt = prompt;
    common_speculative_begin(spec, 0, prompt_tgt);
    llama_token id_last = common_sampler_sample(smpl, ctx_tgt, -1);
    common_sampler_accept(smpl, id_last, true);
    int32_t n_past = (int32_t) prompt_tgt.size();

    std::vector<llama_token> draft;
    common_prompt_checkpoint ckpt;
    bool stop = false;
    int step_index = 0;

    while (summary.output_tokens < max_tokens && !stop) {
        bool checkpoint_created = false;
        if (draft.empty()) {
            ckpt.update_pos(
                (int64_t) prompt_tgt.size(),
                llama_memory_seq_pos_min(mem_tgt, 0),
                llama_memory_seq_pos_max(mem_tgt, 0));
            ckpt.update_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            common_speculative_get_draft_params(spec, 0) = {
                true,
                std::min(ORBIT_MTP_DRAFT_N_MAX, max_tokens - summary.output_tokens),
                (llama_pos) n_past,
                id_last,
                &prompt_tgt,
                &draft,
            };
            common_speculative_draft(spec);
            summary.draft_decode_calls++;
            summary.draft_tokens_total += (int) draft.size();

            if (!draft.empty()) {
                ckpt.update_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                ckpt.load_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
                summary.checkpoint_count++;
                checkpoint_created = true;
            }
        }

        std::vector<llama_token> validate_tokens;
        validate_tokens.push_back(id_last);
        validate_tokens.insert(validate_tokens.end(), draft.begin(), draft.end());

        trace_step step;
        step.index = ++step_index;
        step.mode = summary.mode;
        step.draft = draft;
        step.validated_count = (int) validate_tokens.size();
        step.checkpoint_created = checkpoint_created;
        step.checkpoint_total = summary.checkpoint_count;
        step.restore_total = summary.restore_count;
        step.validate_processed_by_spec = true;
        step.sampler_before = common_sampler_prev_str(smpl, ctx_tgt, 8);
        step.kv_tgt_before_min = llama_memory_seq_pos_min(mem_tgt, 0);
        step.kv_tgt_before_max = llama_memory_seq_pos_max(mem_tgt, 0);
        step.kv_dft_before_min = llama_memory_seq_pos_min(mem_dft, 0);
        step.kv_dft_before_max = llama_memory_seq_pos_max(mem_dft, 0);

        llama_batch validate = llama_batch_init((int32_t) validate_tokens.size(), 0, 1);
        fill_batch(validate, validate_tokens, n_past);
        if (llama_decode(ctx_tgt, validate) != 0) {
            llama_batch_free(validate);
            summary.error = "target validate failed";
            common_sampler_free(smpl);
            return false;
        }
        summary.target_decode_calls++;
        if (!common_speculative_process(spec, validate)) {
            llama_batch_free(validate);
            summary.error = "speculative validate process failed";
            common_sampler_free(smpl);
            return false;
        }
        std::vector<int> rows(validate_tokens.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            rows[i] = (int) i;
        }
        auto * smpl_save = common_sampler_clone(smpl);
        auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, rows, draft);
        llama_batch_free(validate);
        if (ids.empty()) {
            if (smpl_save) {
                common_sampler_free(smpl_save);
            }
            summary.error = "acceptance produced no ids";
            common_sampler_free(smpl);
            return false;
        }

        const int accepted = std::max(0, (int) ids.size() - 1);
        summary.accepted_tokens_total += accepted;
        summary.rejected_tokens_total += (int) draft.size() - accepted;
        step.accepted_draft = accepted;
        step.accepted_ids = ids;
        step.sampler_after = common_sampler_prev_str(smpl, ctx_tgt, 8);

        if (accepted < (int) draft.size()) {
            step.resolution = "partial_restore";
            draft = ids;
            ckpt.load_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(mem_tgt, 0, ckpt.pos_max + 1, -1);
            ckpt.load_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
            prompt_tgt.resize((size_t) ckpt.n_tokens);
            n_past = (int32_t) prompt_tgt.size();
            common_sampler_free(smpl);
            smpl = smpl_save;
            smpl_save = nullptr;
            summary.restore_count++;
            step.restore_executed = true;
            step.restore_total = summary.restore_count;
            step.kv_tgt_after_min = llama_memory_seq_pos_min(mem_tgt, 0);
            step.kv_tgt_after_max = llama_memory_seq_pos_max(mem_tgt, 0);
            step.kv_dft_after_min = llama_memory_seq_pos_min(mem_dft, 0);
            step.kv_dft_after_max = llama_memory_seq_pos_max(mem_dft, 0);
            steps.push_back(step);
            continue;
        }

        if (smpl_save) {
            common_sampler_free(smpl_save);
        }
        step.resolution = "full_accept";
        common_speculative_accept(spec, 0, (uint16_t) accepted);
        for (llama_token tok : ids) {
            prompt_tgt.push_back(id_last);
            id_last = tok;
            if (llama_vocab_is_eog(llama_model_get_vocab(model_tgt), id_last)) {
                stop = true;
                break;
            }
            summary.output_tokens++;
            if (summary.output_tokens >= max_tokens) {
                break;
            }
        }
        n_past = (int32_t) prompt_tgt.size();
        llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
        llama_memory_seq_rm(mem_dft, 0, n_past, -1);
        draft.clear();

        step.kv_tgt_after_min = llama_memory_seq_pos_min(mem_tgt, 0);
        step.kv_tgt_after_max = llama_memory_seq_pos_max(mem_tgt, 0);
        step.kv_dft_after_min = llama_memory_seq_pos_min(mem_dft, 0);
        step.kv_dft_after_max = llama_memory_seq_pos_max(mem_dft, 0);
        steps.push_back(step);
    }

    common_sampler_free(smpl);
    summary.acceptance_ratio = summary.draft_tokens_total > 0
        ? (double) summary.accepted_tokens_total / (double) summary.draft_tokens_total
        : 0.0;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s MODE TARGET.gguf DRAFT.gguf PROMPT MAX_TOKENS\n", argv[0]);
        return 2;
    }

    const std::string mode = argv[1];
    const char * target = argv[2];
    const char * draft = argv[3];
    const char * prompt = argv[4];
    const int max_tokens = std::max(1, std::atoi(argv[5]));

    llama_backend_init();

    auto model_params = llama_model_default_params();
    auto * model_tgt = llama_model_load_from_file(target, model_params);
    auto * model_dft = llama_model_load_from_file(draft, model_params);
    if (!model_tgt || !model_dft) {
        std::fprintf(stderr, "failed to load models\n");
        return 1;
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 128;
    ctx_params.n_threads = 6;
    ctx_params.n_threads_batch = 6;
    ctx_params.n_outputs_max = 256;
    auto * ctx_tgt = llama_init_from_model(model_tgt, ctx_params);
    ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.ctx_other = ctx_tgt;
    ctx_params.n_rs_seq = 0;
    auto * ctx_dft = llama_init_from_model(model_dft, ctx_params);
    if (!ctx_tgt || !ctx_dft) {
        std::fprintf(stderr, "failed to create contexts\n");
        return 1;
    }

    common_params_speculative spec_params;
    spec_params.types = common_speculative_types_from_names({"draft-mtp"});
    spec_params.draft.n_max = ORBIT_MTP_DRAFT_N_MAX;
    spec_params.draft.ctx_tgt = ctx_tgt;
    spec_params.draft.ctx_dft = ctx_dft;
    auto * spec = common_speculative_init(spec_params, 1);
    if (!spec) {
        std::fprintf(stderr, "failed to init speculative state\n");
        return 1;
    }

    trace_summary summary;
    std::vector<trace_step> steps;
    bool ok = false;
    if (mode == "orbit-current") {
        ok = run_orbit_current(model_tgt, ctx_tgt, ctx_dft, spec, prompt, max_tokens, summary, steps);
    } else if (mode == "server-context-like") {
        ok = run_server_context_like(model_tgt, ctx_tgt, ctx_dft, spec, prompt, max_tokens, summary, steps);
    } else {
        summary.mode = mode;
        summary.error = "unknown mode";
    }

    print_trace_json(ok, summary, steps, ctx_tgt);

    common_speculative_free(spec);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    llama_backend_free();
    return ok ? 0 : 1;
}
