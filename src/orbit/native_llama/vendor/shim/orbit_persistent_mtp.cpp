#include "llama.h"
#include "common/speculative.h"
#include "common/sampling.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static void fill_batch(llama_batch & batch, const std::vector<llama_token> & tokens, int32_t pos0);
static void fill_target_prefill_batch(llama_batch & batch, const std::vector<llama_token> & tokens, int32_t pos0);
static std::string token_piece(const llama_vocab * vocab, llama_token token);
static bool can_partial_rollback(llama_context * ctx, uint32_t n_rollback);
using orbit_mtp_token_callback = void (*)(const char * text, void * user_data);
using orbit_mtp_progress_callback = void (*)(int32_t phase, int32_t current, int32_t total, void * user_data);

namespace {

thread_local std::string g_last_error;

static constexpr int32_t ORBIT_MTP_DRAFT_N_MAX = 3;

struct phase_stat {
    double total_ms = 0.0;
    int calls = 0;
};

static double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static void phase_add(phase_stat & stat, std::chrono::steady_clock::time_point start) {
    stat.total_ms += elapsed_ms(start);
    stat.calls += 1;
}

static void set_error(const char * message) {
    g_last_error = message ? message : "persistent mtp operation failed";
}

static bool partial_debug_enabled() {
    const char * value = std::getenv("ORBIT_MTP_PARTIAL_DEBUG");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static bool validate_debug_enabled() {
    const char * value = std::getenv("ORBIT_MTP_VALIDATE_DEBUG");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static bool boundary_split_enabled() {
    const char * value = std::getenv("ORBIT_MTP_BOUNDARY_SPLIT");
    if (!value || !value[0]) {
        return true;
    }
    return std::strcmp(value, "0") != 0;
}

static bool draft_trace_enabled() {
    const char * value = std::getenv("ORBIT_MTP_DRAFT_TRACE");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static bool chat_reuse_debug_enabled() {
    const char * value = std::getenv("ORBIT_MTP_CHAT_REUSE_DEBUG");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static void emit_orbit_frontier_trace(const char * label, const std::string & payload) {
    if (!partial_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "ORBIT_MTP_FRONTIER %s %s\n", label ? label : "event", payload.c_str());
}

static void emit_orbit_draft_trace(const std::string & payload) {
    if (!draft_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "ORBIT_MTP_DRAFT %s\n", payload.c_str());
}

static void emit_orbit_dft_trace(const std::string & payload) {
    if (!draft_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "ORBIT_MTP_DFT %s\n", payload.c_str());
}

static void emit_orbit_validate_trace(const char * label, const std::string & payload) {
    if (!validate_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "ORBIT_MTP_VALIDATE %s %s\n", label ? label : "event", payload.c_str());
}

static void emit_orbit_chat_reuse_trace(const char * label, const std::string & payload) {
    if (!chat_reuse_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "ORBIT_MTP_CHAT_REUSE_SHIM %s %s\n", label ? label : "event", payload.c_str());
}

static uint64_t stable_hash_string(const std::string & value) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char c : value) {
        hash ^= (uint64_t) c;
        hash *= 1099511628211ull;
    }
    return hash;
}

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

struct orbit_mtp_session {
    uint32_t n_batch = 0;
    llama_model * model_dft = nullptr;
    llama_context * ctx_dft = nullptr;
    common_speculative * spec = nullptr;
    common_params_speculative spec_params;
    common_prompt_checkpoint request_boundary_ckpt;
    std::vector<llama_token> request_boundary_prompt_tgt;
    long rss_before_kb = -1;
    long rss_after_init_kb = -1;
    long rss_peak_kb = -1;
    std::vector<llama_token> cached_prompt_tokens;
    std::vector<llama_token> committed_frontier_tokens;
    std::vector<llama_token> pending_followup_suffix_tokens;
    bool pending_followup_suffix_active = false;
    std::string last_raw_emitted_token_ids_json;
    std::string last_end_turn_frontier_token_ids_json;
    std::string last_content;
    int last_output_tokens = 0;
    int last_draft_tokens_total = 0;
    int last_accepted_tokens_total = 0;
    int last_rejected_tokens_total = 0;
    int last_reused_draft_tokens_total = 0;
    int last_reused_accepted_tokens_total = 0;
    int last_reused_rejected_tokens_total = 0;
    double last_acceptance_ratio = 0.0;
    double last_fresh_acceptance_ratio = 0.0;
    double last_consumed_acceptance_ratio = 0.0;
    int last_target_decode_calls = 0;
    int last_draft_decode_calls = 0;
    double last_elapsed_ms = 0.0;
    double last_tokens_per_second = 0.0;
    int last_full_accept_steps = 0;
    int last_replay_steps = 0;
    int last_partial_accept_steps = 0;
    int last_partial_no_replay_steps = 0;
    int last_replay_fallback_steps = 0;
    bool last_seq_rm_supported = false;
    int last_rollback_tokens_total = 0;
    int last_checkpoint_count = 0;
    int last_restore_count = 0;
    std::string last_trace_json;
    std::string last_timing_json;
    std::string last_validate_trace_json;
    std::string last_target_decode_trace_json;
    phase_stat phase_prefix_restore;
    phase_stat phase_suffix_decode_target;
    phase_stat phase_draft_generation;
    phase_stat phase_target_validate;
    phase_stat phase_speculative_process;
    phase_stat phase_sampler_clone;
    phase_stat phase_sampler_restore;
    phase_stat phase_sampler_ops;
    phase_stat phase_seq_rm;
    phase_stat phase_batch_build;
    phase_stat phase_ctx_tgt_checkpoint;
    phase_stat phase_ctx_tgt_restore;
    phase_stat phase_ctx_dft_checkpoint;
    phase_stat phase_ctx_dft_restore;
    phase_stat phase_rollback_replay;
    phase_stat phase_detokenize_bridge;
    phase_stat phase_loop_total;
    int debug_memory_clear_count = 0;
    int debug_seq_rm_count = 0;
    int debug_replay_count = 0;
    int debug_prefill_target_count = 0;
    int debug_prefill_target_suffix_count = 0;
    int debug_validate_decode_count = 0;
    int debug_draft_decode_count = 0;
};

enum class orbit_step_resolution {
    full_accept,
    live_partial,
    restored_partial,
    replay_fallback,
    error,
};

struct orbit_step_outcome {
    orbit_step_resolution resolution = orbit_step_resolution::error;
    std::vector<llama_token> ids;
};

struct orbit_trace_step {
    int index = 0;
    std::string sampler_before;
    std::string sampler_after;
    uint64_t sampler_before_hash = 0;
    uint64_t sampler_after_hash = 0;
    std::vector<llama_token> draft;
    std::vector<llama_token> accepted_ids;
    int accepted_draft = 0;
    int sampled_id = -1;
    int rejected_id = -1;
    std::string resolution;
    int validated_count = 0;
    int checkpoint_total = 0;
    int restore_total = 0;
    int32_t kv_tgt_before_min = -1;
    int32_t kv_tgt_before_max = -1;
    int32_t kv_tgt_after_min = -1;
    int32_t kv_tgt_after_max = -1;
    int32_t kv_dft_before_min = -1;
    int32_t kv_dft_before_max = -1;
    int32_t kv_dft_after_min = -1;
    int32_t kv_dft_after_max = -1;
    bool validate_processed_by_spec = false;
    double validate_batch_prepare_ms = 0.0;
    double validate_logits_rows_setup_ms = 0.0;
    double validate_llama_decode_ms = 0.0;
    double validate_post_decode_sample_ms = 0.0;
    int validate_batch_n_tokens = 0;
    int validate_batch_logits_count = 0;
    int validate_n_outputs_requested = 0;
    int id_last_before = -1;
    int id_last_after = -1;
    int32_t n_past_before = -1;
    int32_t n_past_after = -1;
    int32_t prompt_tgt_size_before = -1;
    int32_t prompt_tgt_size_after = -1;
    int32_t prompt_tgt_pos_next_before = -1;
    int32_t prompt_tgt_pos_next_after = -1;
    int32_t residual_draft_size_after = 0;
    std::string residual_draft_after_json = "[]";
    bool debug_enabled = false;
    std::string partial_state_before_json = "{}";
    std::string partial_state_after_restore_json = "{}";
    std::string partial_state_after_logical_commit_json = "{}";
    std::string sampler_checkpoint_used = "none";
    std::string next_draft_origin = "unknown";
    bool next_draft_is_fresh = false;
    int next_draft_size = 0;
    std::string next_draft_tokens_json = "[]";
    int next_validate_n_tok = 0;
    int fresh_draft_tokens_contrib = 0;
    int fresh_accepted_tokens_contrib = 0;
    int fresh_rejected_tokens_contrib = 0;
    double fresh_acceptance_ratio_contrib = 0.0;
    int consumed_draft_tokens_contrib = 0;
    int consumed_accepted_tokens_contrib = 0;
    int consumed_rejected_tokens_contrib = 0;
    double consumed_acceptance_ratio_contrib = 0.0;
    bool post_step_draft_is_fresh = false;
    bool post_step_need_replay = false;
    std::string extra_target_decode_reason = "none";
    std::string extra_draft_decode_reason = "none";
    int memory_clear_count = 0;
    int seq_rm_count = 0;
    int replay_count = 0;
    int prefill_count = 0;
    std::string pre_sample_state_json = "{}";
};

struct orbit_validate_trace {
    int step = 0;
    double batch_prepare_ms = 0.0;
    double logits_rows_setup_ms = 0.0;
    double llama_decode_validate_ms = 0.0;
    double post_decode_sample_ms = 0.0;
    int token_count_validated = 0;
    int n_seq_tokens = 0;
    int batch_n_tokens = 0;
    int batch_logits_count = 0;
    int n_outputs_requested = 0;
    int32_t kv_before_min = -1;
    int32_t kv_before_max = -1;
    int32_t kv_after_min = -1;
    int32_t kv_after_max = -1;
};

struct orbit_target_decode_trace {
    std::string phase;
    int step = 0;
    int draft_size = 0;
    int accepted_draft_expected = 0;
    long long started_us = 0;
    double decode_ms = 0.0;
    int batch_n_tokens = 0;
    int batch_logits_count = 0;
    int output_reserve_n_outputs = 0;
    std::vector<int> logits_flags;
    std::vector<llama_token> token_ids;
    std::vector<int32_t> positions;
    std::vector<int32_t> seq_ids;
};

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

static std::string token_piece_json(const llama_vocab * vocab, llama_token token) {
    return json_escape(token_piece(vocab, token));
}

static std::vector<llama_token> tail_tokens(const std::vector<llama_token> & tokens, size_t limit) {
    if (tokens.size() <= limit) {
        return tokens;
    }
    return std::vector<llama_token>(tokens.end() - (ptrdiff_t) limit, tokens.end());
}

static std::string token_vec_json(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "{\"id\":" << (int) tokens[i] << ",\"piece\":\"" << token_piece_json(vocab, tokens[i]) << "\"}";
    }
    out << "]";
    return out.str();
}

static std::string token_id_vec_json(const std::vector<llama_token> & tokens) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << (int) tokens[i];
    }
    out << "]";
    return out.str();
}

static std::string checkpoint_summary_json(const common_prompt_checkpoint & ckpt) {
    std::ostringstream out;
    out
        << "{"
        << "\"empty\":" << (ckpt.empty() ? "true" : "false")
        << ",\"n_tokens\":" << ckpt.n_tokens
        << ",\"pos_min\":" << ckpt.pos_min
        << ",\"pos_max\":" << ckpt.pos_max
        << ",\"data_tgt_bytes\":" << ckpt.data_tgt.size()
        << ",\"data_dft_bytes\":" << ckpt.data_dft.size()
        << "}";
    return out.str();
}

static std::string optional_rejected_json(const llama_vocab * vocab, const std::vector<llama_token> & draft, int accepted_draft) {
    if (accepted_draft < 0 || accepted_draft >= (int) draft.size()) {
        return "null";
    }
    std::ostringstream out;
    const auto tok = draft[(size_t) accepted_draft];
    out << "{\"id\":" << (int) tok << ",\"piece\":\"" << token_piece_json(vocab, tok) << "\"}";
    return out.str();
}

static std::string trace_step_json(const llama_vocab * vocab, const orbit_trace_step & step) {
    std::ostringstream out;
    out
        << "{\"index\":" << step.index
        << ",\"sampler_before\":\"" << json_escape(step.sampler_before) << "\""
        << ",\"sampler_after\":\"" << json_escape(step.sampler_after) << "\""
        << ",\"sampler_before_hash\":" << step.sampler_before_hash
        << ",\"sampler_after_hash\":" << step.sampler_after_hash
        << ",\"draft\":" << token_vec_json(vocab, step.draft)
        << ",\"accepted_ids\":" << token_vec_json(vocab, step.accepted_ids)
        << ",\"accepted_draft\":" << step.accepted_draft
        << ",\"sampled_id\":" << step.sampled_id
        << ",\"rejected_id\":" << step.rejected_id
        << ",\"first_rejected\":" << optional_rejected_json(vocab, step.draft, step.accepted_draft)
        << ",\"resolution\":\"" << step.resolution << "\""
        << ",\"id_last_before\":" << step.id_last_before
        << ",\"id_last_after\":" << step.id_last_after
        << ",\"n_past_before\":" << step.n_past_before
        << ",\"n_past_after\":" << step.n_past_after
        << ",\"prompt_tgt_size_before\":" << step.prompt_tgt_size_before
        << ",\"prompt_tgt_size_after\":" << step.prompt_tgt_size_after
        << ",\"prompt_tgt_pos_next_before\":" << step.prompt_tgt_pos_next_before
        << ",\"prompt_tgt_pos_next_after\":" << step.prompt_tgt_pos_next_after
        << ",\"residual_draft_size_after\":" << step.residual_draft_size_after
        << ",\"residual_draft_after\":" << step.residual_draft_after_json
        << ",\"validated_count\":" << step.validated_count
        << ",\"checkpoint_total\":" << step.checkpoint_total
        << ",\"restore_total\":" << step.restore_total
        << ",\"kv_tgt_before\":{\"min\":" << step.kv_tgt_before_min << ",\"max\":" << step.kv_tgt_before_max << "}"
        << ",\"kv_tgt_after\":{\"min\":" << step.kv_tgt_after_min << ",\"max\":" << step.kv_tgt_after_max << "}"
        << ",\"kv_dft_before\":{\"min\":" << step.kv_dft_before_min << ",\"max\":" << step.kv_dft_before_max << "}"
        << ",\"kv_dft_after\":{\"min\":" << step.kv_dft_after_min << ",\"max\":" << step.kv_dft_after_max << "}"
        << ",\"validate_processed_by_spec\":" << (step.validate_processed_by_spec ? "true" : "false")
        << ",\"validate_batch_prepare_ms\":" << step.validate_batch_prepare_ms
        << ",\"validate_logits_rows_setup_ms\":" << step.validate_logits_rows_setup_ms
        << ",\"validate_llama_decode_ms\":" << step.validate_llama_decode_ms
        << ",\"validate_post_decode_sample_ms\":" << step.validate_post_decode_sample_ms
        << ",\"validate_batch_n_tokens\":" << step.validate_batch_n_tokens
        << ",\"validate_batch_logits_count\":" << step.validate_batch_logits_count
        << ",\"validate_n_outputs_requested\":" << step.validate_n_outputs_requested
        ;
    if (step.debug_enabled) {
        out
            << ",\"partial_state_before\":" << step.partial_state_before_json
            << ",\"partial_state_after_restore\":" << step.partial_state_after_restore_json
            << ",\"partial_state_after_logical_commit\":" << step.partial_state_after_logical_commit_json
            << ",\"sampler_checkpoint_used\":\"" << json_escape(step.sampler_checkpoint_used) << "\""
            << ",\"next_draft_origin\":\"" << json_escape(step.next_draft_origin) << "\""
            << ",\"next_draft_is_fresh\":" << (step.next_draft_is_fresh ? "true" : "false")
            << ",\"next_draft_size\":" << step.next_draft_size
            << ",\"next_draft_tokens\":" << step.next_draft_tokens_json
            << ",\"next_validate_n_tok\":" << step.next_validate_n_tok
            << ",\"fresh_draft_tokens_contrib\":" << step.fresh_draft_tokens_contrib
            << ",\"fresh_accepted_tokens_contrib\":" << step.fresh_accepted_tokens_contrib
            << ",\"fresh_rejected_tokens_contrib\":" << step.fresh_rejected_tokens_contrib
            << ",\"fresh_acceptance_ratio_contrib\":" << step.fresh_acceptance_ratio_contrib
            << ",\"consumed_draft_tokens_contrib\":" << step.consumed_draft_tokens_contrib
            << ",\"consumed_accepted_tokens_contrib\":" << step.consumed_accepted_tokens_contrib
            << ",\"consumed_rejected_tokens_contrib\":" << step.consumed_rejected_tokens_contrib
            << ",\"consumed_acceptance_ratio_contrib\":" << step.consumed_acceptance_ratio_contrib
            << ",\"post_step_draft_is_fresh\":" << (step.post_step_draft_is_fresh ? "true" : "false")
            << ",\"post_step_need_replay\":" << (step.post_step_need_replay ? "true" : "false")
            << ",\"extra_target_decode_reason\":\"" << json_escape(step.extra_target_decode_reason) << "\""
            << ",\"extra_draft_decode_reason\":\"" << json_escape(step.extra_draft_decode_reason) << "\""
            << ",\"memory_clear_count\":" << step.memory_clear_count
            << ",\"seq_rm_count\":" << step.seq_rm_count
            << ",\"replay_count\":" << step.replay_count
            << ",\"prefill_count\":" << step.prefill_count
            << ",\"pre_sample_state\":" << step.pre_sample_state_json;
    }
    out << "}";
    return out.str();
}

static std::string validate_trace_json(const std::vector<orbit_validate_trace> & items) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        const auto & item = items[i];
        out
            << "{\"step\":" << item.step
            << ",\"batch_prepare_ms\":" << item.batch_prepare_ms
            << ",\"logits_rows_setup_ms\":" << item.logits_rows_setup_ms
            << ",\"llama_decode_validate_ms\":" << item.llama_decode_validate_ms
            << ",\"post_decode_sample_ms\":" << item.post_decode_sample_ms
            << ",\"token_count_validated\":" << item.token_count_validated
            << ",\"n_seq_tokens\":" << item.n_seq_tokens
            << ",\"batch_n_tokens\":" << item.batch_n_tokens
            << ",\"batch_logits_count\":" << item.batch_logits_count
            << ",\"n_outputs_requested\":" << item.n_outputs_requested
            << ",\"kv_before\":{\"min\":" << item.kv_before_min << ",\"max\":" << item.kv_before_max << "}"
            << ",\"kv_after\":{\"min\":" << item.kv_after_min << ",\"max\":" << item.kv_after_max << "}"
            << "}";
    }
    out << "]";
    return out.str();
}

static std::string int_vec_json(const std::vector<int> & values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

static std::string int32_vec_json(const std::vector<int32_t> & values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

static std::string target_decode_trace_json(const llama_vocab * vocab, const std::vector<orbit_target_decode_trace> & items) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        const auto & item = items[i];
        out
            << "{\"phase\":\"" << item.phase << "\""
            << ",\"step\":" << item.step
            << ",\"draft_size\":" << item.draft_size
            << ",\"accepted_draft_expected\":" << item.accepted_draft_expected
            << ",\"started_us\":" << item.started_us
            << ",\"decode_ms\":" << item.decode_ms
            << ",\"batch_n_tokens\":" << item.batch_n_tokens
            << ",\"batch_n_outputs_requested\":" << item.batch_logits_count
            << ",\"logits_count\":" << item.batch_logits_count
            << ",\"output_reserve_n_outputs\":" << item.output_reserve_n_outputs
            << ",\"logits_flags\":" << int_vec_json(item.logits_flags)
            << ",\"token_ids\":" << token_vec_json(vocab, item.token_ids)
            << ",\"positions\":" << int32_vec_json(item.positions)
            << ",\"seq_ids\":" << int32_vec_json(item.seq_ids)
            << "}";
    }
    out << "]";
    return out.str();
}

static orbit_target_decode_trace make_target_decode_trace(
    const char * phase,
    int step,
    int draft_size,
    int accepted_draft_expected,
    const llama_batch & batch,
    long long started_us = 0,
    double decode_ms = 0.0
) {
    orbit_target_decode_trace item;
    item.phase = phase ? phase : "";
    item.step = step;
    item.draft_size = draft_size;
    item.accepted_draft_expected = accepted_draft_expected;
    item.started_us = started_us;
    item.decode_ms = decode_ms;
    item.batch_n_tokens = batch.n_tokens;
    item.logits_flags.reserve((size_t) batch.n_tokens);
    item.token_ids.reserve((size_t) batch.n_tokens);
    item.positions.reserve((size_t) batch.n_tokens);
    item.seq_ids.reserve((size_t) batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int flag = batch.logits ? (batch.logits[i] ? 1 : 0) : 0;
        item.batch_logits_count += flag;
        item.logits_flags.push_back(flag);
        item.token_ids.push_back(batch.token ? batch.token[i] : LLAMA_TOKEN_NULL);
        item.positions.push_back(batch.pos ? batch.pos[i] : -1);
        int32_t seq0 = 0;
        if (batch.n_seq_id && batch.n_seq_id[i] > 0 && batch.seq_id && batch.seq_id[i]) {
            seq0 = batch.seq_id[i][0];
        }
        item.seq_ids.push_back(seq0);
    }
    // For this single-sequence non-embedding path, balloc->get_n_outputs() is
    // equal to the number of rows marked via batch.logits.
    item.output_reserve_n_outputs = item.batch_logits_count;
    return item;
}

static std::string phase_json(const phase_stat & stat) {
    std::ostringstream out;
    const double avg_ms = stat.calls > 0 ? stat.total_ms / (double) stat.calls : 0.0;
    out << "{\"total_ms\":" << stat.total_ms << ",\"calls\":" << stat.calls << ",\"avg_ms\":" << avg_ms << "}";
    return out.str();
}

static std::string partial_state_json(
    const llama_vocab * vocab,
    const char * stage,
    size_t prompt_tgt_size,
    int32_t prompt_tgt_pos_next,
    llama_token id_last,
    int32_t n_past,
    llama_memory_t mem_tgt,
    llama_memory_t mem_dft,
    bool sampler_checkpoint_used,
    const std::vector<llama_token> & draft,
    bool draft_is_fresh,
    const std::vector<llama_token> * ids
) {
    std::ostringstream out;
    out
        << "{"
        << "\"stage\":\"" << json_escape(stage ? stage : "") << "\""
        << ",\"prompt_tgt_size\":" << prompt_tgt_size
        << ",\"prompt_tgt_pos_next\":" << prompt_tgt_pos_next
        << ",\"id_last\":" << (int) id_last
        << ",\"n_past\":" << n_past
        << ",\"ctx_tgt_expected\":{\"min\":" << llama_memory_seq_pos_min(mem_tgt, 0) << ",\"max\":" << llama_memory_seq_pos_max(mem_tgt, 0) << "}"
        << ",\"ctx_dft_expected\":{\"min\":" << llama_memory_seq_pos_min(mem_dft, 0) << ",\"max\":" << llama_memory_seq_pos_max(mem_dft, 0) << "}"
        << ",\"sampler_checkpoint_used\":" << (sampler_checkpoint_used ? "true" : "false")
        << ",\"draft_origin\":\"" << (draft_is_fresh ? "fresh" : "reused") << "\""
        << ",\"draft_size\":" << draft.size()
        << ",\"draft\":" << token_vec_json(vocab, draft);
    if (ids) {
        out << ",\"ids\":" << token_vec_json(vocab, *ids);
    }
    out << "}";
    return out.str();
}

static std::string pre_sample_state_json(
    const llama_vocab * vocab,
    const std::vector<llama_token> & prompt_tgt,
    int32_t n_past,
    llama_token id_last,
    const std::vector<llama_token> & draft,
    const std::vector<llama_token> & validate_tokens,
    const std::vector<int> & validate_rows,
    llama_memory_t mem_tgt,
    llama_memory_t mem_dft,
    const std::string & sampler_before,
    bool have_ckpt,
    int32_t seq_rm_start_candidate
) {
    std::ostringstream out;
    out
        << "{"
        << "\"frontier_tail\":" << token_vec_json(vocab, tail_tokens(prompt_tgt, 16))
        << ",\"prompt_tgt_size\":" << prompt_tgt.size()
        << ",\"prompt_tgt_pos_next\":" << n_past
        << ",\"n_past\":" << n_past
        << ",\"id_last\":" << (int) id_last
        << ",\"spec_draft\":" << token_vec_json(vocab, draft)
        << ",\"spec_i_batch\":" << int_vec_json(validate_rows)
        << ",\"validate_tokens\":" << token_vec_json(vocab, validate_tokens)
        << ",\"ctx_tgt_max_pos\":" << llama_memory_seq_pos_max(mem_tgt, 0)
        << ",\"ctx_dft_max_pos\":" << llama_memory_seq_pos_max(mem_dft, 0)
        << ",\"sampler_summary\":\"" << json_escape(sampler_before) << "\""
        << ",\"sampler_hash\":" << stable_hash_string(sampler_before)
        << ",\"next_sample_logits_row\":" << (validate_rows.empty() ? -1 : validate_rows.back())
        << ",\"draft_tokens_committed_before_sample\":0"
        << ",\"seq_rm_start_candidate\":" << seq_rm_start_candidate
        << ",\"path_candidate\":\"" << (have_ckpt ? "checkpoint-capable" : "live-only") << "\""
        << ",\"frontier_kv_gap\":" << ((int64_t) n_past - 1 - (int64_t) llama_memory_seq_pos_max(mem_tgt, 0))
        << "}";
    return out.str();
}

static std::string frontier_event_json(
    const llama_vocab * vocab,
    const char * event,
    const char * origin,
    const std::vector<llama_token> & prompt_tgt,
    int32_t n_past,
    llama_token id_last,
    const std::vector<llama_token> & draft,
    llama_memory_t mem_tgt,
    llama_memory_t mem_dft,
    const std::string & sampler_summary,
    const std::vector<llama_token> * emitted_tokens
) {
    std::ostringstream out;
    out
        << "{"
        << "\"event\":\"" << json_escape(event ? event : "") << "\""
        << ",\"origin\":\"" << json_escape(origin ? origin : "") << "\""
        << ",\"prompt_tgt_size\":" << prompt_tgt.size()
        << ",\"prompt_tgt_pos_next\":" << n_past
        << ",\"n_past\":" << n_past
        << ",\"id_last\":" << (int) id_last
        << ",\"ctx_tgt_max\":" << llama_memory_seq_pos_max(mem_tgt, 0)
        << ",\"ctx_dft_max\":" << llama_memory_seq_pos_max(mem_dft, 0)
        << ",\"frontier_tail\":" << token_vec_json(vocab, tail_tokens(prompt_tgt, 16))
        << ",\"spec_draft\":" << token_vec_json(vocab, draft)
        << ",\"sampler_hash\":" << stable_hash_string(sampler_summary)
        << ",\"sampler_summary\":\"" << json_escape(sampler_summary) << "\"";
    if (emitted_tokens) {
        out << ",\"tokens\":" << token_vec_json(vocab, *emitted_tokens);
    }
    out << "}";
    return out.str();
}

static bool suffix_matches(
    const std::vector<llama_token> & haystack,
    const std::vector<llama_token> & needle
) {
    if (needle.size() > haystack.size()) {
        return false;
    }
    return std::equal(needle.begin(), needle.end(), haystack.end() - (ptrdiff_t) needle.size());
}

static std::string validate_pre_decode_json(
    const llama_vocab * vocab,
    const char * mode,
    const std::vector<llama_token> & prompt_tgt,
    int32_t n_past,
    llama_token id_last,
    const std::vector<llama_token> & draft,
    const std::vector<llama_token> & validate_tokens,
    const llama_batch & validate,
    llama_memory_t mem_tgt,
    llama_memory_t mem_dft
) {
    std::vector<int32_t> positions;
    positions.reserve((size_t) validate.n_tokens);
    for (int32_t i = 0; i < validate.n_tokens; ++i) {
        positions.push_back(validate.pos ? validate.pos[i] : -1);
    }
    const int32_t ctx_tgt_max = llama_memory_seq_pos_max(mem_tgt, 0);
    const int32_t ctx_dft_max = llama_memory_seq_pos_max(mem_dft, 0);
    const int32_t validate_start = positions.empty() ? -1 : positions.front();
    const int32_t validate_end = positions.empty() ? -1 : positions.back();
    const bool all_positions_already_in_kv = validate_start >= 0 && validate_end <= ctx_tgt_max;
    const int32_t kv_gap = validate_start >= 0 ? (validate_start - (ctx_tgt_max + 1)) : -1;
    std::ostringstream out;
    out
        << "{"
        << "\"mode\":\"" << json_escape(mode ? mode : "") << "\""
        << ",\"prompt_tgt_size\":" << prompt_tgt.size()
        << ",\"n_past\":" << n_past
        << ",\"ctx_tgt_max_pos\":" << ctx_tgt_max
        << ",\"ctx_dft_max_pos\":" << ctx_dft_max
        << ",\"id_last\":" << (int) id_last
        << ",\"draft\":" << token_vec_json(vocab, draft)
        << ",\"validate_tokens\":" << token_vec_json(vocab, validate_tokens)
        << ",\"validate_positions\":" << int32_vec_json(positions)
        << ",\"frontier_tail\":" << token_vec_json(vocab, tail_tokens(prompt_tgt, 16))
        << ",\"validate_tokens_in_frontier_suffix\":" << (suffix_matches(prompt_tgt, validate_tokens) ? "true" : "false")
        << ",\"positions_already_in_kv\":" << (all_positions_already_in_kv ? "true" : "false")
        << ",\"kv_gap_before_validate\":" << kv_gap
        << ",\"seq_id\":0"
        << "}";
    return out.str();
}

static orbit_step_outcome resolve_validate_accept_restore(
    orbit_mtp_session * session,
    common_speculative * spec,
    llama_context * ctx_tgt,
    llama_context * ctx_dft,
    llama_memory_t mem_tgt,
    llama_memory_t mem_dft,
    common_sampler *& smpl,
    const common_prompt_checkpoint & ckpt,
    bool have_ckpt,
    std::vector<llama_token> & prompt_tgt,
    std::vector<llama_token> & draft,
    bool & draft_is_fresh,
    int32_t & n_past,
    const std::vector<llama_token> & validate_tokens,
    int32_t validate_pos0,
    bool boundary_committed_live,
    size_t frontier_logical_base,
    const llama_vocab * vocab_tgt,
    orbit_trace_step * debug_trace_step,
    bool debug_partial,
    orbit_validate_trace * validate_trace,
    std::vector<orbit_target_decode_trace> * decode_traces,
    int trace_step_index,
    std::string * replay_reason
) {
    orbit_step_outcome outcome;

    if (validate_debug_enabled()) {
        emit_orbit_validate_trace("pre", validate_pre_decode_json(
            vocab_tgt,
            boundary_committed_live ? "committed_live_pre" : "baseline",
            prompt_tgt,
            n_past,
            validate_tokens.empty() ? LLAMA_TOKEN_NULL : validate_tokens.front(),
            draft,
            validate_tokens,
            llama_batch{},
            mem_tgt,
            mem_dft));

        std::vector<llama_token> prompt_shadow = prompt_tgt;
        prompt_shadow.insert(prompt_shadow.end(), validate_tokens.begin(), validate_tokens.end());
        llama_batch shadow = llama_batch_init((int32_t) validate_tokens.size(), 0, 1);
        fill_batch(shadow, validate_tokens, (int32_t) prompt_shadow.size());
        emit_orbit_validate_trace("pre", validate_pre_decode_json(
            vocab_tgt,
            "boundary_split_shadow",
            prompt_shadow,
            (int32_t) prompt_shadow.size(),
            validate_tokens.empty() ? LLAMA_TOKEN_NULL : validate_tokens.front(),
            draft,
            validate_tokens,
            shadow,
            mem_tgt,
            mem_dft));
        llama_batch_free(shadow);
    }

    const auto batch_prepare_start = std::chrono::steady_clock::now();
    llama_batch validate = llama_batch_init((int32_t) validate_tokens.size(), 0, 1);
    fill_batch(validate, validate_tokens, validate_pos0);
    phase_add(session->phase_batch_build, batch_prepare_start);
    if (validate_debug_enabled()) {
        emit_orbit_validate_trace("pre", validate_pre_decode_json(
            vocab_tgt,
            boundary_committed_live ? "committed_live_with_batch" : "baseline_with_batch",
            prompt_tgt,
            n_past,
            validate_tokens.empty() ? LLAMA_TOKEN_NULL : validate_tokens.front(),
            draft,
            validate_tokens,
            validate,
            mem_tgt,
            mem_dft));
    }
    if (decode_traces) {
        decode_traces->push_back(make_target_decode_trace(
            "validate",
            trace_step_index,
            (int) draft.size(),
            (int) draft.size(),
            validate));
    }
    if (validate_trace) {
        validate_trace->batch_prepare_ms = elapsed_ms(batch_prepare_start);
        validate_trace->token_count_validated = (int) validate_tokens.size();
        validate_trace->n_seq_tokens = (int) validate_tokens.size();
        validate_trace->batch_n_tokens = validate.n_tokens;
        int logits_count = 0;
        for (int32_t i = 0; i < validate.n_tokens; ++i) {
            logits_count += validate.logits[i] ? 1 : 0;
        }
        validate_trace->batch_logits_count = logits_count;
        validate_trace->n_outputs_requested = logits_count;
        validate_trace->kv_before_min = llama_memory_seq_pos_min(mem_tgt, 0);
        validate_trace->kv_before_max = llama_memory_seq_pos_max(mem_tgt, 0);
    }
    {
        const auto phase_start = std::chrono::steady_clock::now();
        if (llama_decode(ctx_tgt, validate) != 0) {
            if (validate_debug_enabled()) {
                std::ostringstream out;
                out << "{\"mode\":\"" << (boundary_committed_live ? "committed_live_with_batch" : "baseline_with_batch") << "\",\"llama_decode_rc\":-1}";
                emit_orbit_validate_trace("decode_error", out.str());
            }
            phase_add(session->phase_target_validate, phase_start);
            if (validate_trace) {
                validate_trace->llama_decode_validate_ms = elapsed_ms(phase_start);
            }
            llama_batch_free(validate);
            set_error("failed to validate speculative batch on target");
            return outcome;
        }
        phase_add(session->phase_target_validate, phase_start);
        if (validate_trace) {
            validate_trace->llama_decode_validate_ms = elapsed_ms(phase_start);
            validate_trace->kv_after_min = llama_memory_seq_pos_min(mem_tgt, 0);
            validate_trace->kv_after_max = llama_memory_seq_pos_max(mem_tgt, 0);
        }
        if (boundary_committed_live) {
            n_past = (int32_t) prompt_tgt.size();
        }
        if (validate_debug_enabled()) {
            std::ostringstream out;
            out
                << "{\"mode\":\"" << (boundary_committed_live ? "committed_live_with_batch" : "baseline_with_batch") << "\",\"llama_decode_rc\":0"
                << ",\"ctx_tgt_max_after\":" << llama_memory_seq_pos_max(mem_tgt, 0)
                << ",\"ctx_dft_max_after\":" << llama_memory_seq_pos_max(mem_dft, 0)
                << ",\"n_past_after\":" << n_past
                << "}";
            emit_orbit_validate_trace("decode_result", out.str());
        }
    }
    session->last_target_decode_calls++;
    if (spec) {
        const auto phase_start = std::chrono::steady_clock::now();
        const bool ok = common_speculative_process(spec, validate);
        phase_add(session->phase_speculative_process, phase_start);
        if (!ok) {
            llama_batch_free(validate);
            set_error("failed to process speculative validate batch");
            return outcome;
        }
    }

    common_sampler * smpl_save = nullptr;
    if (have_ckpt) {
        const auto phase_start = std::chrono::steady_clock::now();
        smpl_save = common_sampler_clone(smpl);
        phase_add(session->phase_sampler_clone, phase_start);
    }

    std::vector<int> validate_rows(validate_tokens.size());
    const auto logits_rows_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < validate_rows.size(); ++i) {
        validate_rows[i] = (int) i;
    }
    if (validate_trace) {
        validate_trace->logits_rows_setup_ms = elapsed_ms(logits_rows_start);
    }
    const auto sample_start = std::chrono::steady_clock::now();
    outcome.ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, validate_rows, draft);
    phase_add(session->phase_sampler_ops, sample_start);
    if (validate_trace) {
        validate_trace->post_decode_sample_ms = elapsed_ms(sample_start);
    }
    llama_batch_free(validate);

    if (outcome.ids.empty()) {
        if (smpl_save) {
            common_sampler_free(smpl_save);
        }
        set_error("speculative acceptance produced no ids");
        return outcome;
    }

    const int accepted = std::max(0, (int) outcome.ids.size() - 1);
    const uint32_t n_rollback = (uint32_t) ((int) draft.size() + 1 - (int) outcome.ids.size());
    if (draft_is_fresh) {
        session->last_rollback_tokens_total += (int) n_rollback;
        session->last_accepted_tokens_total += accepted;
        session->last_rejected_tokens_total += (int) draft.size() - accepted;
    } else {
        session->last_reused_draft_tokens_total += (int) draft.size();
        session->last_reused_accepted_tokens_total += accepted;
        session->last_reused_rejected_tokens_total += (int) draft.size() - accepted;
    }

    if (accepted == (int) draft.size()) {
        if (smpl_save) {
            common_sampler_free(smpl_save);
        }
        outcome.resolution = orbit_step_resolution::full_accept;
        if (debug_partial && debug_trace_step) {
            debug_trace_step->partial_state_after_logical_commit_json = partial_state_json(
                vocab_tgt,
                "full_accept_commit",
                prompt_tgt.size(),
                n_past,
                outcome.ids.empty() ? validate_tokens.front() : outcome.ids.back(),
                n_past,
                mem_tgt,
                mem_dft,
                false,
                draft,
                draft_is_fresh,
                &outcome.ids);
        }
        return outcome;
    }

    session->last_partial_accept_steps++;
    session->last_seq_rm_supported = false;
    if (boundary_committed_live && have_ckpt) {
        common_speculative_accept(spec, 0, (uint16_t) accepted);
        const size_t committed_prompt_size = frontier_logical_base + 1 + (size_t) accepted;
        if (committed_prompt_size <= prompt_tgt.size()) {
            prompt_tgt.resize(committed_prompt_size);
            n_past = (int32_t) prompt_tgt.size();
            {
                const auto phase_start = std::chrono::steady_clock::now();
                llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
                llama_memory_seq_rm(mem_dft, 0, n_past, -1);
                phase_add(session->phase_seq_rm, phase_start);
            }
            session->debug_seq_rm_count += 2;
            const bool live_ok =
                prompt_tgt.size() == (size_t) n_past &&
                llama_memory_seq_pos_max(mem_tgt, 0) == n_past - 1 &&
                llama_memory_seq_pos_max(mem_dft, 0) == n_past - 1;
            if (live_ok) {
                if (smpl_save) {
                    common_sampler_free(smpl_save);
                    smpl_save = nullptr;
                }
                draft.clear();
                draft_is_fresh = true;
                session->last_partial_no_replay_steps++;
                session->last_seq_rm_supported = true;
                outcome.resolution = orbit_step_resolution::live_partial;
                if (debug_partial && debug_trace_step) {
                    debug_trace_step->partial_state_after_logical_commit_json = partial_state_json(
                        vocab_tgt,
                        "after_live_partial_consume",
                        prompt_tgt.size(),
                        n_past,
                        outcome.ids.back(),
                        n_past,
                        mem_tgt,
                        mem_dft,
                        false,
                        draft,
                        draft_is_fresh,
                        &outcome.ids);
                }
                return outcome;
            }
            prompt_tgt.resize(frontier_logical_base);
            n_past = (int32_t) prompt_tgt.size();
        }
    }
    if (have_ckpt && smpl_save) {
        {
            const auto phase_start = std::chrono::steady_clock::now();
            ckpt.load_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            {
                const auto phase_start = std::chrono::steady_clock::now();
                llama_memory_seq_rm(mem_tgt, 0, ckpt.pos_max + 1, -1);
                phase_add(session->phase_seq_rm, phase_start);
            }
            session->debug_seq_rm_count++;
            phase_add(session->phase_ctx_tgt_restore, phase_start);
        }

        {
            const auto phase_start = std::chrono::steady_clock::now();
            ckpt.load_dft(ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (draft_trace_enabled()) {
                std::ostringstream out;
                out
                    << "{"
                    << "\"op\":\"load_dft_ckpt\""
                    << ",\"step_index\":" << (validate_trace ? validate_trace->step : -1)
                    << ",\"reason\":\"partial_restore\""
                    << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                    << ",\"n_past\":" << n_past
                    << ",\"id_last\":" << (int) validate_tokens.front()
                    << ",\"ctx_dft_max_after\":" << llama_memory_seq_pos_max(mem_dft, 0)
                    << ",\"frontier_tail\":" << token_vec_json(vocab_tgt, tail_tokens(prompt_tgt, 24))
                    << "}";
                emit_orbit_dft_trace(out.str());
            }
            {
                const auto phase_start_seq_rm = std::chrono::steady_clock::now();
                const int32_t before_max = llama_memory_seq_pos_max(mem_dft, 0);
                llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
                phase_add(session->phase_seq_rm, phase_start_seq_rm);
                if (draft_trace_enabled()) {
                    std::ostringstream out;
                    out
                        << "{"
                        << "\"op\":\"seq_rm_dft\""
                        << ",\"step_index\":" << (validate_trace ? validate_trace->step : -1)
                        << ",\"reason\":\"partial_restore\""
                        << ",\"start_pos\":" << (ckpt.pos_max + 1)
                        << ",\"end_pos\":-1"
                        << ",\"ctx_dft_max_before\":" << before_max
                        << ",\"ctx_dft_max_after\":" << llama_memory_seq_pos_max(mem_dft, 0)
                        << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                        << ",\"n_past\":" << n_past
                        << ",\"id_last\":" << (int) validate_tokens.front()
                        << ",\"frontier_tail\":" << token_vec_json(vocab_tgt, tail_tokens(prompt_tgt, 24))
                        << "}";
                    emit_orbit_dft_trace(out.str());
                }
            }
            session->debug_seq_rm_count++;
            phase_add(session->phase_ctx_dft_restore, phase_start);
        }

        if (debug_partial && debug_trace_step) {
            debug_trace_step->partial_state_after_restore_json = partial_state_json(
                vocab_tgt,
                "after_restore_before_logical_commit",
                prompt_tgt.size(),
                n_past,
                validate_tokens.front(),
                n_past,
                mem_tgt,
                mem_dft,
                true,
                draft,
                draft_is_fresh,
                &outcome.ids);
        }
        prompt_tgt.resize((size_t) ckpt.n_tokens);
        n_past = (int32_t) prompt_tgt.size();

        draft = outcome.ids;
        draft_is_fresh = false;

        {
            const auto phase_start = std::chrono::steady_clock::now();
            common_sampler_free(smpl);
            smpl = smpl_save;
            phase_add(session->phase_sampler_restore, phase_start);
        }
        smpl_save = nullptr;

        session->last_restore_count++;
        outcome.resolution = orbit_step_resolution::restored_partial;
        if (debug_partial && debug_trace_step) {
            debug_trace_step->partial_state_after_logical_commit_json = partial_state_json(
                vocab_tgt,
                "after_partial_restore_logical_frontier",
                prompt_tgt.size(),
                n_past,
                outcome.ids.back(),
                n_past,
                mem_tgt,
                mem_dft,
                true,
                draft,
                draft_is_fresh,
                &outcome.ids);
        }
        return outcome;
    }

    const bool smpl_clone_available = smpl_save != nullptr;
    if (smpl_save) {
        common_sampler_free(smpl_save);
        smpl_save = nullptr;
    }
    if (debug_partial && debug_trace_step) {
        debug_trace_step->partial_state_after_logical_commit_json = partial_state_json(
            vocab_tgt,
            "replay_fallback_decision",
            prompt_tgt.size(),
            n_past,
            validate_tokens.empty() ? LLAMA_TOKEN_NULL : validate_tokens.front(),
            n_past,
            mem_tgt,
            mem_dft,
            true,
            draft,
            draft_is_fresh,
            &outcome.ids);
        std::ostringstream replay_out;
        replay_out
            << "{"
            << "\"boundary_split_live\":" << (boundary_committed_live ? "true" : "false")
            << ",\"boundary_logical_base\":" << frontier_logical_base
            << ",\"have_ckpt\":" << (have_ckpt ? "true" : "false")
            << ",\"sampler_clone_available\":"
            << (smpl_clone_available ? "true" : "false")
            << ",\"n_past\":" << n_past
            << ",\"prompt_tgt_size\":" << prompt_tgt.size()
            << ",\"id_last\":" << (validate_tokens.empty() ? (int) LLAMA_TOKEN_NULL : (int) validate_tokens.front())
            << ",\"draft_size\":" << draft.size()
            << ",\"draft\":" << token_vec_json(vocab_tgt, draft)
            << ",\"draft_is_fresh\":" << (draft_is_fresh ? "true" : "false")
            << ",\"ctx_tgt_max_before\":" << llama_memory_seq_pos_max(mem_tgt, 0)
            << ",\"ctx_dft_max_before\":" << llama_memory_seq_pos_max(mem_dft, 0)
            << ",\"reason\":\"replay_fallback\""
            << "}";
        emit_orbit_validate_trace("need_replay", replay_out.str());
        emit_orbit_frontier_trace("need_replay", replay_out.str());
    }
    session->last_replay_fallback_steps++;
    if (replay_reason) {
        if (!have_ckpt) {
            *replay_reason = "replay_fallback: no checkpoint";
        } else if (boundary_committed_live) {
            *replay_reason = "replay_fallback: boundary split fallback";
        } else {
            *replay_reason = "replay_fallback: unsupported partial path";
        }
    }
    outcome.resolution = orbit_step_resolution::replay_fallback;
    return outcome;
}

static void cleanup_session(orbit_mtp_session * session) {
    if (!session) {
        return;
    }
    if (session->spec) {
        common_speculative_free(session->spec);
        session->spec = nullptr;
    }
    if (session->ctx_dft) {
        llama_free(session->ctx_dft);
        session->ctx_dft = nullptr;
    }
    if (session->model_dft) {
        llama_model_free(session->model_dft);
        session->model_dft = nullptr;
    }
}

static bool reset_speculative_request_state(
    orbit_mtp_session * session,
    llama_context * ctx_tgt
) {
    if (!session || !ctx_tgt) {
        set_error("missing speculative request state handles");
        return false;
    }

    if (session->spec) {
        common_speculative_free(session->spec);
        session->spec = nullptr;
    }

    session->spec_params.draft.ctx_tgt = ctx_tgt;
    session->spec_params.draft.ctx_dft = session->ctx_dft;
    session->spec = common_speculative_init(session->spec_params, 1);
    if (!session->spec) {
        set_error("failed to reinitialize speculative request state");
        return false;
    }

    return true;
}

} // namespace

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

static bool tokenize_prompt(const llama_model * model, const char * text, std::vector<llama_token> & out) {
    auto * vocab = llama_model_get_vocab(model);
    const bool add_special = std::strncmp(text, "<bos>", 5) != 0;
    const int32_t n_tok = -llama_tokenize(vocab, text, (int32_t) std::strlen(text), nullptr, 0, add_special, true);
    if (n_tok <= 0) {
        return false;
    }
    out.resize((size_t) n_tok);
    return llama_tokenize(vocab, text, (int32_t) std::strlen(text), out.data(), n_tok, add_special, true) >= 0;
}

static bool can_partial_rollback(
    llama_context * ctx,
    uint32_t n_rollback
) {
    const auto mode = common_context_can_seq_rm(ctx);
    if (mode == COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
        return true;
    }
    if (mode == COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
        return n_rollback <= (uint32_t) llama_n_rs_seq(ctx);
    }
    return false;
}

static common_params_sampling make_reference_sampling_params() {
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

static size_t shared_prefix_tokens(
    const std::vector<llama_token> & a,
    const std::vector<llama_token> & b
) {
    const size_t max_common = std::min(a.size(), b.size());
    size_t common = 0;
    while (common < max_common && a[common] == b[common]) {
        common++;
    }
    return common;
}

static bool is_token_prefix(
    const std::vector<llama_token> & prefix,
    const std::vector<llama_token> & tokens
) {
    return prefix.size() <= tokens.size() &&
        shared_prefix_tokens(prefix, tokens) == prefix.size();
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

static void fill_target_prefill_batch(llama_batch & batch, const std::vector<llama_token> & tokens, int32_t pos0) {
    batch.n_tokens = (int32_t) tokens.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[(size_t) i];
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = 1;
    }
}

extern "C" const char * orbit_mtp_last_error() {
    return g_last_error.c_str();
}

extern "C" void * orbit_mtp_session_create(
    const char * draft_model_path,
    void * ctx_tgt_ptr,
    uint32_t n_ctx,
    uint32_t n_batch,
    uint32_t n_ubatch,
    int32_t n_threads,
    int32_t n_threads_batch
) {
    g_last_error.clear();
    if (!draft_model_path || !draft_model_path[0]) {
        set_error("draft model path is required");
        return nullptr;
    }
    if (!ctx_tgt_ptr) {
        set_error("target context is required");
        return nullptr;
    }

    std::unique_ptr<orbit_mtp_session> session(new orbit_mtp_session());
    session->n_batch = std::max<uint32_t>(1, n_batch);
    session->rss_before_kb = rss_kb();
    session->rss_peak_kb = session->rss_before_kb;

    auto model_params = llama_model_default_params();
    session->model_dft = llama_model_load_from_file(draft_model_path, model_params);
    session->rss_peak_kb = std::max(session->rss_peak_kb, rss_kb());
    if (!session->model_dft) {
        set_error("failed to load draft model");
        return nullptr;
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_ubatch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads_batch;
    ctx_params.n_outputs_max = 1 + ORBIT_MTP_DRAFT_N_MAX;
    ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.n_rs_seq = 0;
    ctx_params.ctx_other = static_cast<llama_context *>(ctx_tgt_ptr);

    session->ctx_dft = llama_init_from_model(session->model_dft, ctx_params);
    session->rss_peak_kb = std::max(session->rss_peak_kb, rss_kb());
    if (!session->ctx_dft) {
        set_error("failed to create MTP draft context");
        cleanup_session(session.get());
        return nullptr;
    }

    session->spec_params.types = common_speculative_types_from_names({"draft-mtp"});
    session->spec_params.draft.n_max = ORBIT_MTP_DRAFT_N_MAX;
    session->spec_params.draft.ctx_tgt = static_cast<llama_context *>(ctx_tgt_ptr);
    session->spec_params.draft.ctx_dft = session->ctx_dft;

    session->spec = common_speculative_init(session->spec_params, 1);
    session->rss_peak_kb = std::max(session->rss_peak_kb, rss_kb());
    if (!session->spec) {
        set_error("failed to initialize speculative MTP state");
        cleanup_session(session.get());
        return nullptr;
    }

    session->rss_after_init_kb = rss_kb();
    return session.release();
}

extern "C" bool orbit_mtp_session_reset(void * handle, void * ctx_tgt_ptr) {
    g_last_error.clear();
    auto * session = static_cast<orbit_mtp_session *>(handle);
    auto * ctx_tgt = static_cast<llama_context *>(ctx_tgt_ptr);
    if (!session || !ctx_tgt) {
        set_error("persistent MTP session reset requires valid handles");
        return false;
    }

    if (session->spec) {
        common_speculative_free(session->spec);
        session->spec = nullptr;
    }

    auto * mem = llama_get_memory(session->ctx_dft);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    session->spec_params.draft.ctx_tgt = ctx_tgt;
    session->spec_params.draft.ctx_dft = session->ctx_dft;
    session->spec = common_speculative_init(session->spec_params, 1);
    if (!session->spec) {
        set_error("failed to reinitialize speculative MTP state");
        return false;
    }
    session->request_boundary_ckpt.clear();
    session->request_boundary_prompt_tgt.clear();
    session->cached_prompt_tokens.clear();
    session->committed_frontier_tokens.clear();
    session->pending_followup_suffix_tokens.clear();
    session->pending_followup_suffix_active = false;
    session->last_raw_emitted_token_ids_json = "[]";
    session->last_end_turn_frontier_token_ids_json = "[]";

    return true;
}

extern "C" void orbit_mtp_session_free(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    cleanup_session(session);
    delete session;
}

extern "C" void * orbit_mtp_session_ctx_dft(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? static_cast<void *>(session->ctx_dft) : nullptr;
}

extern "C" void * orbit_mtp_session_spec(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? static_cast<void *>(session->spec) : nullptr;
}

extern "C" bool orbit_mtp_session_set_followup_suffix_tokens(
    void * handle,
    const int32_t * token_ids,
    int32_t token_count
) {
    g_last_error.clear();
    auto * session = static_cast<orbit_mtp_session *>(handle);
    if (!session) {
        set_error("persistent MTP followup suffix requires a valid session");
        return false;
    }
    if (token_count < 0) {
        set_error("persistent MTP followup suffix token count must be non-negative");
        return false;
    }
    session->pending_followup_suffix_tokens.clear();
    session->pending_followup_suffix_active = true;
    if (token_count == 0) {
        return true;
    }
    if (!token_ids) {
        set_error("persistent MTP followup suffix tokens are missing");
        session->pending_followup_suffix_active = false;
        return false;
    }
    session->pending_followup_suffix_tokens.reserve((size_t) token_count);
    for (int32_t i = 0; i < token_count; ++i) {
        session->pending_followup_suffix_tokens.push_back((llama_token) token_ids[i]);
    }
    return true;
}

extern "C" long orbit_mtp_session_rss_before_kb(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->rss_before_kb : -1;
}

extern "C" long orbit_mtp_session_rss_after_init_kb(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->rss_after_init_kb : -1;
}

extern "C" long orbit_mtp_session_rss_peak_kb(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->rss_peak_kb : -1;
}

extern "C" bool orbit_mtp_session_complete(
    void * handle,
    void * ctx_tgt_ptr,
    const char * prompt_text,
    int32_t max_tokens,
    orbit_mtp_token_callback token_callback,
    orbit_mtp_progress_callback progress_callback,
    void * callback_user_data
) {
    g_last_error.clear();
    auto * session = static_cast<orbit_mtp_session *>(handle);
    auto * ctx_tgt = static_cast<llama_context *>(ctx_tgt_ptr);
    if (!session || !ctx_tgt || !prompt_text) {
        set_error("persistent MTP completion requires valid handles and prompt");
        return false;
    }
    if (!session->ctx_dft || !session->spec) {
        set_error("persistent MTP session is not initialized");
        return false;
    }
    if (!reset_speculative_request_state(session, ctx_tgt)) {
        return false;
    }

    session->last_content.clear();
    session->last_output_tokens = 0;
    session->last_draft_tokens_total = 0;
    session->last_accepted_tokens_total = 0;
    session->last_rejected_tokens_total = 0;
    session->last_reused_draft_tokens_total = 0;
    session->last_reused_accepted_tokens_total = 0;
    session->last_reused_rejected_tokens_total = 0;
    session->last_acceptance_ratio = 0.0;
    session->last_fresh_acceptance_ratio = 0.0;
    session->last_consumed_acceptance_ratio = 0.0;
    session->last_target_decode_calls = 0;
    session->last_draft_decode_calls = 0;
    session->last_elapsed_ms = 0.0;
    session->last_tokens_per_second = 0.0;
    session->last_full_accept_steps = 0;
    session->last_replay_steps = 0;
    session->last_partial_accept_steps = 0;
    session->last_partial_no_replay_steps = 0;
    session->last_replay_fallback_steps = 0;
    session->last_seq_rm_supported = false;
    session->last_rollback_tokens_total = 0;
    session->last_checkpoint_count = 0;
    session->last_restore_count = 0;
    session->last_trace_json = "[]";
    session->last_timing_json = "{}";
    session->last_validate_trace_json = "[]";
    session->last_target_decode_trace_json = "[]";
    session->last_raw_emitted_token_ids_json = "[]";
    session->last_end_turn_frontier_token_ids_json = "[]";
    session->phase_prefix_restore = {};
    session->phase_suffix_decode_target = {};
    session->phase_draft_generation = {};
    session->phase_target_validate = {};
    session->phase_speculative_process = {};
    session->phase_sampler_clone = {};
    session->phase_sampler_restore = {};
    session->phase_sampler_ops = {};
    session->phase_seq_rm = {};
    session->phase_batch_build = {};
    session->phase_ctx_tgt_checkpoint = {};
    session->phase_ctx_tgt_restore = {};
    session->phase_ctx_dft_checkpoint = {};
    session->phase_ctx_dft_restore = {};
    session->phase_rollback_replay = {};
    session->phase_detokenize_bridge = {};
    session->phase_loop_total = {};
    session->debug_memory_clear_count = 0;
    session->debug_seq_rm_count = 0;
    session->debug_replay_count = 0;
    session->debug_prefill_target_count = 0;
    session->debug_prefill_target_suffix_count = 0;
    session->debug_validate_decode_count = 0;
    session->debug_draft_decode_count = 0;

    auto * model_tgt = llama_get_model(ctx_tgt);
    auto * vocab_tgt = llama_model_get_vocab(model_tgt);
    std::vector<llama_token> prompt;
    if (!tokenize_prompt(model_tgt, prompt_text, prompt)) {
        set_error("failed to tokenize prompt");
        return false;
    }
    if (prompt.size() < 2) {
        set_error("prompt too short for persistent mtp completion");
        return false;
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(session->ctx_dft);
    if (!mem_tgt || !mem_dft) {
        set_error("failed to access llama memory");
        return false;
    }

    const bool use_live_followup_suffix = session->pending_followup_suffix_active;
    const std::vector<llama_token> followup_suffix_tokens = session->pending_followup_suffix_tokens;
    session->pending_followup_suffix_active = false;
    session->pending_followup_suffix_tokens.clear();
    std::vector<llama_token> prompt_tgt(prompt);
    if (use_live_followup_suffix) {
        if (session->committed_frontier_tokens.empty()) {
            set_error("missing committed frontier for persistent MTP followup reuse");
            return false;
        }
        prompt_tgt = session->committed_frontier_tokens;
        prompt_tgt.insert(prompt_tgt.end(), followup_suffix_tokens.begin(), followup_suffix_tokens.end());
    }
    llama_token id_last = LLAMA_TOKEN_NULL;
    int32_t n_past = (int32_t) prompt_tgt.size();

    auto sampling_params = make_reference_sampling_params();
    common_sampler * smpl = common_sampler_init(model_tgt, sampling_params);
    if (!smpl) {
        set_error("failed to initialize common sampler");
        return false;
    }
    if (chat_reuse_debug_enabled()) {
        std::ostringstream out;
        out
            << "{"
            << "\"prompt_tokenized_size\":" << prompt.size()
            << ",\"prompt_tail\":" << token_vec_json(vocab_tgt, tail_tokens(prompt, 48))
            << ",\"cached_prompt_tokens_size\":" << session->cached_prompt_tokens.size()
            << ",\"cached_prompt_tokens_tail\":" << token_vec_json(vocab_tgt, tail_tokens(session->cached_prompt_tokens, 48))
            << ",\"committed_frontier_tokens_size\":" << session->committed_frontier_tokens.size()
            << ",\"committed_frontier_tokens_tail\":" << token_vec_json(vocab_tgt, tail_tokens(session->committed_frontier_tokens, 48))
            << ",\"use_live_followup_suffix\":" << (use_live_followup_suffix ? "true" : "false")
            << ",\"followup_suffix_tokens\":" << token_vec_json(vocab_tgt, followup_suffix_tokens)
            << ",\"request_boundary_ckpt\":" << checkpoint_summary_json(session->request_boundary_ckpt)
            << ",\"request_boundary_prompt_tgt_size\":" << session->request_boundary_prompt_tgt.size()
            << ",\"request_boundary_prompt_tgt_tail\":" << token_vec_json(vocab_tgt, tail_tokens(session->request_boundary_prompt_tgt, 48))
            << ",\"prompt_tgt_size_initial\":" << prompt_tgt.size()
            << ",\"prompt_tgt_tail_initial\":" << token_vec_json(vocab_tgt, tail_tokens(prompt_tgt, 48))
            << ",\"n_past_initial\":" << n_past
            << ",\"ctx_tgt_max_pos_initial\":" << llama_memory_seq_pos_max(mem_tgt, 0)
            << ",\"ctx_dft_max_pos_initial\":" << llama_memory_seq_pos_max(mem_dft, 0)
            << ",\"sampler_hash_initial\":" << stable_hash_string(common_sampler_prev_str(smpl, ctx_tgt, 8))
            << "}";
        emit_orbit_chat_reuse_trace("start", out.str());
    }
    std::vector<llama_token> generated;
    generated.reserve((size_t) std::max(1, std::min(32, (int) max_tokens)));
    const auto t0 = std::chrono::steady_clock::now();
    const int32_t progress_prefill_phase = 0;
    const int32_t progress_generation_phase = 1;
    auto emit_output_token = [&](llama_token token) {
        session->last_output_tokens++;
        if (progress_callback) {
            progress_callback(progress_generation_phase, session->last_output_tokens, max_tokens, callback_user_data);
        }
        const auto phase_start = std::chrono::steady_clock::now();
        const std::string piece = token_piece(vocab_tgt, token);
        session->last_content += piece;
        phase_add(session->phase_detokenize_bridge, phase_start);
        if (token_callback && !piece.empty()) {
            token_callback(piece.c_str(), callback_user_data);
        }
    };
    bool need_replay = true;
    bool is_recovery_replay = false;
    std::vector<llama_token> draft;
    common_prompt_checkpoint ckpt;
    bool have_ckpt = false;
    bool draft_is_fresh = false;
    std::vector<orbit_trace_step> trace_steps;
    std::vector<orbit_validate_trace> validate_traces;
    std::vector<orbit_target_decode_trace> target_decode_traces;
    int trace_step_index = 0;
    int pending_partial_trace_index = -1;
    const bool debug_partial = partial_debug_enabled();
    bool frontier_trace_before_first_partial = true;
    const size_t reusable_request_prefix =
        session->request_boundary_ckpt.empty() ? 0 : session->request_boundary_prompt_tgt.size();
    const bool can_restore_request_boundary =
        !session->request_boundary_ckpt.empty() &&
        is_token_prefix(session->request_boundary_prompt_tgt, prompt_tgt);
    bool used_request_boundary = false;

    while ((int) generated.size() < std::max(1, std::min(32, (int) max_tokens))) {
        const auto loop_phase_start = std::chrono::steady_clock::now();
        if (need_replay) {
            const bool replay_is_recovery = is_recovery_replay;
            const auto replay_phase_start = std::chrono::steady_clock::now();
            const bool use_request_boundary = generated.empty() && can_restore_request_boundary && !used_request_boundary;
            const bool use_live_frontier_suffix = generated.empty() && use_live_followup_suffix;
            const size_t live_frontier_prefix = use_live_frontier_suffix ? session->committed_frontier_tokens.size() : 0;
            if (chat_reuse_debug_enabled() && trace_step_index == 0) {
                std::ostringstream out;
                out
                    << "{"
                    << "\"path\":\"" << (
                        use_live_frontier_suffix ? "live_frontier_suffix" :
                        (use_request_boundary ? "request_boundary_restore" : "full_replay")) << "\""
                    << ",\"generated_empty\":" << (generated.empty() ? "true" : "false")
                    << ",\"use_live_frontier_suffix\":" << (use_live_frontier_suffix ? "true" : "false")
                    << ",\"can_restore_request_boundary\":" << (can_restore_request_boundary ? "true" : "false")
                    << ",\"used_request_boundary\":" << (used_request_boundary ? "true" : "false")
                    << ",\"reusable_request_prefix\":" << reusable_request_prefix
                    << ",\"live_frontier_prefix\":" << live_frontier_prefix
                    << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                    << ",\"ctx_tgt_max_pos\":" << llama_memory_seq_pos_max(mem_tgt, 0)
                    << ",\"ctx_dft_max_pos\":" << llama_memory_seq_pos_max(mem_dft, 0)
                    << "}";
                emit_orbit_chat_reuse_trace("replay_path", out.str());
            }
            if (debug_partial) {
                const auto replay_origin =
                    use_request_boundary ? "request_boundary" :
                    (prompt_tgt.empty() ? "initial_replay" : "target_replay_suffix_or_validate");
                emit_orbit_frontier_trace("replay", frontier_event_json(
                    vocab_tgt,
                    replay_origin,
                    "replay_entry",
                    prompt_tgt,
                    n_past,
                    id_last,
                    draft,
                    mem_tgt,
                    mem_dft,
                    common_sampler_prev_str(smpl, ctx_tgt, 8),
                    nullptr));
            }
            if (replay_is_recovery) {
                session->last_replay_steps++;
                session->debug_replay_count++;
            }
            if (!use_live_frontier_suffix) {
                llama_memory_clear(mem_tgt, true);
                llama_memory_clear(mem_dft, true);
                session->debug_memory_clear_count += 2;
            }

            if (!prompt_tgt.empty()) {
                if (use_request_boundary) {
                    {
                        const auto phase_start = std::chrono::steady_clock::now();
                        session->request_boundary_ckpt.load_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        session->request_boundary_ckpt.load_dft(session->ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        phase_add(session->phase_prefix_restore, phase_start);
                    }
                    used_request_boundary = true;
                } else if (!use_live_frontier_suffix) {
                    const size_t chunk_size = (size_t) std::max<uint32_t>(1, session->n_batch);
                    for (size_t offset = 0; offset < prompt_tgt.size(); offset += chunk_size) {
                        const size_t count = std::min(chunk_size, prompt_tgt.size() - offset);
                        std::vector<llama_token> chunk(
                            prompt_tgt.begin() + (ptrdiff_t) offset,
                            prompt_tgt.begin() + (ptrdiff_t) (offset + count));
                        const auto batch_build_start = std::chrono::steady_clock::now();
                        llama_batch prefill_tgt = llama_batch_init((int32_t) count, 0, 1);
                        fill_target_prefill_batch(prefill_tgt, chunk, (int32_t) offset);
                        phase_add(session->phase_batch_build, batch_build_start);
                        const long long decode_started_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
                        double decode_ms = 0.0;
                        session->debug_prefill_target_count++;
                        {
                            const auto phase_start = std::chrono::steady_clock::now();
                            if (llama_decode(ctx_tgt, prefill_tgt) != 0) {
                                decode_ms = elapsed_ms(phase_start);
                                target_decode_traces.push_back(make_target_decode_trace(
                                    "prefill_target",
                                    trace_step_index,
                                    0,
                                    0,
                                    prefill_tgt,
                                    decode_started_us,
                                    decode_ms));
                                phase_add(session->phase_suffix_decode_target, phase_start);
                                llama_batch_free(prefill_tgt);
                                common_sampler_free(smpl);
                                set_error("failed to decode target prefill");
                                return false;
                            }
                            decode_ms = elapsed_ms(phase_start);
                            phase_add(session->phase_suffix_decode_target, phase_start);
                        }
                        target_decode_traces.push_back(make_target_decode_trace(
                            "prefill_target",
                            trace_step_index,
                            0,
                            0,
                            prefill_tgt,
                            decode_started_us,
                            decode_ms));
                        session->last_target_decode_calls++;
                        llama_batch_free(prefill_tgt);
                        if (progress_callback) {
                            progress_callback(
                                progress_prefill_phase,
                                (int32_t) (offset + count),
                                (int32_t) prompt_tgt.size(),
                                callback_user_data);
                        }
                    }
                }
                std::vector<llama_token> process_tokens;
                int32_t process_pos0 = 0;
                if (use_request_boundary) {
                    process_tokens.assign(
                        prompt_tgt.begin() + (ptrdiff_t) reusable_request_prefix,
                        prompt_tgt.end());
                    process_pos0 = (int32_t) reusable_request_prefix;
                } else if (use_live_frontier_suffix) {
                    process_tokens = followup_suffix_tokens;
                    process_pos0 = (int32_t) live_frontier_prefix;
                } else {
                    process_tokens = prompt_tgt;
                    process_pos0 = 0;
                }
                if (!process_tokens.empty()) {
                    if (use_request_boundary || use_live_frontier_suffix) {
                        const size_t chunk_size = (size_t) std::max<uint32_t>(1, session->n_batch);
                        for (size_t offset = 0; offset < process_tokens.size(); offset += chunk_size) {
                            const size_t count = std::min(chunk_size, process_tokens.size() - offset);
                            std::vector<llama_token> chunk(
                                process_tokens.begin() + (ptrdiff_t) offset,
                                process_tokens.begin() + (ptrdiff_t) (offset + count));
                            const auto batch_build_start = std::chrono::steady_clock::now();
                            llama_batch prefill_tgt = llama_batch_init((int32_t) count, 0, 1);
                            fill_target_prefill_batch(prefill_tgt, chunk, process_pos0 + (int32_t) offset);
                            phase_add(session->phase_batch_build, batch_build_start);
                            const long long decode_started_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
                            double decode_ms = 0.0;
                            session->debug_prefill_target_suffix_count++;
                            {
                                const auto phase_start = std::chrono::steady_clock::now();
                                if (llama_decode(ctx_tgt, prefill_tgt) != 0) {
                                    decode_ms = elapsed_ms(phase_start);
                                    target_decode_traces.push_back(make_target_decode_trace(
                                        "prefill_target_suffix",
                                        trace_step_index,
                                        0,
                                        0,
                                        prefill_tgt,
                                        decode_started_us,
                                        decode_ms));
                                    phase_add(session->phase_suffix_decode_target, phase_start);
                                    llama_batch_free(prefill_tgt);
                                    common_sampler_free(smpl);
                                    set_error("failed to decode target prefill suffix");
                                    return false;
                                }
                                decode_ms = elapsed_ms(phase_start);
                                phase_add(session->phase_suffix_decode_target, phase_start);
                            }
                            target_decode_traces.push_back(make_target_decode_trace(
                                "prefill_target_suffix",
                                trace_step_index,
                                0,
                                0,
                                prefill_tgt,
                                decode_started_us,
                                decode_ms));
                            session->last_target_decode_calls++;
                            llama_batch_free(prefill_tgt);
                            if (progress_callback) {
                                progress_callback(
                                    progress_prefill_phase,
                                    process_pos0 + (int32_t) (offset + count),
                                    (int32_t) prompt_tgt.size(),
                                    callback_user_data);
                            }
                        }
                    }
                    const size_t chunk_size = (size_t) std::max<uint32_t>(1, session->n_batch);
                    for (size_t offset = 0; offset < process_tokens.size(); offset += chunk_size) {
                        const size_t count = std::min(chunk_size, process_tokens.size() - offset);
                        std::vector<llama_token> chunk(
                            process_tokens.begin() + (ptrdiff_t) offset,
                            process_tokens.begin() + (ptrdiff_t) (offset + count));
                        const auto batch_build_start = std::chrono::steady_clock::now();
                        llama_batch prefill = llama_batch_init((int32_t) count, 0, 1);
                        fill_batch(prefill, chunk, process_pos0 + (int32_t) offset);
                        phase_add(session->phase_batch_build, batch_build_start);
                        {
                            const auto phase_start = std::chrono::steady_clock::now();
                            const bool ok = common_speculative_process(session->spec, prefill);
                            phase_add(session->phase_speculative_process, phase_start);
                            if (!ok) {
                                llama_batch_free(prefill);
                                common_sampler_free(smpl);
                                set_error("failed to process speculative prefill");
                                return false;
                            }
                        }
                        llama_batch_free(prefill);
                    }
                }
                if (progress_callback) {
                    progress_callback(
                        progress_prefill_phase,
                        (int32_t) prompt_tgt.size(),
                        (int32_t) prompt_tgt.size(),
                        callback_user_data);
                }
                common_speculative_begin(session->spec, 0, prompt_tgt);
                if (generated.empty()) {
                    session->request_boundary_ckpt.clear();
                    session->request_boundary_ckpt.update_pos(
                        (int64_t) prompt_tgt.size(),
                        llama_memory_seq_pos_min(mem_tgt, 0),
                        llama_memory_seq_pos_max(mem_tgt, 0));
                    {
                        const auto phase_start = std::chrono::steady_clock::now();
                        session->request_boundary_ckpt.update_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        phase_add(session->phase_ctx_tgt_checkpoint, phase_start);
                    }
                    {
                        const auto phase_start = std::chrono::steady_clock::now();
                        session->request_boundary_ckpt.update_dft(session->ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        phase_add(session->phase_ctx_dft_checkpoint, phase_start);
                    }
                    session->last_checkpoint_count++;
                    session->request_boundary_prompt_tgt = prompt_tgt;
                }
            } else {
                common_speculative_begin(session->spec, 0, prompt_tgt);
                if (generated.empty()) {
                    session->request_boundary_ckpt.clear();
                    session->request_boundary_prompt_tgt.clear();
                }
            }
            draft.clear();
            ckpt.clear();
            have_ckpt = false;
            draft_is_fresh = false;
            n_past = (int32_t) prompt_tgt.size();
            if (generated.empty()) {
                {
                    const auto phase_start = std::chrono::steady_clock::now();
                    id_last = common_sampler_sample(smpl, ctx_tgt, -1);
                    common_sampler_accept(smpl, id_last, true);
                    phase_add(session->phase_sampler_ops, phase_start);
                }
                if (llama_vocab_is_eog(vocab_tgt, id_last)) {
                    goto done;
                }
                generated.push_back(id_last);
                emit_output_token(id_last);
                if (chat_reuse_debug_enabled()) {
                    std::ostringstream out;
                    out
                        << "{"
                        << "\"first_sampled_id\":{\"id\":" << (int) id_last << ",\"piece\":\"" << token_piece_json(vocab_tgt, id_last) << "\"}"
                        << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                        << ",\"n_past\":" << n_past
                        << ",\"ctx_tgt_max_pos\":" << llama_memory_seq_pos_max(mem_tgt, 0)
                        << ",\"ctx_dft_max_pos\":" << llama_memory_seq_pos_max(mem_dft, 0)
                        << ",\"sampler_hash\":" << stable_hash_string(common_sampler_prev_str(smpl, ctx_tgt, 8))
                        << "}";
                    emit_orbit_chat_reuse_trace("first_sample", out.str());
                }
                if (frontier_trace_before_first_partial) {
                    const std::vector<llama_token> tok = {id_last};
                    emit_orbit_frontier_trace("advance", frontier_event_json(
                        vocab_tgt,
                        "initial_sample",
                        "target_sample",
                        prompt_tgt,
                        n_past,
                        id_last,
                        draft,
                        mem_tgt,
                        mem_dft,
                        common_sampler_prev_str(smpl, ctx_tgt, 8),
                        &tok));
                }
                if ((int) generated.size() >= std::max(1, std::min(32, (int) max_tokens))) {
                    goto done;
                }
            }
            need_replay = false;
            if (replay_is_recovery) {
                phase_add(session->phase_rollback_replay, replay_phase_start);
            }
            is_recovery_replay = false;
        }

        size_t n_draft = draft.size();
        if (draft.empty()) {
            const int32_t draft_ctx_tgt_max_before = llama_memory_seq_pos_max(mem_tgt, 0);
            const int32_t draft_ctx_dft_max_before = llama_memory_seq_pos_max(mem_dft, 0);
            const std::string draft_sampler_before = common_sampler_prev_str(smpl, ctx_tgt, 8);
            ckpt.update_pos(
                (int64_t) prompt_tgt.size(),
                llama_memory_seq_pos_min(mem_tgt, 0),
                llama_memory_seq_pos_max(mem_tgt, 0));
            ckpt.update_dft(session->ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (draft_trace_enabled()) {
                std::ostringstream out;
                out
                    << "{"
                    << "\"op\":\"update_dft_ckpt\""
                    << ",\"step_index\":" << (trace_step_index + 1)
                    << ",\"reason\":\"before_fresh_draft\""
                    << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                    << ",\"n_past\":" << n_past
                    << ",\"id_last\":" << (int) id_last
                    << ",\"ctx_dft_max_after\":" << llama_memory_seq_pos_max(mem_dft, 0)
                    << ",\"frontier_tail\":" << token_vec_json(vocab_tgt, tail_tokens(prompt_tgt, 24))
                    << "}";
                emit_orbit_dft_trace(out.str());
            }

            common_speculative_get_draft_params(session->spec, 0) = {
                true,
                std::min(ORBIT_MTP_DRAFT_N_MAX, std::max(1, std::min(32, (int) max_tokens)) - (int) generated.size()),
                (llama_pos) n_past,
                id_last,
                &prompt_tgt,
                &draft,
            };
            {
                const auto phase_start = std::chrono::steady_clock::now();
                common_speculative_draft(session->spec);
                phase_add(session->phase_draft_generation, phase_start);
            }
            if (draft_trace_enabled()) {
                std::ostringstream out;
                out
                    << "{"
                    << "\"op\":\"draft_decode\""
                    << ",\"step_index\":" << (trace_step_index + 1)
                    << ",\"reason\":\"fresh_draft_generation\""
                    << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                    << ",\"n_past\":" << n_past
                    << ",\"id_last\":" << (int) id_last
                    << ",\"draft_tokens\":" << token_vec_json(vocab_tgt, draft)
                    << ",\"draft_positions_start\":" << n_past
                    << ",\"draft_positions_end\":" << (draft.empty() ? n_past - 1 : n_past + (int32_t) draft.size() - 1)
                    << ",\"n_outputs\":" << (int) draft.size()
                    << ",\"ctx_dft_max_before\":" << draft_ctx_dft_max_before
                    << ",\"ctx_dft_max_after_expected\":" << (draft.empty() ? draft_ctx_dft_max_before : draft_ctx_dft_max_before + (int32_t) draft.size())
                    << ",\"frontier_tail\":" << token_vec_json(vocab_tgt, tail_tokens(prompt_tgt, 24))
                    << "}";
                emit_orbit_dft_trace(out.str());
            }
            session->last_draft_decode_calls++;
            session->debug_draft_decode_count++;
            session->last_draft_tokens_total += (int) draft.size();
            draft_is_fresh = true;
            n_draft = draft.size();
            if (chat_reuse_debug_enabled() && trace_step_index == 0) {
                std::ostringstream out;
                out
                    << "{"
                    << "\"first_draft_tokens\":" << token_vec_json(vocab_tgt, draft)
                    << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                    << ",\"n_past\":" << n_past
                    << ",\"ctx_dft_max_before\":" << draft_ctx_dft_max_before
                    << ",\"ctx_dft_max_after_expected\":" << (draft.empty() ? draft_ctx_dft_max_before : draft_ctx_dft_max_before + (int32_t) draft.size())
                    << "}";
                emit_orbit_chat_reuse_trace("first_draft", out.str());
            }
            if (draft_trace_enabled()) {
                const uint64_t sampler_hash_before = stable_hash_string(draft_sampler_before);
                const uint64_t sampler_hash_after = stable_hash_string(common_sampler_prev_str(smpl, ctx_tgt, 8));
                for (size_t i = 0; i < draft.size(); ++i) {
                    const llama_token input_token = i == 0 ? id_last : draft[i - 1];
                    const llama_token sampled_token = draft[i];
                    std::ostringstream out;
                    out
                        << "{"
                        << "\"step_index\":" << (trace_step_index + 1)
                        << ",\"draft_index\":" << (int) i
                        << ",\"input_token\":{\"id\":" << (int) input_token << ",\"piece\":\"" << token_piece_json(vocab_tgt, input_token) << "\"}"
                        << ",\"sampled_draft_token\":{\"id\":" << (int) sampled_token << ",\"piece\":\"" << token_piece_json(vocab_tgt, sampled_token) << "\"}"
                        << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                        << ",\"n_past\":" << n_past
                        << ",\"ctx_tgt_max_before\":" << draft_ctx_tgt_max_before
                        << ",\"ctx_dft_max_before\":" << draft_ctx_dft_max_before
                        << ",\"ctx_dft_max_after_expected\":" << (draft_ctx_dft_max_before + (int32_t) i + 1)
                        << ",\"batch_position\":" << (n_past + (int32_t) i)
                        << ",\"logits_row\":" << (int) i
                        << ",\"sampler_hash_before\":" << sampler_hash_before
                        << ",\"sampler_hash_after\":" << sampler_hash_after
                        << ",\"memory_clear_count\":" << session->debug_memory_clear_count
                        << ",\"seq_rm_count\":" << session->debug_seq_rm_count
                        << ",\"batch_n_tokens\":" << (int) draft.size()
                        << ",\"batch_n_outputs\":" << (int) draft.size()
                        << ",\"boundary_split\":" << (boundary_split_enabled() ? "true" : "false")
                        << "}";
                    emit_orbit_draft_trace(out.str());
                }
            }
            if (frontier_trace_before_first_partial) {
                emit_orbit_frontier_trace("advance", frontier_event_json(
                    vocab_tgt,
                    "draft_generated",
                    "draft_generation",
                    prompt_tgt,
                    n_past,
                    id_last,
                    draft,
                    mem_tgt,
                    mem_dft,
                    common_sampler_prev_str(smpl, ctx_tgt, 8),
                    nullptr));
            }

            if (!draft.empty()) {
                {
                    const auto phase_start = std::chrono::steady_clock::now();
                    ckpt.update_tgt(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    phase_add(session->phase_ctx_tgt_checkpoint, phase_start);
                }
                {
                    const auto phase_start = std::chrono::steady_clock::now();
                    ckpt.load_dft(session->ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    if (draft_trace_enabled()) {
                        std::ostringstream out;
                        out
                            << "{"
                            << "\"op\":\"load_dft_ckpt\""
                            << ",\"step_index\":" << (trace_step_index + 1)
                            << ",\"reason\":\"post_fresh_draft_restore\""
                            << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                            << ",\"n_past\":" << n_past
                            << ",\"id_last\":" << (int) id_last
                            << ",\"ctx_dft_max_after\":" << llama_memory_seq_pos_max(mem_dft, 0)
                            << ",\"frontier_tail\":" << token_vec_json(vocab_tgt, tail_tokens(prompt_tgt, 24))
                            << "}";
                        emit_orbit_dft_trace(out.str());
                    }
                    {
                        const auto phase_start = std::chrono::steady_clock::now();
                        const int32_t before_max = llama_memory_seq_pos_max(mem_dft, 0);
                        llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
                        phase_add(session->phase_seq_rm, phase_start);
                        if (draft_trace_enabled()) {
                            std::ostringstream out;
                            out
                                << "{"
                                << "\"op\":\"seq_rm_dft\""
                                << ",\"step_index\":" << (trace_step_index + 1)
                                << ",\"reason\":\"post_fresh_draft_trim\""
                                << ",\"start_pos\":" << (ckpt.pos_max + 1)
                                << ",\"end_pos\":-1"
                                << ",\"ctx_dft_max_before\":" << before_max
                                << ",\"ctx_dft_max_after\":" << llama_memory_seq_pos_max(mem_dft, 0)
                                << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                                << ",\"n_past\":" << n_past
                                << ",\"id_last\":" << (int) id_last
                                << ",\"frontier_tail\":" << token_vec_json(vocab_tgt, tail_tokens(prompt_tgt, 24))
                                << "}";
                            emit_orbit_dft_trace(out.str());
                        }
                    }
                    session->debug_seq_rm_count++;
                    phase_add(session->phase_ctx_dft_restore, phase_start);
                }
                session->last_checkpoint_count++;
                have_ckpt = true;
            } else {
                have_ckpt = false;
            }
        }

        std::vector<llama_token> validate_tokens;
        validate_tokens.reserve(draft.size() + 1);
        validate_tokens.push_back(id_last);
        validate_tokens.insert(validate_tokens.end(), draft.begin(), draft.end());
        const int32_t validate_pos0 = n_past;
        const size_t frontier_logical_base = prompt_tgt.size();
        bool boundary_committed_live = false;
        if (boundary_split_enabled() &&
            have_ckpt &&
            !draft.empty() &&
            n_past == (int32_t) prompt_tgt.size() &&
            llama_memory_seq_pos_max(mem_tgt, 0) == n_past - 1 &&
            llama_memory_seq_pos_max(mem_dft, 0) == n_past - 1) {
            prompt_tgt.push_back(id_last);
            prompt_tgt.insert(prompt_tgt.end(), draft.begin(), draft.end());
            boundary_committed_live = true;
        }
        if (chat_reuse_debug_enabled() && trace_step_index == 0) {
            std::ostringstream out;
            out
                << "{"
                << "\"first_validate_n_tok\":" << validate_tokens.size()
                << ",\"validate_tokens\":" << token_vec_json(vocab_tgt, validate_tokens)
                << ",\"validate_pos0\":" << validate_pos0
                << ",\"boundary_committed_live\":" << (boundary_committed_live ? "true" : "false")
                << ",\"frontier_logical_base\":" << frontier_logical_base
                << ",\"prompt_tgt_size\":" << prompt_tgt.size()
                << ",\"n_past\":" << n_past
                << ",\"ctx_tgt_max_pos\":" << llama_memory_seq_pos_max(mem_tgt, 0)
                << ",\"ctx_dft_max_pos\":" << llama_memory_seq_pos_max(mem_dft, 0)
                << "}";
            emit_orbit_chat_reuse_trace("first_validate", out.str());
        }

        if (debug_partial && pending_partial_trace_index > 0) {
            for (auto & prev : trace_steps) {
                if (prev.index == pending_partial_trace_index) {
                    prev.next_draft_origin = draft.empty() ? "fresh" : (draft_is_fresh ? "fresh" : "reused");
                    prev.next_draft_is_fresh = draft_is_fresh;
                    prev.next_draft_size = (int) draft.size();
                    prev.next_draft_tokens_json = token_vec_json(vocab_tgt, draft);
                    prev.extra_target_decode_reason = need_replay
                        ? (generated.empty() ? "target_replay_prefill_plus_validate" : "target_replay_suffix_or_validate")
                        : "validate_only";
                    prev.extra_draft_decode_reason = draft.empty() ? "fresh_draft_generation" : "reuse_residual_draft";
                    prev.next_validate_n_tok = (int) validate_tokens.size();
                    prev.prefill_count = session->debug_prefill_target_count + session->debug_prefill_target_suffix_count;
                    prev.validated_count = (int) validate_tokens.size();
                    break;
                }
            }
            pending_partial_trace_index = -1;
        }

        orbit_trace_step trace_step;
        trace_step.index = ++trace_step_index;
        trace_step.debug_enabled = debug_partial;
        trace_step.sampler_before = common_sampler_prev_str(smpl, ctx_tgt, 8);
        trace_step.sampler_before_hash = stable_hash_string(trace_step.sampler_before);
        trace_step.draft = draft;
        trace_step.sampled_id = -1;
        trace_step.rejected_id = -1;
        trace_step.validated_count = (int) validate_tokens.size();
        trace_step.checkpoint_total = session->last_checkpoint_count;
        trace_step.restore_total = session->last_restore_count;
        trace_step.id_last_before = (int) id_last;
        trace_step.n_past_before = n_past;
        trace_step.prompt_tgt_size_before = (int32_t) prompt_tgt.size();
        trace_step.prompt_tgt_pos_next_before = n_past;
        trace_step.kv_tgt_before_min = llama_memory_seq_pos_min(mem_tgt, 0);
        trace_step.kv_tgt_before_max = llama_memory_seq_pos_max(mem_tgt, 0);
        trace_step.kv_dft_before_min = llama_memory_seq_pos_min(mem_dft, 0);
        trace_step.kv_dft_before_max = llama_memory_seq_pos_max(mem_dft, 0);
        if (debug_partial) {
            trace_step.partial_state_before_json = partial_state_json(
                vocab_tgt,
                "before_partial_or_validate",
                prompt_tgt.size(),
                n_past,
                id_last,
                n_past,
                mem_tgt,
                mem_dft,
                have_ckpt,
                draft,
                draft_is_fresh,
                nullptr);
            trace_step.sampler_checkpoint_used = have_ckpt ? "checkpoint" : "none";
            trace_step.extra_target_decode_reason = need_replay
                ? (generated.empty() ? "target_replay_prefill_plus_validate" : "target_replay_suffix_or_validate")
                : "validate_only";
            trace_step.extra_draft_decode_reason = draft.empty() ? "fresh_draft_generation" : "reuse_residual_draft";
            trace_step.memory_clear_count = session->debug_memory_clear_count;
            trace_step.seq_rm_count = session->debug_seq_rm_count;
            trace_step.replay_count = session->debug_replay_count;
            trace_step.prefill_count = session->debug_prefill_target_count + session->debug_prefill_target_suffix_count;
            std::vector<int> validate_rows_preview(validate_tokens.size());
            for (size_t i = 0; i < validate_rows_preview.size(); ++i) {
                validate_rows_preview[i] = (int) i;
            }
            trace_step.pre_sample_state_json = pre_sample_state_json(
                vocab_tgt,
                prompt_tgt,
                n_past,
                id_last,
                draft,
                validate_tokens,
                validate_rows_preview,
                mem_tgt,
                mem_dft,
                trace_step.sampler_before,
                have_ckpt,
                have_ckpt ? ckpt.pos_max + 1 : n_past);
        }
        orbit_validate_trace validate_trace;
        validate_trace.step = trace_step.index;

        std::string replay_reason = "none";
        orbit_step_outcome step = resolve_validate_accept_restore(
            session,
            session->spec,
            ctx_tgt,
            session->ctx_dft,
            mem_tgt,
            mem_dft,
            smpl,
            ckpt,
            have_ckpt,
            prompt_tgt,
            draft,
            draft_is_fresh,
            n_past,
            validate_tokens,
            validate_pos0,
            boundary_committed_live,
            frontier_logical_base,
            vocab_tgt,
            &trace_step,
            debug_partial,
            &validate_trace,
            &target_decode_traces,
            trace_step.index,
            &replay_reason);
        session->debug_validate_decode_count++;
        if (step.resolution == orbit_step_resolution::error) {
            common_sampler_free(smpl);
            return false;
        }
        const std::vector<llama_token> & ids = step.ids;
        const int accepted = std::max(0, (int) ids.size() - 1);
        const bool full_accept = step.resolution == orbit_step_resolution::full_accept;
        const int rejected = std::max(0, (int) trace_step.draft.size() - accepted);
        trace_step.accepted_ids = ids;
        trace_step.accepted_draft = accepted;
        trace_step.fresh_draft_tokens_contrib = draft_is_fresh ? (int) trace_step.draft.size() : 0;
        trace_step.fresh_accepted_tokens_contrib = draft_is_fresh ? accepted : 0;
        trace_step.fresh_rejected_tokens_contrib = draft_is_fresh ? rejected : 0;
        trace_step.fresh_acceptance_ratio_contrib = trace_step.fresh_draft_tokens_contrib > 0
            ? (double) trace_step.fresh_accepted_tokens_contrib / (double) trace_step.fresh_draft_tokens_contrib
            : 0.0;
        trace_step.consumed_draft_tokens_contrib = (int) trace_step.draft.size();
        trace_step.consumed_accepted_tokens_contrib = accepted;
        trace_step.consumed_rejected_tokens_contrib = rejected;
        trace_step.consumed_acceptance_ratio_contrib = trace_step.consumed_draft_tokens_contrib > 0
            ? (double) trace_step.consumed_accepted_tokens_contrib / (double) trace_step.consumed_draft_tokens_contrib
            : 0.0;
        trace_step.sampler_after = common_sampler_prev_str(smpl, ctx_tgt, 8);
        trace_step.sampler_after_hash = stable_hash_string(trace_step.sampler_after);
        trace_step.sampled_id = ids.empty() ? -1 : (int) ids.back();
        trace_step.rejected_id = (accepted >= 0 && accepted < (int) trace_step.draft.size()) ? (int) trace_step.draft[(size_t) accepted] : -1;
        trace_step.id_last_after = ids.empty() ? (int) id_last : (int) ids.back();
        trace_step.validate_processed_by_spec = true;
        trace_step.validate_batch_prepare_ms = validate_trace.batch_prepare_ms;
        trace_step.validate_logits_rows_setup_ms = validate_trace.logits_rows_setup_ms;
        trace_step.validate_llama_decode_ms = validate_trace.llama_decode_validate_ms;
        trace_step.validate_post_decode_sample_ms = validate_trace.post_decode_sample_ms;
        trace_step.validate_batch_n_tokens = validate_trace.batch_n_tokens;
        trace_step.validate_batch_logits_count = validate_trace.batch_logits_count;
        trace_step.validate_n_outputs_requested = validate_trace.n_outputs_requested;
        if (full_accept) {
            session->last_full_accept_steps++;
            common_speculative_accept(session->spec, 0, (uint16_t) accepted);
            trace_step.resolution = "full_accept";
        } else if (step.resolution == orbit_step_resolution::live_partial) {
            frontier_trace_before_first_partial = false;
            trace_step.resolution = "live_partial";
            if (debug_partial) {
                pending_partial_trace_index = trace_step.index;
            }
        } else if (step.resolution == orbit_step_resolution::restored_partial) {
            frontier_trace_before_first_partial = false;
            trace_step.resolution = "partial_restore";
            trace_step.restore_total = session->last_restore_count;
            trace_step.kv_tgt_after_min = llama_memory_seq_pos_min(mem_tgt, 0);
            trace_step.kv_tgt_after_max = llama_memory_seq_pos_max(mem_tgt, 0);
            trace_step.kv_dft_after_min = llama_memory_seq_pos_min(mem_dft, 0);
            trace_step.kv_dft_after_max = llama_memory_seq_pos_max(mem_dft, 0);
            trace_step.n_past_after = n_past;
            trace_step.prompt_tgt_size_after = (int32_t) prompt_tgt.size();
            trace_step.prompt_tgt_pos_next_after = n_past;
            trace_step.residual_draft_size_after = (int32_t) draft.size();
            trace_step.residual_draft_after_json = token_vec_json(vocab_tgt, draft);
            if (debug_partial) {
                pending_partial_trace_index = trace_step.index;
            }
            validate_traces.push_back(validate_trace);
            trace_steps.push_back(trace_step);
            need_replay = false;
            phase_add(session->phase_loop_total, loop_phase_start);
            continue;
        } else {
            if (boundary_committed_live) {
                prompt_tgt.resize(frontier_logical_base);
                n_past = (int32_t) prompt_tgt.size();
            }
            trace_step.resolution = "replay_fallback";
            if (debug_partial) {
                trace_step.extra_target_decode_reason = replay_reason;
            }
        }

        for (size_t i = 0; i < ids.size() && (int) generated.size() < std::max(1, std::min(32, (int) max_tokens)); ++i) {
            if (!boundary_committed_live) {
                prompt_tgt.push_back(id_last);
            }
            id_last = ids[i];

            if (llama_vocab_is_eog(vocab_tgt, id_last)) {
                goto done;
            }

            generated.push_back(id_last);
            emit_output_token(id_last);
            if (frontier_trace_before_first_partial) {
                const std::vector<llama_token> tok = {id_last};
                emit_orbit_frontier_trace("advance", frontier_event_json(
                    vocab_tgt,
                    "accept_commit",
                    "validate_accept",
                    prompt_tgt,
                    n_past,
                    id_last,
                    draft,
                    mem_tgt,
                    mem_dft,
                    common_sampler_prev_str(smpl, ctx_tgt, 8),
                    &tok));
            }
        }

        if (full_accept) {
            n_past = (int32_t) prompt_tgt.size();
            {
                const auto phase_start = std::chrono::steady_clock::now();
                llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
                llama_memory_seq_rm(mem_dft, 0, n_past, -1);
                phase_add(session->phase_seq_rm, phase_start);
            }
            session->debug_seq_rm_count += 2;
            draft.clear();
            have_ckpt = false;
            draft_is_fresh = false;
            need_replay = false;
        } else if (step.resolution == orbit_step_resolution::live_partial) {
            n_past = (int32_t) prompt_tgt.size();
            draft.clear();
            have_ckpt = false;
            draft_is_fresh = true;
            need_replay = false;
        } else {
            need_replay = true;
        }
        trace_step.post_step_draft_is_fresh = draft_is_fresh;
        trace_step.post_step_need_replay = need_replay;
        trace_step.kv_tgt_after_min = llama_memory_seq_pos_min(mem_tgt, 0);
        trace_step.kv_tgt_after_max = llama_memory_seq_pos_max(mem_tgt, 0);
        trace_step.kv_dft_after_min = llama_memory_seq_pos_min(mem_dft, 0);
        trace_step.kv_dft_after_max = llama_memory_seq_pos_max(mem_dft, 0);
        trace_step.n_past_after = n_past;
        trace_step.prompt_tgt_size_after = (int32_t) prompt_tgt.size();
        trace_step.prompt_tgt_pos_next_after = n_past;
        trace_step.residual_draft_size_after = (int32_t) draft.size();
        trace_step.residual_draft_after_json = token_vec_json(vocab_tgt, draft);
        validate_traces.push_back(validate_trace);
        trace_steps.push_back(trace_step);
        phase_add(session->phase_loop_total, loop_phase_start);
    }

done:
    common_sampler_free(smpl);
    session->cached_prompt_tokens = prompt_tgt;
    session->last_raw_emitted_token_ids_json = token_id_vec_json(generated);
    {
        std::vector<llama_token> end_turn_frontier = prompt_tgt;
        end_turn_frontier.insert(end_turn_frontier.end(), generated.begin(), generated.end());
        session->committed_frontier_tokens = end_turn_frontier;
        session->last_end_turn_frontier_token_ids_json = token_id_vec_json(end_turn_frontier);
    }
    if (chat_reuse_debug_enabled()) {
        std::vector<llama_token> first_emitted = generated;
        if (first_emitted.size() > 20) {
            first_emitted.resize(20);
        }
        std::ostringstream out;
        out
            << "{"
            << "\"first_20_emitted_raw_ids\":" << token_vec_json(vocab_tgt, first_emitted)
            << ",\"raw_last_content\":\"" << json_escape(session->last_content) << "\""
            << ",\"cached_prompt_tokens_size_after\":" << session->cached_prompt_tokens.size()
            << ",\"cached_prompt_tokens_tail_after\":" << token_vec_json(vocab_tgt, tail_tokens(session->cached_prompt_tokens, 48))
            << ",\"request_boundary_ckpt_after\":" << checkpoint_summary_json(session->request_boundary_ckpt)
            << ",\"request_boundary_prompt_tgt_size_after\":" << session->request_boundary_prompt_tgt.size()
            << ",\"request_boundary_prompt_tgt_tail_after\":" << token_vec_json(vocab_tgt, tail_tokens(session->request_boundary_prompt_tgt, 48))
            << ",\"ctx_tgt_max_pos_after\":" << llama_memory_seq_pos_max(mem_tgt, 0)
            << ",\"ctx_dft_max_pos_after\":" << llama_memory_seq_pos_max(mem_dft, 0)
            << ",\"output_tokens\":" << session->last_output_tokens
            << "}";
        emit_orbit_chat_reuse_trace("done", out.str());
    }
    session->last_fresh_acceptance_ratio = session->last_draft_tokens_total > 0
        ? (double) session->last_accepted_tokens_total / (double) session->last_draft_tokens_total
        : 0.0;
    const int consumed_draft_tokens_total = session->last_draft_tokens_total + session->last_reused_draft_tokens_total;
    const int consumed_accepted_tokens_total = session->last_accepted_tokens_total + session->last_reused_accepted_tokens_total;
    session->last_consumed_acceptance_ratio = consumed_draft_tokens_total > 0
        ? (double) consumed_accepted_tokens_total / (double) consumed_draft_tokens_total
        : 0.0;
    session->last_acceptance_ratio = session->last_fresh_acceptance_ratio;
    session->last_elapsed_ms = elapsed_s(t0) * 1000.0;
    session->last_tokens_per_second = session->last_elapsed_ms > 0.0
        ? ((double) session->last_output_tokens / session->last_elapsed_ms) * 1000.0
        : 0.0;
    {
        std::ostringstream out;
        out << "[";
        for (size_t i = 0; i < trace_steps.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << trace_step_json(vocab_tgt, trace_steps[i]);
        }
        out << "]";
        session->last_trace_json = out.str();
    }
    session->last_validate_trace_json = validate_trace_json(validate_traces);
    session->last_target_decode_trace_json = target_decode_trace_json(vocab_tgt, target_decode_traces);
    {
        std::ostringstream out;
        out
            << "{"
            << "\"prompt_prefix_restore\":" << phase_json(session->phase_prefix_restore) << ","
            << "\"suffix_decode_target\":" << phase_json(session->phase_suffix_decode_target) << ","
            << "\"draft_generation\":" << phase_json(session->phase_draft_generation) << ","
            << "\"target_validate\":" << phase_json(session->phase_target_validate) << ","
            << "\"speculative_process\":" << phase_json(session->phase_speculative_process) << ","
            << "\"sampler_clone\":" << phase_json(session->phase_sampler_clone) << ","
            << "\"sampler_restore\":" << phase_json(session->phase_sampler_restore) << ","
            << "\"sampler_ops\":" << phase_json(session->phase_sampler_ops) << ","
            << "\"seq_rm\":" << phase_json(session->phase_seq_rm) << ","
            << "\"batch_build\":" << phase_json(session->phase_batch_build) << ","
            << "\"ctx_tgt_checkpoint\":" << phase_json(session->phase_ctx_tgt_checkpoint) << ","
            << "\"ctx_tgt_restore\":" << phase_json(session->phase_ctx_tgt_restore) << ","
            << "\"ctx_dft_checkpoint\":" << phase_json(session->phase_ctx_dft_checkpoint) << ","
            << "\"ctx_dft_restore\":" << phase_json(session->phase_ctx_dft_restore) << ","
            << "\"rollback_replay\":" << phase_json(session->phase_rollback_replay) << ","
            << "\"detokenize_output_bridge\":" << phase_json(session->phase_detokenize_bridge) << ","
            << "\"speculative_loop_total\":" << phase_json(session->phase_loop_total);
        if (debug_partial) {
            out
                << ",\"partial_debug\":{"
                << "\"memory_clear_count\":" << session->debug_memory_clear_count << ","
                << "\"seq_rm_count\":" << session->debug_seq_rm_count << ","
                << "\"replay_count\":" << session->debug_replay_count << ","
                << "\"prefill_target_count\":" << session->debug_prefill_target_count << ","
                << "\"prefill_target_suffix_count\":" << session->debug_prefill_target_suffix_count << ","
                << "\"validate_decode_count\":" << session->debug_validate_decode_count << ","
                << "\"draft_decode_count\":" << session->debug_draft_decode_count
                << "}";
        }
        out
            << "}";
        session->last_timing_json = out.str();
    }
    return true;
}

extern "C" const char * orbit_mtp_session_last_content(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_content.c_str() : "";
}

extern "C" int32_t orbit_mtp_session_last_output_tokens(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_output_tokens : 0;
}

extern "C" const char * orbit_mtp_session_last_raw_emitted_token_ids(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_raw_emitted_token_ids_json.c_str() : "[]";
}

extern "C" const char * orbit_mtp_session_last_end_turn_frontier_token_ids(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_end_turn_frontier_token_ids_json.c_str() : "[]";
}

extern "C" int32_t orbit_mtp_session_last_draft_tokens_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_draft_tokens_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_accepted_tokens_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_accepted_tokens_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_rejected_tokens_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_rejected_tokens_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_reused_draft_tokens_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_reused_draft_tokens_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_reused_accepted_tokens_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_reused_accepted_tokens_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_reused_rejected_tokens_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_reused_rejected_tokens_total : 0;
}

extern "C" double orbit_mtp_session_last_acceptance_ratio(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_acceptance_ratio : 0.0;
}

extern "C" double orbit_mtp_session_last_fresh_acceptance_ratio(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_fresh_acceptance_ratio : 0.0;
}

extern "C" double orbit_mtp_session_last_consumed_acceptance_ratio(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_consumed_acceptance_ratio : 0.0;
}

extern "C" int32_t orbit_mtp_session_last_target_decode_calls(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_target_decode_calls : 0;
}

extern "C" int32_t orbit_mtp_session_last_draft_decode_calls(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_draft_decode_calls : 0;
}

extern "C" double orbit_mtp_session_last_elapsed_ms(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_elapsed_ms : 0.0;
}

extern "C" double orbit_mtp_session_last_tokens_per_second(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_tokens_per_second : 0.0;
}

extern "C" int32_t orbit_mtp_session_last_full_accept_steps(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_full_accept_steps : 0;
}

extern "C" int32_t orbit_mtp_session_last_replay_steps(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_replay_steps : 0;
}

extern "C" int32_t orbit_mtp_session_last_partial_accept_steps(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_partial_accept_steps : 0;
}

extern "C" int32_t orbit_mtp_session_last_partial_no_replay_steps(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_partial_no_replay_steps : 0;
}

extern "C" int32_t orbit_mtp_session_last_replay_fallback_steps(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_replay_fallback_steps : 0;
}

extern "C" bool orbit_mtp_session_last_seq_rm_supported(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_seq_rm_supported : false;
}

extern "C" int32_t orbit_mtp_session_last_rollback_tokens_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_rollback_tokens_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_checkpoint_count(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_checkpoint_count : 0;
}

extern "C" int32_t orbit_mtp_session_last_restore_count(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_restore_count : 0;
}

extern "C" const char * orbit_mtp_session_last_trace_json(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_trace_json.c_str() : "[]";
}

extern "C" const char * orbit_mtp_session_last_timing_json(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_timing_json.c_str() : "{}";
}

extern "C" const char * orbit_mtp_session_last_validate_trace_json(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_validate_trace_json.c_str() : "[]";
}

extern "C" const char * orbit_mtp_session_last_target_decode_trace_json(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_target_decode_trace_json.c_str() : "[]";
}
