#include "llama.h"
// `llama_get_ctx_other` reports whether a draft context physically shares its
// memory with the target. It lives in the internal extension header rather than
// the public one; build_support already puts <llama_root>/src on the include
// path, so this needs no build change.
#include "llama-ext.h"
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

static bool mtp_trace_enabled() {
    const char * value = std::getenv("ORBIT_MTP_TRACE");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static bool boundary_split_enabled() {
    const char * value = std::getenv("ORBIT_MTP_BOUNDARY_SPLIT");
    if (!value || !value[0]) {
        return true;
    }
    return std::strcmp(value, "0") != 0;
}

// Target-side event trace, for qualifying persistent target-KV behaviour.
// Deliberately env-gated and stderr-only, exactly like the emitters above: it
// exports no symbol, so the base and self-MTP ABI contracts are untouched and a
// packaged shim stays interchangeable. Every call site is additionally guarded
// by `target_trace_enabled()` so a disabled trace costs one boolean test and
// never builds a payload.
//
// Only mem_tgt operations are recorded. The legacy debug counters cannot serve
// this purpose: they mix target with draft, count decode batches rather than
// tokens, and ignore seq_rm results.
static bool target_trace_enabled() {
    const char * value = std::getenv("ORBIT_MTP_TARGET_TRACE");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static void emit_orbit_target_trace(const std::string & payload) {
    std::fprintf(stderr, "ORBIT_MTP_TARGET %s\n", payload.c_str());
}

static void trace_target_clear(const char * site) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "TARGET_CLEAR site=" << (site ? site : "?");
    emit_orbit_target_trace(out.str());
}

static void trace_target_prefill(const char * site, int32_t first_pos, size_t n_tokens) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "TARGET_PREFILL site=" << (site ? site : "?")
        << " first_pos=" << first_pos
        << " n_tokens=" << n_tokens;
    emit_orbit_target_trace(out.str());
}

static void trace_target_seq_rm(const char * site, int32_t p0, int32_t p1, bool result) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "TARGET_SEQ_RM site=" << (site ? site : "?")
        << " p0=" << p0 << " p1=" << p1
        << " result=" << (result ? "true" : "false");
    emit_orbit_target_trace(out.str());
}

static void trace_target_frontier(const char * label, llama_memory_t mem) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "TARGET_FRONTIER label=" << (label ? label : "?")
        << " pos_max=" << (mem ? (long long) llama_memory_seq_pos_max(mem, 0) : -1);
    emit_orbit_target_trace(out.str());
}

// Phase-B draft observability. Deliberately separate helpers rather than
// parameterising the target ones above: those bytes are frozen evidence, and a
// recorder that can mislabel a draft event as a target event is worse than no
// recorder. Same env gate, same stderr sink, same "no payload when disabled"
// discipline. The wire prefix is ORBIT_MTP_OBSDRAFT, chosen so it is neither a
// prefix nor an extension of any existing ORBIT_MTP_* stream (DRAFT, DFT,
// TARGET, FRONTIER, VALIDATE). A prefix-extension such as ORBIT_MTP_DRAFTOBS
// would let `startswith("ORBIT_MTP_DRAFT")` swallow these records into that
// stream's JSON parser and silently lose them from a run that looked fine.
//
// The speculative implementation owns `pending_h`, the predecessor hidden-state
// row that seeds every proposal. Its CONTENT is now directly observable through
// `common_speculative_pending_state`, so SPEC_STATE carries both the impl's
// lifetime (epoch/pointer) and the row's provenance, fingerprint and write
// count. Destroying the impl still re-zeroes `pending_h` by construction, but
// that no longer has to be deduced -- the witness reports it.
static void emit_orbit_draftobs_trace(const std::string & payload) {
    std::fprintf(stderr, "ORBIT_MTP_OBSDRAFT %s\n", payload.c_str());
}

static void trace_draft_frontier(const char * label, llama_memory_t mem) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "DRAFT_FRONTIER label=" << (label ? label : "?")
        << " pos_max=" << (mem ? (long long) llama_memory_seq_pos_max(mem, 0) : -1);
    emit_orbit_draftobs_trace(out.str());
}

static void trace_resident_admission(
        int32_t claim,
        bool admitted,
        bool pair_trusted,
        bool identity_ok,
        int32_t pending_pos,
        unsigned long long pending_gen) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "RESIDENT_ADMISSION claim=" << claim
        << " admitted=" << (admitted ? "true" : "false")
        << " pair_trusted=" << (pair_trusted ? "true" : "false")
        << " identity_ok=" << (identity_ok ? "true" : "false")
        << " pending_pos=" << pending_pos
        << " pending_gen=" << pending_gen;
    emit_orbit_draftobs_trace(out.str());
}

static void trace_pair_trust(
        const char * verdict,
        int32_t frontier,
        bool target_ok,
        bool draft_ok,
        bool pending_aligned,
        bool identity_ok) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "PAIR_TRUST verdict=" << (verdict ? verdict : "?")
        << " frontier=" << frontier
        << " target_ok=" << (target_ok ? "true" : "false")
        << " draft_ok=" << (draft_ok ? "true" : "false")
        << " pending_aligned=" << (pending_aligned ? "true" : "false")
        << " identity_ok=" << (identity_ok ? "true" : "false");
    emit_orbit_draftobs_trace(out.str());
}

static void trace_pending_discarded(const char * site) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "PENDING_DISCARDED site=" << (site ? site : "?");
    emit_orbit_draftobs_trace(out.str());
}

static void trace_request_reset_mode(const char * mode) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "REQUEST_RESET_MODE mode=" << (mode ? mode : "?");
    emit_orbit_draftobs_trace(out.str());
}

static void trace_draft_clear(const char * site) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "DRAFT_CLEAR site=" << (site ? site : "?");
    emit_orbit_draftobs_trace(out.str());
}

static void trace_draft_process(const char * site, int32_t first_pos, size_t n_tokens) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "DRAFT_PROCESS site=" << (site ? site : "?")
        << " first_pos=" << first_pos
        << " n_tokens=" << n_tokens;
    emit_orbit_draftobs_trace(out.str());
}

static void trace_draft_seq_rm(const char * site, int32_t p0, int32_t p1, bool result) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "DRAFT_SEQ_RM site=" << (site ? site : "?")
        << " p0=" << p0 << " p1=" << p1
        << " result=" << (result ? "true" : "false");
    emit_orbit_draftobs_trace(out.str());
}

// `epoch` is authoritative: it counts speculative-implementation constructions
// performed by this shim, so a preserved impl keeps its epoch and a recreated one
// increments. `spec` is corroborating only -- an allocator may hand back the same
// address, so an unchanged pointer proves nothing on its own. `ctx_dft` is a
// different object from the impl and is NOT freed by reset; recording it shows
// which half of the pair changed.
static void trace_spec_state(
        const char * label,
        unsigned long long epoch,
        const void * spec,
        const void * ctx_dft) {
    if (!target_trace_enabled()) { return; }
    std::ostringstream out;
    out << "SPEC_STATE label=" << (label ? label : "?")
        << " epoch=" << epoch
        << " spec=" << spec
        << " ctx_dft=" << ctx_dft;
    // The epoch is a LIFETIME witness only. `pending_h` is the row that seeds
    // every draft proposal, and its CONTENT state is what decides whether a
    // following suffix has the predecessor it needs; an unchanged epoch says the
    // object survived, never that the row is aligned. These three come from the
    // read-only diagnostic accessor and are recorded here so a trace can answer
    // that question directly.
    int32_t pend_pos = -1;
    uint64_t pend_fp = 0;
    uint64_t pend_gen = 0;
    const bool pend_ok = spec != nullptr && common_speculative_pending_state(
        static_cast<const common_speculative *>(spec), 0,
        &pend_pos, &pend_fp, &pend_gen);
    out << " pending_ok=" << (pend_ok ? "true" : "false")
        << " pending_pos=" << pend_pos
        << " pending_fp=" << pend_fp
        << " pending_gen=" << pend_gen;
    emit_orbit_draftobs_trace(out.str());
}

static bool draft_trace_enabled() {
    const char * value = std::getenv("ORBIT_MTP_DRAFT_TRACE");
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

static uint64_t stable_hash_string(const std::string & value) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char c : value) {
        hash ^= (uint64_t) c;
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t stable_hash_tokens(const std::vector<llama_token> & tokens) {
    uint64_t hash = 1469598103934665603ull;
    for (llama_token token : tokens) {
        const uint32_t value = (uint32_t) token;
        for (int i = 0; i < 4; ++i) {
            hash ^= (uint64_t) ((value >> (i * 8)) & 0xffu);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

static uint64_t stable_hash_token(llama_token token) {
    const std::vector<llama_token> tokens = {token};
    return stable_hash_tokens(tokens);
}

static uint64_t stable_hash_logits(llama_context * ctx, const llama_vocab * vocab) {
    if (!ctx || !vocab) {
        return 0;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        return 0;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        return 0;
    }
    const auto * bytes = reinterpret_cast<const unsigned char *>(logits);
    uint64_t hash = 1469598103934665603ull;
    const size_t size = (size_t) n_vocab * sizeof(float);
    for (size_t i = 0; i < size; ++i) {
        hash ^= (uint64_t) bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t stable_hash_frontier(int32_t min_pos, int32_t max_pos) {
    uint64_t hash = 1469598103934665603ull;
    const int32_t values[2] = {min_pos, max_pos};
    for (int32_t pos : values) {
        const uint32_t value = (uint32_t) pos;
        for (int i = 0; i < 4; ++i) {
            hash ^= (uint64_t) ((value >> (i * 8)) & 0xffu);
            hash *= 1099511628211ull;
        }
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
    // Whether this session allocated model_dft and must free it. False for the
    // single-GGUF self-MTP path, where the draft head lives inside the model
    // the caller already loaded and still owns: freeing it here would be a
    // double free the moment the caller tears its own client down.
    bool owns_model_dft = true;
    // Borrowed in both modes. Recorded only so teardown can be audited: this
    // session never frees it, and the client that created it frees it after
    // this session is destroyed.
    llama_context * ctx_tgt_borrowed = nullptr;
    llama_context * ctx_dft = nullptr;
    common_speculative * spec = nullptr;
    common_params_speculative spec_params;
    common_prompt_checkpoint request_boundary_ckpt;
    std::vector<llama_token> request_boundary_prompt_tgt;
    // A resident-prefix claim supplied by the runtime for ONE completion. The
    // runtime has already proven token identity against its committed
    // sequence; this side only ever checks that physical target memory agrees.
    // Consumed (zeroed) at the start of each completion so a stale claim can
    // never leak into a later request.
    int32_t pending_resident_prefix_len = 0;
    // The target context the speculative implementation was CONSTRUCTED with.
    // `common_speculative_impl` copies ctx_tgt into its params and derives
    // is_mem_shared from that exact pair, so a preserved implementation keeps
    // whatever target it was built against. Soft reuse must therefore prove the
    // caller is still presenting the same context; production creates ctx_tgt
    // once per model load, but that is caller behaviour, not an invariant of
    // this function, and relying on caller ordering is how D3b-R defects were
    // written.
    llama_context * spec_pinned_ctx_tgt = nullptr;
    // One trust bit for the WHOLE pair: target KV, draft KV and pending_h. The
    // halves advance in lockstep and are only ever consumed together, so a
    // half-trusted state is not reachable and would only be a way to salvage one
    // side of a mismatched pair. Any uncertain mutation on either half, or an
    // identity change, poisons the pair until a hard rebuild re-proves it.
    bool persistent_pair_untrusted = true;
    // Diagnostic only: which reset path the last completion entry took.
    bool last_request_reset_was_soft = false;
    // Counts speculative-implementation constructions performed by this shim.
    // Diagnostic only: never read by any decision, only reported by the draft
    // recorder so a trace can distinguish a preserved impl (epoch unchanged)
    // from a destroyed-and-recreated one (epoch incremented). The object holding
    // `pending_h` is not otherwise observable from here.
    unsigned long long spec_epoch = 0;
    // Whether the last completion actually reused a resident prefix. Set from
    // the same predicate that drives the replay decision, so it cannot report
    // reuse that did not happen.
    bool last_resident_reuse_active = false;
    // Set when a target seq_rm was refused, so the resident prefix cannot be
    // proven canonical. A refused removal leaves rejected speculative tokens
    // resident above the committed frontier; a later completion claiming
    // frontier+1 would then "match" a poisoned frontier. Correct target state
    // is mandatory, cache reuse is not.
    bool last_target_untrusted = false;
    long rss_before_kb = -1;
    long rss_after_init_kb = -1;
    long rss_peak_kb = -1;
    std::vector<llama_token> cached_prompt_tokens;
    std::string last_content;
    // The exact token ids this completion decoded into the target, in order.
    // The runtime needs these to publish a token-level committed identity for
    // the next turn; retokenizing `last_content` is not equivalent, because a
    // round trip through text is not guaranteed to reproduce the ids that were
    // actually made resident, and an identity that misdescribes KV is a false
    // cache hit rather than a slow path.
    std::vector<llama_token> last_generated_tokens;
    // The token sequence physically resident in the target KV at the end of the
    // last completion. This is NOT `prompt + last_generated_tokens`: a sampled
    // token enters `generated` at sample time but only enters `prompt_tgt` on
    // the following iteration, so the two differ by the final token. A runtime
    // identity built from `generated` would claim one more resident token than
    // exists, and a claim that overstates KV is a false cache hit -- the next
    // prompt would be decoded one position early. Publishing the loop's own
    // `prompt_tgt` keeps identity physically derived rather than reconstructed.
    std::vector<llama_token> last_resident_tokens;
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
    int last_validate_steps = 0;
    int last_rows_requested_total = 0;
    int last_rows_consumed_estimated_total = 0;
    int last_rows_wasted_estimated_total = 0;
    int last_accepted_draft_hist_0 = 0;
    int last_accepted_draft_hist_1 = 0;
    int last_accepted_draft_hist_2 = 0;
    int last_accepted_draft_hist_3 = 0;
    int last_accepted_draft_hist_ge4 = 0;
    std::string last_trace_json;
    std::string last_timing_json;
    std::string last_validate_trace_json;
    std::string last_target_decode_trace_json;
    std::string last_validate_equivalence_json;
    std::vector<uint64_t> last_output_token_hashes;
    std::string last_output_token_hashes_json;
    std::string last_first_sample_trace_json;
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
    int rejected_draft = 0;
    int sampled_id = -1;
    int rejected_id = -1;
    std::string resolution;
    int validated_count = 0;
    std::string draft_origin = "unknown";
    bool draft_is_fresh = false;
    bool need_replay_before = false;
    int validate_n_tok = 0;
    int32_t validate_pos0 = -1;
    int32_t old_n_past = -1;
    int32_t new_n_past = -1;
    int32_t prompt_tgt_len = -1;
    int32_t prompt_dft_len = -1;
    uint64_t prompt_tgt_hash = 0;
    uint64_t ctx_tgt_frontier_hash = 0;
    uint64_t ctx_dft_frontier_hash = 0;
    uint64_t sampler_state_hash_before = 0;
    uint64_t sampler_state_hash_after = 0;
    int remaining_generation_cap = 0;
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
    (void) vocab;
    out << "{\"count\":" << tokens.size() << ",\"hash\":" << stable_hash_tokens(tokens) << "}";
    return out.str();
}

static std::string optional_rejected_json(const llama_vocab * vocab, const std::vector<llama_token> & draft, int accepted_draft) {
    (void) vocab;
    if (accepted_draft < 0 || accepted_draft >= (int) draft.size()) {
        return "null";
    }
    std::ostringstream out;
    const auto tok = draft[(size_t) accepted_draft];
    out << "{\"hash\":" << stable_hash_token(tok) << "}";
    return out.str();
}

static std::string trace_step_json(const llama_vocab * vocab, const orbit_trace_step & step) {
    std::ostringstream out;
    out
        << "{\"index\":" << step.index
        << ",\"sampler_before_hash\":" << step.sampler_before_hash
        << ",\"sampler_after_hash\":" << step.sampler_after_hash
        << ",\"draft_count\":" << step.draft.size()
        << ",\"draft_hash\":" << stable_hash_tokens(step.draft)
        << ",\"accepted_id_count\":" << step.accepted_ids.size()
        << ",\"accepted_hash\":" << stable_hash_tokens(step.accepted_ids)
        << ",\"accepted_draft\":" << step.accepted_draft
        << ",\"rejected_draft\":" << step.rejected_draft
        << ",\"sampled_hash\":" << (step.sampled_id >= 0 ? stable_hash_token((llama_token) step.sampled_id) : 0)
        << ",\"rejected_hash\":" << (step.rejected_id >= 0 ? stable_hash_token((llama_token) step.rejected_id) : 0)
        << ",\"first_rejected\":" << optional_rejected_json(vocab, step.draft, step.accepted_draft)
        << ",\"resolution\":\"" << step.resolution << "\""
        << ",\"draft_origin\":\"" << json_escape(step.draft_origin) << "\""
        << ",\"draft_is_fresh\":" << (step.draft_is_fresh ? "true" : "false")
        << ",\"need_replay_before\":" << (step.need_replay_before ? "true" : "false")
        << ",\"validate_n_tok\":" << step.validate_n_tok
        << ",\"validate_pos0\":" << step.validate_pos0
        << ",\"old_n_past\":" << step.old_n_past
        << ",\"new_n_past\":" << step.new_n_past
        << ",\"prompt_tgt_len\":" << step.prompt_tgt_len
        << ",\"prompt_dft_len\":" << step.prompt_dft_len
        << ",\"prompt_tgt_hash\":" << step.prompt_tgt_hash
        << ",\"ctx_tgt_frontier_hash\":" << step.ctx_tgt_frontier_hash
        << ",\"ctx_dft_frontier_hash\":" << step.ctx_dft_frontier_hash
        << ",\"sampler_state_hash_before\":" << step.sampler_state_hash_before
        << ",\"sampler_state_hash_after\":" << step.sampler_state_hash_after
        << ",\"remaining_generation_cap\":" << step.remaining_generation_cap
        << ",\"id_last_before_hash\":" << stable_hash_token((llama_token) step.id_last_before)
        << ",\"id_last_after_hash\":" << stable_hash_token((llama_token) step.id_last_after)
        << ",\"n_past_before\":" << step.n_past_before
        << ",\"n_past_after\":" << step.n_past_after
        << ",\"prompt_tgt_size_before\":" << step.prompt_tgt_size_before
        << ",\"prompt_tgt_size_after\":" << step.prompt_tgt_size_after
        << ",\"prompt_tgt_pos_next_before\":" << step.prompt_tgt_pos_next_before
        << ",\"prompt_tgt_pos_next_after\":" << step.prompt_tgt_pos_next_after
        << ",\"residual_draft_size_after\":" << step.residual_draft_size_after
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
            << ",\"prefill_count\":" << step.prefill_count;
    }
    out << "}";
    return out.str();
}

static std::string validate_equivalence_json(
    const std::vector<orbit_trace_step> & steps,
    int rows_requested_total,
    int rows_consumed_estimated_total,
    int rows_wasted_estimated_total
) {
    static constexpr size_t max_step_sample = 64;
    int hist_0 = 0;
    int hist_1 = 0;
    int hist_2 = 0;
    int hist_3 = 0;
    int hist_ge4 = 0;
    bool all_steps_have_frontier = !steps.empty();
    bool all_steps_have_sampler_hash = !steps.empty();
    for (const auto & step : steps) {
        if (step.accepted_draft <= 0) {
            hist_0++;
        } else if (step.accepted_draft == 1) {
            hist_1++;
        } else if (step.accepted_draft == 2) {
            hist_2++;
        } else if (step.accepted_draft == 3) {
            hist_3++;
        } else {
            hist_ge4++;
        }
        all_steps_have_frontier = all_steps_have_frontier &&
            step.kv_tgt_before_min >= 0 &&
            step.kv_tgt_before_max >= step.kv_tgt_before_min &&
            step.kv_tgt_after_min >= 0 &&
            step.kv_tgt_after_max >= step.kv_tgt_after_min;
        all_steps_have_sampler_hash = all_steps_have_sampler_hash &&
            step.sampler_state_hash_before != 0 &&
            step.sampler_state_hash_after != 0;
    }
    const double rows_wasted_estimated_ratio = rows_requested_total > 0
        ? (double) rows_wasted_estimated_total / (double) rows_requested_total
        : 0.0;

    std::ostringstream out;
    out
        << "{"
        << "\"steps\":" << steps.size()
        << ",\"steps_recorded\":" << std::min(steps.size(), max_step_sample)
        << ",\"rows_requested_total\":" << rows_requested_total
        << ",\"rows_consumed_estimated_total\":" << rows_consumed_estimated_total
        << ",\"rows_wasted_estimated_total\":" << rows_wasted_estimated_total
        << ",\"rows_wasted_estimated_ratio\":" << rows_wasted_estimated_ratio
        << ",\"accepted_draft_histogram\":{"
        << "\"0\":" << hist_0
        << ",\"1\":" << hist_1
        << ",\"2\":" << hist_2
        << ",\"3\":" << hist_3
        << ",\"ge4\":" << hist_ge4
        << "}"
        << ",\"all_steps_have_frontier\":" << (steps.empty() ? "null" : (all_steps_have_frontier ? "true" : "false"))
        << ",\"all_steps_have_sampler_hash\":" << (steps.empty() ? "null" : (all_steps_have_sampler_hash ? "true" : "false"))
        << ",\"step_sample\":[";
    const size_t n_sample = std::min(steps.size(), max_step_sample);
    for (size_t i = 0; i < n_sample; ++i) {
        const auto & step = steps[i];
        const int draft_size = (int) step.draft.size();
        const int rows_requested = draft_size + 1;
        const bool full_accept = step.resolution == "full_accept";
        const int rows_consumed_estimated = full_accept ? rows_requested : std::min(rows_requested, step.accepted_draft + 1);
        const int rows_wasted_estimated = std::max(0, rows_requested - rows_consumed_estimated);
        if (i > 0) {
            out << ",";
        }
        out
            << "{"
            << "\"step\":" << step.index
            << ",\"draft_size\":" << draft_size
            << ",\"accepted_draft\":" << step.accepted_draft
            << ",\"resolution\":\"" << json_escape(step.resolution) << "\""
            << ",\"rows_requested\":" << rows_requested
            << ",\"rows_consumed_estimated\":" << rows_consumed_estimated
            << ",\"rows_wasted_estimated\":" << rows_wasted_estimated
            << ",\"sampler_before_hash\":" << (step.sampler_state_hash_before ? std::to_string(step.sampler_state_hash_before) : "null")
            << ",\"sampler_after_hash\":" << (step.sampler_state_hash_after ? std::to_string(step.sampler_state_hash_after) : "null");
        if (step.kv_tgt_before_min >= 0 && step.kv_tgt_before_max >= step.kv_tgt_before_min) {
            out
                << ",\"frontier_before\":{\"min\":" << step.kv_tgt_before_min
                << ",\"max\":" << step.kv_tgt_before_max
                << ",\"hash\":" << stable_hash_frontier(step.kv_tgt_before_min, step.kv_tgt_before_max)
                << "}";
        } else {
            out << ",\"frontier_before\":null";
        }
        if (step.kv_tgt_after_min >= 0 && step.kv_tgt_after_max >= step.kv_tgt_after_min) {
            out
                << ",\"frontier_after\":{\"min\":" << step.kv_tgt_after_min
                << ",\"max\":" << step.kv_tgt_after_max
                << ",\"hash\":" << stable_hash_frontier(step.kv_tgt_after_min, step.kv_tgt_after_max)
                << "}";
        } else {
            out << ",\"frontier_after\":null";
        }
        out << "}";
    }
    out << "]}";
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

static std::string uint64_vec_json(const std::vector<uint64_t> & values) {
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

static std::string first_sample_trace_json(
    const char * path_name,
    const std::vector<llama_token> & prompt_tgt,
    llama_context * ctx_tgt,
    const llama_vocab * vocab_tgt,
    llama_memory_t mem_tgt,
    int32_t n_past,
    int last_prefill_batch_n_tokens,
    int logits_row_count,
    int n_outputs,
    const std::string & sampler_before,
    llama_token first_sample,
    const std::string & sampler_after,
    int generated_count_after,
    bool request_boundary_restore_used,
    bool request_boundary_logits_refreshed
) {
    const int32_t min_before = mem_tgt ? llama_memory_seq_pos_min(mem_tgt, 0) : -1;
    const int32_t max_before = mem_tgt ? llama_memory_seq_pos_max(mem_tgt, 0) : -1;
    std::ostringstream out;
    out
        << "{\"path_name\":\"" << json_escape(path_name ? path_name : "mtp") << "\""
        << ",\"prompt_hash\":" << stable_hash_tokens(prompt_tgt)
        << ",\"prompt_count\":" << prompt_tgt.size()
        << ",\"ctx_n_past\":" << n_past
        << ",\"ctx_frontier_min\":" << min_before
        << ",\"ctx_frontier_max\":" << max_before
        << ",\"ctx_frontier_hash\":" << stable_hash_frontier(min_before, max_before)
        << ",\"ctx_max_pos\":" << max_before
        << ",\"batch_n_tokens\":" << last_prefill_batch_n_tokens
        << ",\"logits_row_count\":" << logits_row_count
        << ",\"n_outputs\":" << n_outputs
        << ",\"last_logits_hash\":" << stable_hash_logits(ctx_tgt, vocab_tgt)
        << ",\"sampler_config_hash\":" << stable_hash_string("common_sampler:top_k=1:temp=0:top_p=1:min_p=0:repeat=1")
        << ",\"sampler_state_hash\":" << stable_hash_string(sampler_before)
        << ",\"sampler_chain_type\":\"common_sampler(top_k,temp)\""
        << ",\"seed_hash\":null"
        << ",\"temperature\":0"
        << ",\"top_k\":1"
        << ",\"top_p\":1"
        << ",\"min_p\":0"
        << ",\"repeat_penalty\":1"
        << ",\"generated_count\":0"
        << ",\"first_sample_hash\":" << stable_hash_token(first_sample)
        << ",\"sampler_state_hash_after\":" << stable_hash_string(sampler_after)
        << ",\"ctx_frontier_min_after\":" << (mem_tgt ? llama_memory_seq_pos_min(mem_tgt, 0) : -1)
        << ",\"ctx_frontier_max_after\":" << (mem_tgt ? llama_memory_seq_pos_max(mem_tgt, 0) : -1)
        << ",\"ctx_frontier_hash_after\":" << stable_hash_frontier(
            mem_tgt ? llama_memory_seq_pos_min(mem_tgt, 0) : -1,
            mem_tgt ? llama_memory_seq_pos_max(mem_tgt, 0) : -1)
        << ",\"generated_count_after\":" << generated_count_after
        << ",\"request_boundary_restore_used\":" << (request_boundary_restore_used ? "true" : "false")
        << ",\"request_boundary_logits_refreshed\":" << (request_boundary_logits_refreshed ? "true" : "false")
        << "}";
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
            << ",\"logits_flag_count\":" << item.logits_flags.size()
            << ",\"token_count\":" << item.token_ids.size()
            << ",\"token_hash\":" << stable_hash_tokens(item.token_ids)
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
        // The per-step draft advance. This, not the prefill, is what moves the
        // draft frontier during generation and updates `pending_h` on every
        // step; leaving it untraced would make the exit frontier unexplainable
        // from the event stream. Emitted before the timing bracket opens.
        trace_draft_process("validate", validate_pos0, validate_tokens.size());
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
                const bool tgt_rm_ok = llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
                trace_target_seq_rm("live_partial", n_past, -1, tgt_rm_ok);
                if (!tgt_rm_ok) {
                    session->last_target_untrusted = true;
                }
                const bool dft_rm_ok = llama_memory_seq_rm(mem_dft, 0, n_past, -1);
                phase_add(session->phase_seq_rm, phase_start);
                trace_draft_seq_rm("live_partial", n_past, -1, dft_rm_ok);
            }
            session->debug_seq_rm_count += 2;
            // `live_ok` re-derives the frontier physically, so a refused removal
            // already falls through to replay. The explicit result check above
            // makes that intent visible rather than incidental.
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
                const bool ckpt_rm_ok = llama_memory_seq_rm(mem_tgt, 0, ckpt.pos_max + 1, -1);
                trace_target_seq_rm("ckpt_restore", ckpt.pos_max + 1, -1, ckpt_rm_ok);
                if (!ckpt_rm_ok) {
                    // The restore did not land: tokens remain above the
                    // checkpoint. The caller replays from here, but the target
                    // must not be treated as canonical in the meantime.
                    session->last_target_untrusted = true;
                }
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
                const bool ckpt_dft_rm_ok =
                    llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
                phase_add(session->phase_seq_rm, phase_start_seq_rm);
                trace_draft_seq_rm(
                    "ckpt_restore", ckpt.pos_max + 1, -1, ckpt_dft_rm_ok);
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
            trace_draft_frontier("ckpt_load_dft", llama_get_memory(ctx_dft));
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
        if (session->owns_model_dft) {
            llama_model_free(session->model_dft);
        }
        // Cleared either way: a borrowed pointer must not outlive this
        // session's view of it, and clearing makes repeated cleanup safe.
        session->model_dft = nullptr;
    }
}

// Whether the persistent pair may be reused without rebuilding it. Every term is
// physical or identity-based: nothing here trusts the caller's intent.
//   - the pair must not be poisoned by an earlier uncertain mutation
//   - the speculative implementation must exist and still belong to THIS target
//   - target and draft must agree on the same frontier F
//   - pending_h must hold the predecessor for a suffix beginning at F+1, which is
//     position F; frontier agreement alone is not sufficient, because a draft can
//     be frontier-correct and content-wrong (proved by the D3b-R2 review)
static bool persistent_pair_is_reusable(
    orbit_mtp_session * session,
    llama_context * ctx_tgt,
    llama_memory_t mem_tgt,
    llama_memory_t mem_dft
) {
    if (!session || !ctx_tgt || !mem_tgt || !mem_dft) { return false; }
    if (session->persistent_pair_untrusted) { return false; }
    if (!session->spec) { return false; }
    if (session->spec_pinned_ctx_tgt != ctx_tgt) { return false; }

    const int32_t tgt_max = llama_memory_seq_pos_max(mem_tgt, 0);
    const int32_t dft_max = llama_memory_seq_pos_max(mem_dft, 0);
    if (tgt_max < 0 || tgt_max != dft_max) { return false; }

    int32_t pend_pos = -1;
    uint64_t pend_fp = 0;
    uint64_t pend_gen = 0;
    if (!common_speculative_pending_state(session->spec, 0, &pend_pos, &pend_fp, &pend_gen)) {
        return false;
    }
    // The predecessor of position F+1 is F.
    return pend_pos == tgt_max && pend_gen > 0;
}

// Soft reset: keep the canonical pair, clear only per-request bookkeeping.
// Deliberately does NOT touch mem_tgt, mem_dft, or the speculative implementation.
// Everything inside the implementation that is per-request -- i_batch_beg/end,
// verify_h, verify_h_rows, verify_pos, last_n_drafted, the samplers -- is
// rewritten before it is read within a request, and every accept() is preceded by
// a process() in the same request, so none of it carries stale meaning across the
// boundary. pending_h, which does carry meaning, is exactly what must survive.
static void soft_reset_request_state(orbit_mtp_session * session) {
    session->request_boundary_ckpt.clear();
    session->request_boundary_prompt_tgt.clear();
    // The resident claim is deliberately NOT cleared here. This runs at the top
    // of the completion the claim was set for, so clearing it would zero the
    // claim before it is read below, and `resident_ok` would be false on exactly
    // the turns where the pair IS reusable -- resident reuse could never happen.
    // Nothing leaks: the claim is consumed one-shot where it is read (it is
    // zeroed on the same two lines), and the hard path clears it separately.
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
    session->spec_epoch++;
    // Pin the target this implementation was built against; soft reuse checks it.
    session->spec_pinned_ctx_tgt = session->spec_params.draft.ctx_tgt;
    if (!session->spec) {
        trace_spec_state(
            "request_reset_after",
            session->spec_epoch,
            static_cast<const void *>(session->spec),
            static_cast<const void *>(session->ctx_dft));
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

extern "C" bool orbit_mtp_session_request_boundary_refill_marker() {
    return true;
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
    session->owns_model_dft = true;
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
    session->spec_epoch++;
    // Pin the target this implementation was built against; soft reuse checks it.
    session->spec_pinned_ctx_tgt = session->spec_params.draft.ctx_tgt;
    // Anchor the epoch baseline. Without this the first event of any run carries
    // a nonzero epoch with no preceding construction event to explain it.
    trace_spec_state(
        "session_create",
        session->spec_epoch,
        static_cast<const void *>(session->spec),
        static_cast<const void *>(session->ctx_dft));
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

    // Observe the pair BEFORE anything is destroyed. This is the decisive
    // measurement of the mission: whatever survives from here to the matching
    // "reset_after" pair is what a future persistent-draft design would inherit.
    {
        auto * mem_before = llama_get_memory(session->ctx_dft);
        trace_draft_frontier("reset_before", mem_before);
        trace_spec_state(
            "reset_before",
            session->spec_epoch,
            static_cast<const void *>(session->spec),
            static_cast<const void *>(session->ctx_dft));
    }

    if (session->spec) {
        common_speculative_free(session->spec);
        session->spec = nullptr;
    }

    auto * mem = llama_get_memory(session->ctx_dft);
    if (mem) {
        llama_memory_clear(mem, true);
        trace_draft_clear("reset");
    }

    session->spec_params.draft.ctx_tgt = ctx_tgt;
    session->spec_params.draft.ctx_dft = session->ctx_dft;
    session->spec = common_speculative_init(session->spec_params, 1);
    session->spec_epoch++;
    // Pin the target this implementation was built against; soft reuse checks it.
    session->spec_pinned_ctx_tgt = session->spec_params.draft.ctx_tgt;
    if (!session->spec) {
        // Emit before returning: the epoch already advanced, so without this the
        // stream would show an unexplained +1 with no closing record. spec=0
        // identifies the failure.
        trace_spec_state(
            "reset_after",
            session->spec_epoch,
            static_cast<const void *>(session->spec),
            static_cast<const void *>(session->ctx_dft));
        set_error("failed to reinitialize speculative MTP state");
        return false;
    }
    {
        auto * mem_after = llama_get_memory(session->ctx_dft);
        trace_draft_frontier("reset_after", mem_after);
        trace_spec_state(
            "reset_after",
            session->spec_epoch,
            static_cast<const void *>(session->spec),
            static_cast<const void *>(session->ctx_dft));
    }
    session->request_boundary_ckpt.clear();
    session->request_boundary_prompt_tgt.clear();
    // A pending claim describes one specific completion. Resetting speculative
    // state invalidates it, but deliberately does NOT touch mem_tgt: draft
    // state and canonical target KV are separate concerns.
    session->pending_resident_prefix_len = 0;
    // This reset just destroyed the draft half and the implementation holding
    // pending_h. A trust bit left asserting "canonical pair" would be describing
    // state that no longer exists. Physical checks happen to catch it downstream,
    // but a flag that survives the operation destroying what it describes is a
    // latent false-trust path, not a design.
    session->persistent_pair_untrusted = true;

    return true;
}

// Single-GGUF self-MTP. Deliberately a separate symbol rather than inferring
// self-MTP from `target_path == draft_path`: the two modes differ in who owns
// the model, and inferring ownership from an incidental string comparison is
// how double frees are written.
//
// The caller passes the llama_model * it already loaded and continues to own
// it. This session creates only the MTP context and the speculative state.
extern "C" void * orbit_selfmtp_session_create(
    void * model_ptr,
    void * ctx_tgt_ptr,
    uint32_t n_ctx,
    uint32_t n_batch,
    uint32_t n_ubatch,
    int32_t n_threads,
    int32_t n_threads_batch
) {
    g_last_error.clear();
    if (!model_ptr) {
        set_error("self-MTP requires the already-loaded model");
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

    // Borrowed, never loaded here and never freed here. Same for ctx_tgt.
    session->owns_model_dft = false;
    session->model_dft = static_cast<llama_model *>(model_ptr);
    session->ctx_tgt_borrowed = static_cast<llama_context *>(ctx_tgt_ptr);

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
        set_error("failed to create self-MTP draft context");
        cleanup_session(session.get());
        return nullptr;
    }

    session->spec_params.types = common_speculative_types_from_names({"draft-mtp"});
    session->spec_params.draft.n_max = ORBIT_MTP_DRAFT_N_MAX;
    session->spec_params.draft.ctx_tgt = static_cast<llama_context *>(ctx_tgt_ptr);
    session->spec_params.draft.ctx_dft = session->ctx_dft;

    session->spec = common_speculative_init(session->spec_params, 1);
    session->spec_epoch++;
    // Pin the target this implementation was built against; soft reuse checks it.
    session->spec_pinned_ctx_tgt = session->spec_params.draft.ctx_tgt;
    // Anchor the epoch baseline. Without this the first event of any run carries
    // a nonzero epoch with no preceding construction event to explain it.
    trace_spec_state(
        "session_create",
        session->spec_epoch,
        static_cast<const void *>(session->spec),
        static_cast<const void *>(session->ctx_dft));
    session->rss_peak_kb = std::max(session->rss_peak_kb, rss_kb());
    if (!session->spec) {
        set_error("failed to initialize self-MTP speculative state");
        cleanup_session(session.get());
        return nullptr;
    }

    session->rss_after_init_kb = rss_kb();
    return session.release();
}

// Declare a resident target-KV prefix for the NEXT completion only.
//
// `n` is a length the runtime has already verified semantically -- it proved
// `prompt_tokens[:n] == committed_tokens` before calling. This layer never
// repeats that comparison; it verifies only that physical target memory is
// consistent with the claim, and falls back to a full replay if it is not.
//
// Deliberately a separate symbol rather than a new parameter on
// `orbit_mtp_session_complete`: the base ABI stays stable, and a shim without
// this symbol remains valid for every path that does not ask for resident
// reuse.
extern "C" bool orbit_mtp_session_set_resident_prefix_len(void * handle, int32_t n) {
    g_last_error.clear();
    auto * session = static_cast<orbit_mtp_session *>(handle);
    if (!session) {
        set_error("resident prefix requires a valid session");
        return false;
    }
    if (n < 0) {
        set_error("resident prefix length must be non-negative");
        return false;
    }
    session->pending_resident_prefix_len = n;
    return true;
}

// Whether the last completion actually preserved and reused a resident prefix.
// False whenever the claim was absent, rejected, or physically unverifiable --
// there is no "maybe" state.
extern "C" bool orbit_mtp_session_last_resident_reuse_active(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_resident_reuse_active : false;
}

extern "C" bool orbit_mtp_session_owns_model(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->owns_model_dft : false;
}

// Destroy a self-MTP session. Frees ONLY what this session created -- the
// speculative state and the MTP context. The model and the target context are
// borrowed and are freed by the client that owns them, AFTER this returns.
//
// Idempotent: every pointer is nulled as it is released, so a second call is a
// no-op rather than a double free.
extern "C" void orbit_selfmtp_session_destroy(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
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
    // Borrowed. Dropped, never freed.
    session->model_dft = nullptr;
    session->ctx_tgt_borrowed = nullptr;
    delete session;
}

// Diagnostics for the ownership tests: whether the borrowed handles are still
// the ones the caller passed in, so a test can prove they were not disturbed.
extern "C" void * orbit_mtp_session_borrowed_model(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? static_cast<void *>(session->model_dft) : nullptr;
}

extern "C" void * orbit_mtp_session_borrowed_ctx_tgt(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? static_cast<void *>(session->ctx_tgt_borrowed) : nullptr;
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
    // Bracket this destruction explicitly. `reset_speculative_request_state`
    // frees and recreates the speculative implementation on EVERY completion,
    // independently of `orbit_mtp_session_reset`. Without its own labelled pair
    // the epoch would jump between one request's exit and the next request's
    // entry with nothing naming the cause, and an analyst would attribute all
    // destruction to the session reset -- concluding that softening that reset
    // would preserve `pending_h`, which is false while this site exists.
    {
        auto * mem_dft_pre = llama_get_memory(session->ctx_dft);
        trace_draft_frontier("request_reset_before", mem_dft_pre);
        trace_spec_state(
            "request_reset_before",
            session->spec_epoch,
            static_cast<const void *>(session->spec),
            static_cast<const void *>(session->ctx_dft));
    }
    // Soft when the whole pair is provably canonical and still belongs to this
    // target, hard otherwise. A soft reset keeps ctx_tgt, ctx_dft, the
    // speculative implementation and pending_h, and clears only per-request
    // bookkeeping, so the next request can append its suffix instead of
    // rebuilding history. Anything uncertain rebuilds: correctness over cache.
    {
        auto * mem_tgt_pre = llama_get_memory(ctx_tgt);
        auto * mem_dft_pre2 = llama_get_memory(session->ctx_dft);
        const bool soft = persistent_pair_is_reusable(
            session, ctx_tgt, mem_tgt_pre, mem_dft_pre2);
        session->last_request_reset_was_soft = soft;
        if (soft) {
            soft_reset_request_state(session);
            trace_request_reset_mode("soft");
        } else {
            trace_request_reset_mode("hard");
            if (!reset_speculative_request_state(session, ctx_tgt)) {
                return false;
            }
            // A rebuilt pair has proven nothing yet.
            session->persistent_pair_untrusted = true;
        }
    }
    {
        auto * mem_dft_post = llama_get_memory(session->ctx_dft);
        trace_draft_frontier("request_reset_after", mem_dft_post);
        trace_spec_state(
            "request_reset_after",
            session->spec_epoch,
            static_cast<const void *>(session->spec),
            static_cast<const void *>(session->ctx_dft));
    }

    session->last_content.clear();
    session->last_output_tokens = 0;
    session->last_generated_tokens.clear();
    session->last_resident_tokens.clear();
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
    session->last_validate_steps = 0;
    session->last_rows_requested_total = 0;
    session->last_rows_consumed_estimated_total = 0;
    session->last_rows_wasted_estimated_total = 0;
    session->last_accepted_draft_hist_0 = 0;
    session->last_accepted_draft_hist_1 = 0;
    session->last_accepted_draft_hist_2 = 0;
    session->last_accepted_draft_hist_3 = 0;
    session->last_accepted_draft_hist_ge4 = 0;
    session->last_trace_json = "[]";
    session->last_timing_json = "{}";
    session->last_validate_trace_json = "[]";
    session->last_target_decode_trace_json = "[]";
    session->last_validate_equivalence_json = "{}";
    session->last_output_token_hashes.clear();
    session->last_output_token_hashes_json = "[]";
    session->last_first_sample_trace_json = "{}";
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

    std::vector<llama_token> prompt_tgt(prompt);
    llama_token id_last = LLAMA_TOKEN_NULL;
    int32_t n_past = (int32_t) prompt_tgt.size();

    auto sampling_params = make_reference_sampling_params();
    common_sampler * smpl = common_sampler_init(model_tgt, sampling_params);
    if (!smpl) {
        set_error("failed to initialize common sampler");
        return false;
    }
    const int generation_limit = std::max(1, (int) max_tokens);
    std::vector<llama_token> generated;
    generated.reserve((size_t) generation_limit);
    const auto t0 = std::chrono::steady_clock::now();
    const int32_t progress_prefill_phase = 0;
    const int32_t progress_generation_phase = 1;
    auto emit_output_token = [&](llama_token token) {
        session->last_output_tokens++;
        session->last_output_token_hashes.push_back(stable_hash_token(token));
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
    trace_target_frontier("request_entry", mem_tgt);
    trace_draft_frontier("request_entry", mem_dft);
    trace_spec_state("request_entry", session->spec_epoch,
        static_cast<const void *>(session->spec),
        static_cast<const void *>(session->ctx_dft));

    // Resident-prefix decision. The runtime may declare that the first N prompt
    // tokens are already resident in target memory, having proven token
    // identity against its own committed sequence. This layer does not repeat
    // that comparison -- it asks only whether PHYSICAL memory agrees:
    //
    //   * the claim fits inside this prompt;
    //   * the target frontier is exactly N-1, so N tokens really are resident;
    //   * bounded recurrent rollback is available, since a rejected
    //     speculative tail must be removable without discarding the prefix.
    //
    // `llama_n_rs_seq` is read directly rather than via `can_partial_rollback`,
    // whose probe clears memory and would destroy the very prefix at stake.
    //
    // Any disagreement falls through to the existing replay path with no claim
    // of reuse: a wrong prefix is a correctness bug, a full replay is only slow.
    // Carried in from the PREVIOUS completion: if a target seq_rm was refused
    // there, the frontier may include tokens that were supposed to be removed,
    // so a claim of frontier+1 could match a poisoned state. Read it before
    // clearing, so the flag gates exactly one following completion.
    const bool target_untrusted_from_previous = session->last_target_untrusted;
    session->last_target_untrusted = false;
    const int32_t resident_claim = session->pending_resident_prefix_len;
    session->pending_resident_prefix_len = 0;
    // The claim must be a STRICT PROPER prefix: `< size()`, not `<=`. A claim
    // covering the whole prompt would leave an empty suffix, so this completion
    // would decode nothing into the target, and the first sample would then read
    // logits left behind by the PREVIOUS completion. Falling closed to the full
    // replay costs one prefill and guarantees fresh logits before sampling.
    // Resident reuse now requires the WHOLE pair, not just the target. The draft
    // must physically agree on the same frontier, and pending_h must hold the
    // predecessor for a suffix beginning at the claim -- position claim-1. Without
    // the pending term a draft could be frontier-correct and content-wrong, which
    // is exactly the defect the D3b-R2 review found in the earlier attempt.
    // The pair carried in from the previous completion is consulted before it is
    // re-armed below, so one bad completion gates the next.
    int32_t entry_pend_pos = -1;
    uint64_t entry_pend_fp = 0;
    uint64_t entry_pend_gen = 0;
    const bool entry_pend_ok = common_speculative_pending_state(
        session->spec, 0, &entry_pend_pos, &entry_pend_fp, &entry_pend_gen);
    const bool pair_trusted_from_previous = !session->persistent_pair_untrusted;
    const bool identity_ok =
        session->spec != nullptr && session->spec_pinned_ctx_tgt == ctx_tgt;
    const bool resident_ok =
        resident_claim > 0 &&
        !target_untrusted_from_previous &&
        pair_trusted_from_previous &&
        identity_ok &&
        resident_claim < (int32_t) prompt_tgt.size() &&
        llama_n_rs_seq(ctx_tgt) > 0 &&
        llama_memory_seq_pos_max(mem_tgt, 0) == resident_claim - 1 &&
        llama_memory_seq_pos_max(mem_dft, 0) == resident_claim - 1 &&
        entry_pend_ok &&
        entry_pend_pos == resident_claim - 1 &&
        entry_pend_gen > 0;
    session->last_resident_reuse_active = resident_ok;
    trace_resident_admission(
        resident_claim, resident_ok, pair_trusted_from_previous, identity_ok,
        entry_pend_ok ? entry_pend_pos : -1, entry_pend_gen);

    // Fail-closed target-trust contract. From here on this completion may mutate
    // physical target state (clear, prefill decode, validation decode, seq_rm).
    // Any exit that is not the single proven-canonical success return at the end
    // of this function leaves that state unverified -- notably a failed
    // `llama_decode(ctx_tgt, validate)`, which has already written speculative
    // tokens above the committed frontier by the time it reports failure. Rather
    // than patch each individual return site, arm the flag once here and clear it
    // only after the loop has completed normally. A later resident claim is then
    // refused until a cold clear/full replay has re-established canonical state.
    session->last_target_untrusted = true;
    // The pair is poisoned for the same reason and at the same moment as the
    // target: from here on this completion may mutate target KV, draft KV and
    // pending_h, and only the proven-canonical exit re-establishes them together.
    session->persistent_pair_untrusted = true;

    bool need_replay = !resident_ok;
    // The prefill block below lives under `need_replay`. On the resident path
    // we still need to enter it once -- to decode the SUFFIX -- while skipping
    // the target clear. This one-shot flag distinguishes that first pass from a
    // genuine replay.
    bool resident_prefill_pending = resident_ok;
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
        session->request_boundary_ckpt.data_dft.empty() ? 0 : session->request_boundary_prompt_tgt.size();
    const bool can_restore_request_boundary =
        !session->request_boundary_ckpt.data_dft.empty() &&
        is_token_prefix(session->request_boundary_prompt_tgt, prompt_tgt);
    bool used_request_boundary = false;
    bool request_boundary_logits_refreshed = false;
    int last_target_prefill_batch_n_tokens = 0;

    while ((int) generated.size() < generation_limit) {
        const auto loop_phase_start = std::chrono::steady_clock::now();
        const bool loop_need_replay_before = need_replay;
        if (need_replay || resident_prefill_pending) {
            const bool resident_pass = resident_prefill_pending;
            resident_prefill_pending = false;
            const bool replay_is_recovery = is_recovery_replay;
            const auto replay_phase_start = std::chrono::steady_clock::now();
            // A soft reset preserved `pending_h` for a suffix append at F+1. If
            // this completion instead replays from position 0, that row is the
            // WRONG predecessor: process() would seed draft slot 0 with the last
            // frontier's hidden state where "no predecessor" is required, and the
            // vendored code states the invariant directly -- "-1 == no
            // predecessor: correct only for a batch starting at position 0".
            // The resulting draft KV would be built from a wrong seed and the
            // exit check would still certify it trusted, because process()
            // overwrites pending_h with the last row before that check runs.
            // Discard the carryover whenever this pass is not a resident append.
            if (!resident_pass && session->spec) {
                if (!common_speculative_reset_pending(session->spec, 0)) {
                    // Cannot prove the carryover is safe for a from-zero replay,
                    // so refuse to reuse the implementation at all.
                    session->persistent_pair_untrusted = true;
                }
                trace_pending_discarded("replay_from_zero");
            }
            // Resident reuse takes precedence over request-boundary restore, and
            // the exclusion is written here rather than assumed: the boundary
            // branch clears mem_tgt and replays the prompt from 0, which would
            // destroy the very prefix a resident pass is preserving. Both flags
            // are computed independently and can be true together, so nothing
            // but this `!resident_pass` keeps them apart. The caller currently
            // resets the session before each completion, which clears the
            // boundary checkpoint and makes the collision unlikely -- but that
            // is caller ordering, not an invariant of this function.
            const bool use_request_boundary = generated.empty() && can_restore_request_boundary
                && !used_request_boundary && !resident_pass;
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
            // Both halves are cleared together or not at all. A resident pass
            // preserves the target prefix AND the draft state that matches it;
            // clearing only the draft would leave the pair mismatched -- draft
            // empty while the target holds prompt[0:N) -- and pending_h naming a
            // predecessor the draft no longer contains. The guard was previously
            // on the target alone, which is what made the draft half unusable.
            if (!resident_pass) {
                llama_memory_clear(mem_tgt, true);
                trace_target_clear("replay");
                session->debug_memory_clear_count++;
                llama_memory_clear(mem_dft, true);
                trace_draft_clear("replay");
                session->debug_memory_clear_count++;
            }

            if (!prompt_tgt.empty()) {
                if (use_request_boundary) {
                    {
                        const auto phase_start = std::chrono::steady_clock::now();
                        session->request_boundary_ckpt.load_dft(session->ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        phase_add(session->phase_prefix_restore, phase_start);
                        trace_draft_frontier("boundary_load_dft", llama_get_memory(session->ctx_dft));
                    }
                    {
                        const auto phase_start = std::chrono::steady_clock::now();
                        llama_memory_clear(mem_tgt, true);
                        trace_target_clear("request_boundary");
                        phase_add(session->phase_seq_rm, phase_start);
                    }
                    session->debug_memory_clear_count++;
                    const size_t chunk_size = (size_t) std::max<uint32_t>(1, session->n_batch);
                    for (size_t offset = 0; offset < prompt_tgt.size(); offset += chunk_size) {
                        const size_t count = std::min(chunk_size, prompt_tgt.size() - offset);
                        std::vector<llama_token> chunk(
                            prompt_tgt.begin() + (ptrdiff_t) offset,
                            prompt_tgt.begin() + (ptrdiff_t) (offset + count));
                        const auto batch_build_start = std::chrono::steady_clock::now();
                        llama_batch refresh_tgt = llama_batch_init((int32_t) count, 0, 1);
                        fill_target_prefill_batch(refresh_tgt, chunk, (int32_t) offset);
                        trace_target_prefill("boundary_refresh", (int32_t) offset, chunk.size());
                        last_target_prefill_batch_n_tokens = refresh_tgt.n_tokens;
                        phase_add(session->phase_batch_build, batch_build_start);
                        const long long decode_started_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
                        double decode_ms = 0.0;
                        session->debug_prefill_target_suffix_count++;
                        {
                            const auto phase_start = std::chrono::steady_clock::now();
                            if (llama_decode(ctx_tgt, refresh_tgt) != 0) {
                                decode_ms = elapsed_ms(phase_start);
                                target_decode_traces.push_back(make_target_decode_trace(
                                    "request_boundary_target_refill",
                                    trace_step_index,
                                    0,
                                    0,
                                    refresh_tgt,
                                    decode_started_us,
                                    decode_ms));
                                phase_add(session->phase_suffix_decode_target, phase_start);
                                llama_batch_free(refresh_tgt);
                                common_sampler_free(smpl);
                                set_error("failed to refill restored target prompt");
                                return false;
                            }
                            decode_ms = elapsed_ms(phase_start);
                            phase_add(session->phase_suffix_decode_target, phase_start);
                        }
                        target_decode_traces.push_back(make_target_decode_trace(
                            "request_boundary_target_refill",
                            trace_step_index,
                            0,
                            0,
                            refresh_tgt,
                            decode_started_us,
                            decode_ms));
                        session->last_target_decode_calls++;
                        llama_batch_free(refresh_tgt);
                    }
                    request_boundary_logits_refreshed = true;
                    used_request_boundary = true;
                } else if (!resident_pass) {
                    // Cold replay only. A resident pass must not land here: the
                    // target already holds prompt[0:resident_claim], so
                    // decoding the whole prompt at position 0 would overlap
                    // resident positions and llama_decode rejects it. The
                    // resident suffix is prefilled by the positional block
                    // below, at pos0 = resident_claim.
                    const size_t chunk_size = (size_t) std::max<uint32_t>(1, session->n_batch);
                    for (size_t offset = 0; offset < prompt_tgt.size(); offset += chunk_size) {
                        const size_t count = std::min(chunk_size, prompt_tgt.size() - offset);
                        std::vector<llama_token> chunk(
                            prompt_tgt.begin() + (ptrdiff_t) offset,
                            prompt_tgt.begin() + (ptrdiff_t) (offset + count));
                        const auto batch_build_start = std::chrono::steady_clock::now();
                        llama_batch prefill_tgt = llama_batch_init((int32_t) count, 0, 1);
                        fill_target_prefill_batch(prefill_tgt, chunk, (int32_t) offset);
                        trace_target_prefill("full_replay", (int32_t) offset, chunk.size());
                        last_target_prefill_batch_n_tokens = prefill_tgt.n_tokens;
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
                if (resident_pass) {
                    // prompt[0:resident_claim] is already resident; decode only
                    // the suffix, based at the resident boundary. When the claim
                    // covers the whole prompt this is empty and nothing is
                    // decoded.
                    process_tokens.assign(
                        prompt_tgt.begin() + (ptrdiff_t) resident_claim,
                        prompt_tgt.end());
                    process_pos0 = resident_claim;
                } else if (use_request_boundary) {
                    process_tokens.assign(
                        prompt_tgt.begin() + (ptrdiff_t) reusable_request_prefix,
                        prompt_tgt.end());
                    process_pos0 = (int32_t) reusable_request_prefix;
                } else {
                    process_tokens = prompt_tgt;
                    process_pos0 = 0;
                }
                if (!process_tokens.empty()) {
                    // Both the request-boundary and resident paths decode a
                    // slice at a non-zero base, so they share this loop; the
                    // `else` below is the from-zero full replay.
                    if (use_request_boundary || resident_pass) {
                        const size_t chunk_size = (size_t) std::max<uint32_t>(1, session->n_batch);
                        for (size_t offset = 0; offset < process_tokens.size(); offset += chunk_size) {
                            const size_t count = std::min(chunk_size, process_tokens.size() - offset);
                            std::vector<llama_token> chunk(
                                process_tokens.begin() + (ptrdiff_t) offset,
                                process_tokens.begin() + (ptrdiff_t) (offset + count));
                            const auto batch_build_start = std::chrono::steady_clock::now();
                            llama_batch prefill_tgt = llama_batch_init((int32_t) count, 0, 1);
                            fill_target_prefill_batch(prefill_tgt, chunk, process_pos0 + (int32_t) offset);
                            trace_target_prefill("suffix", process_pos0 + (int32_t) offset, chunk.size());
                            last_target_prefill_batch_n_tokens = prefill_tgt.n_tokens;
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
                    // The draft context has its OWN lifecycle. The target may reuse a
                    // resident prefix and prefill only the suffix, but the draft memory
                    // was cleared unconditionally above, so replaying only the suffix
                    // there would leave positions [0, resident_claim) empty and the MTP
                    // head would attend over a hole. Whether the draft can inherit the
                    // target's history is a native fact, not a model-name fact:
                    // llama.cpp keeps `ctx_other` only for architectures whose draft
                    // memory is physically shared with the target (see
                    // llama-context.cpp), and `common_speculative` skips its catch-up
                    // decode exactly when that holds. When it does NOT hold, the draft
                    // must be rebuilt contiguously from position 0.
                    // SAME-SUFFIX CONTRACT. `common_speculative_process` consumes
                    // the target's nextn hidden rows from the target's MOST RECENT
                    // decode, indexed by batch slot. The batch handed to it must
                    // therefore be exactly the batch the target just decoded.
                    //
                    // On a resident pass the target decoded only the suffix, and a
                    // persistent draft already holds the prefix with pending_h
                    // naming its last position, so the draft appends the SAME
                    // suffix. Processing the full prompt here -- which an earlier
                    // attempt did -- reads rows the target never produced for those
                    // slots and silently builds the draft from stale conditioning.
                    //
                    // On a cold/replay pass the target decoded the whole prompt, so
                    // the same rule yields the whole prompt. One rule, both cases:
                    // the draft always mirrors the target's last decode.
                    const std::vector<llama_token> & draft_tokens = process_tokens;
                    const int32_t draft_pos0 = process_pos0;
                    const size_t chunk_size = (size_t) std::max<uint32_t>(1, session->n_batch);
                    for (size_t offset = 0; offset < draft_tokens.size(); offset += chunk_size) {
                        const size_t count = std::min(chunk_size, draft_tokens.size() - offset);
                        std::vector<llama_token> chunk(
                            draft_tokens.begin() + (ptrdiff_t) offset,
                            draft_tokens.begin() + (ptrdiff_t) (offset + count));
                        const auto batch_build_start = std::chrono::steady_clock::now();
                        llama_batch prefill = llama_batch_init((int32_t) count, 0, 1);
                        fill_batch(prefill, chunk, draft_pos0 + (int32_t) offset);
                        phase_add(session->phase_batch_build, batch_build_start);
                        trace_draft_process(
                            "prefill", draft_pos0 + (int32_t) offset, chunk.size());
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
                const std::string first_sampler_before = common_sampler_prev_str(smpl, ctx_tgt, 8);
                {
                    const auto phase_start = std::chrono::steady_clock::now();
                    id_last = common_sampler_sample(smpl, ctx_tgt, -1);
                    common_sampler_accept(smpl, id_last, true);
                    phase_add(session->phase_sampler_ops, phase_start);
                }
                const std::string first_sampler_after = common_sampler_prev_str(smpl, ctx_tgt, 8);
                session->last_first_sample_trace_json = first_sample_trace_json(
                    "mtp",
                    prompt_tgt,
                    ctx_tgt,
                    vocab_tgt,
                    mem_tgt,
                    n_past,
                    last_target_prefill_batch_n_tokens,
                    1,
                    1,
                    first_sampler_before,
                    id_last,
                    first_sampler_after,
                    1,
                    used_request_boundary,
                    request_boundary_logits_refreshed);
                if (llama_vocab_is_eog(vocab_tgt, id_last)) {
                    goto done;
                }
                generated.push_back(id_last);
                emit_output_token(id_last);
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
                if ((int) generated.size() >= generation_limit) {
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
                std::min(ORBIT_MTP_DRAFT_N_MAX, generation_limit - (int) generated.size()),
                (llama_pos) n_past,
                id_last,
                &prompt_tgt,
                &draft,
            };
            // Proposal generation also DECODES into ctx_dft: on the non-shared
            // memory path it adds tokens at n_past + i + 1, so the draft frontier
            // advances by up to MTP_DRAFT_N_MAX per step. That motion has no other
            // event class in this stream, so bracket it -- otherwise the exit
            // frontier is larger than the traced writes can account for, which is
            // the ambiguity this recorder exists to remove. The reads sit OUTSIDE
            // the phase bracket so observation cost never lands in exported timing.
            trace_draft_frontier("draft_generate_before", mem_dft);
            {
                const auto phase_start = std::chrono::steady_clock::now();
                common_speculative_draft(session->spec);
                phase_add(session->phase_draft_generation, phase_start);
            }
            trace_draft_frontier("draft_generate_after", mem_dft);
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
                        << ",\"input_token_hash\":" << stable_hash_token(input_token)
                        << ",\"sampled_draft_token_hash\":" << stable_hash_token(sampled_token)
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
                        const bool ckpt_dft_rm_ok2 =
                            llama_memory_seq_rm(mem_dft, 0, ckpt.pos_max + 1, -1);
                        phase_add(session->phase_seq_rm, phase_start);
                        trace_draft_seq_rm(
                            "ckpt_restore_step", ckpt.pos_max + 1, -1, ckpt_dft_rm_ok2);
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
                    trace_draft_frontier("ckpt_load_dft_step", llama_get_memory(session->ctx_dft));
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
        trace_step.sampler_state_hash_before = trace_step.sampler_before_hash;
        trace_step.draft = draft;
        trace_step.draft_origin = draft.empty() ? "empty" : (draft_is_fresh ? "fresh" : "reused");
        trace_step.draft_is_fresh = draft_is_fresh;
        trace_step.need_replay_before = loop_need_replay_before;
        trace_step.sampled_id = -1;
        trace_step.rejected_id = -1;
        trace_step.validated_count = (int) validate_tokens.size();
        trace_step.validate_n_tok = (int) validate_tokens.size();
        trace_step.validate_pos0 = validate_pos0;
        trace_step.checkpoint_total = session->last_checkpoint_count;
        trace_step.restore_total = session->last_restore_count;
        trace_step.id_last_before = (int) id_last;
        trace_step.n_past_before = n_past;
        trace_step.old_n_past = n_past;
        trace_step.prompt_tgt_size_before = (int32_t) prompt_tgt.size();
        trace_step.prompt_tgt_len = (int32_t) prompt_tgt.size();
        trace_step.prompt_dft_len = llama_memory_seq_pos_max(mem_dft, 0) >= 0
            ? llama_memory_seq_pos_max(mem_dft, 0) + 1
            : -1;
        trace_step.prompt_tgt_hash = stable_hash_tokens(prompt_tgt);
        trace_step.prompt_tgt_pos_next_before = n_past;
        trace_step.kv_tgt_before_min = llama_memory_seq_pos_min(mem_tgt, 0);
        trace_step.kv_tgt_before_max = llama_memory_seq_pos_max(mem_tgt, 0);
        trace_step.kv_dft_before_min = llama_memory_seq_pos_min(mem_dft, 0);
        trace_step.kv_dft_before_max = llama_memory_seq_pos_max(mem_dft, 0);
        trace_step.ctx_tgt_frontier_hash = stable_hash_frontier(trace_step.kv_tgt_before_min, trace_step.kv_tgt_before_max);
        trace_step.ctx_dft_frontier_hash = stable_hash_frontier(trace_step.kv_dft_before_min, trace_step.kv_dft_before_max);
        trace_step.remaining_generation_cap = std::max(0, generation_limit - (int) generated.size());
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
        const int draft_size = (int) trace_step.draft.size();
        const int rows_requested = draft_size + 1;
        const int rows_consumed_estimated = full_accept ? rows_requested : std::min(rows_requested, accepted + 1);
        const int rows_wasted_estimated = std::max(0, rows_requested - rows_consumed_estimated);
        session->last_validate_steps++;
        session->last_rows_requested_total += rows_requested;
        session->last_rows_consumed_estimated_total += rows_consumed_estimated;
        session->last_rows_wasted_estimated_total += rows_wasted_estimated;
        if (accepted <= 0) {
            session->last_accepted_draft_hist_0++;
        } else if (accepted == 1) {
            session->last_accepted_draft_hist_1++;
        } else if (accepted == 2) {
            session->last_accepted_draft_hist_2++;
        } else if (accepted == 3) {
            session->last_accepted_draft_hist_3++;
        } else {
            session->last_accepted_draft_hist_ge4++;
        }
        trace_step.accepted_ids = ids;
        trace_step.accepted_draft = accepted;
        trace_step.rejected_draft = rejected;
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
        trace_step.sampler_state_hash_after = trace_step.sampler_after_hash;
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
            trace_step.new_n_past = n_past;
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

        for (size_t i = 0; i < ids.size() && (int) generated.size() < generation_limit; ++i) {
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
            bool need_replay_after_failed_rm = false;
            n_past = (int32_t) prompt_tgt.size();
            {
                const auto phase_start = std::chrono::steady_clock::now();
                const bool full_rm_ok = llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
                trace_target_seq_rm("full_accept", n_past, -1, full_rm_ok);
                const bool full_dft_rm_ok = llama_memory_seq_rm(mem_dft, 0, n_past, -1);
                phase_add(session->phase_seq_rm, phase_start);
                trace_draft_seq_rm("full_accept", n_past, -1, full_dft_rm_ok);
                if (!full_rm_ok) {
                    // The verification tail is still resident above the
                    // committed frontier. Declaring "no replay needed" here
                    // would leave those tokens in place and let a later
                    // completion claiming frontier+1 match a poisoned frontier.
                    // Rebuild instead, and mark the target unproven so no
                    // resident claim can rest on it.
                    session->last_target_untrusted = true;
                    need_replay_after_failed_rm = true;
                }
            }
            session->debug_seq_rm_count += 2;
            draft.clear();
            have_ckpt = false;
            draft_is_fresh = false;
            need_replay = need_replay_after_failed_rm;
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
        trace_step.new_n_past = n_past;
        trace_step.prompt_tgt_size_after = (int32_t) prompt_tgt.size();
        trace_step.prompt_tgt_pos_next_after = n_past;
        trace_step.residual_draft_size_after = (int32_t) draft.size();
        trace_step.residual_draft_after_json = token_vec_json(vocab_tgt, draft);
        validate_traces.push_back(validate_trace);
        trace_steps.push_back(trace_step);
        phase_add(session->phase_loop_total, loop_phase_start);
    }

done:
    trace_target_frontier("request_exit", mem_tgt);
    trace_draft_frontier("request_exit", mem_dft);
    trace_spec_state("request_exit", session->spec_epoch,
        static_cast<const void *>(session->spec),
        static_cast<const void *>(session->ctx_dft));
    common_sampler_free(smpl);
    session->cached_prompt_tokens = prompt;
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
    if (mtp_trace_enabled()) {
        session->last_validate_equivalence_json = validate_equivalence_json(
            trace_steps,
            session->last_rows_requested_total,
            session->last_rows_consumed_estimated_total,
            session->last_rows_wasted_estimated_total);
    }
    session->last_output_token_hashes_json = uint64_vec_json(session->last_output_token_hashes);
    {
        const double suffix_target_prefill_ms = session->phase_suffix_decode_target.total_ms;
        const double speculative_loop_including_suffix_ms = session->phase_loop_total.total_ms;
        const double speculative_loop_ms = std::max(0.0, speculative_loop_including_suffix_ms - suffix_target_prefill_ms);
        const double checkpoint_restore_ms =
            session->phase_prefix_restore.total_ms +
            session->phase_ctx_tgt_checkpoint.total_ms +
            session->phase_ctx_tgt_restore.total_ms +
            session->phase_ctx_dft_checkpoint.total_ms +
            session->phase_ctx_dft_restore.total_ms;
        const double sampler_ms =
            session->phase_sampler_clone.total_ms +
            session->phase_sampler_restore.total_ms +
            session->phase_sampler_ops.total_ms;
        const double non_loop_overhead_ms = std::max(0.0, session->last_elapsed_ms - speculative_loop_including_suffix_ms);
        std::ostringstream out;
        out
            << "{"
            << "\"summary\":{"
            << "\"total_wall_ms\":" << session->last_elapsed_ms << ","
            << "\"suffix_target_prefill_ms\":" << suffix_target_prefill_ms << ","
            << "\"speculative_loop_ms\":" << speculative_loop_ms << ","
            << "\"speculative_loop_including_suffix_ms\":" << speculative_loop_including_suffix_ms << ","
            << "\"target_validate_ms\":" << session->phase_target_validate.total_ms << ","
            << "\"draft_generation_ms\":" << session->phase_draft_generation.total_ms << ","
            << "\"checkpoint_restore_ms\":" << checkpoint_restore_ms << ","
            << "\"sampler_ms\":" << sampler_ms << ","
            << "\"seq_rm_ms\":" << session->phase_seq_rm.total_ms << ","
            << "\"non_loop_overhead_ms\":" << non_loop_overhead_ms
            << "},"
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
    // Proven-canonical exit: the speculative loop ran to completion, so every
    // accepted token is committed and every rejected tail has been rolled back.
    // Only here may the target be declared trusted again -- and only if the
    // physical frontier actually agrees with the committed length, which is what
    // a following resident claim will be validated against. A seq_rm refusal
    // earlier in this completion leaves the frontier long; in that case the flag
    // stays armed so the next completion falls back to a clear/full replay.
    // `n_past` is the committed frontier the loop maintains, so compare the
    // physical frontier against it directly rather than reconstructing a length.
    session->last_target_untrusted =
        llama_memory_seq_pos_max(mem_tgt, 0) != n_past - 1;
    // The pair is trusted only when ALL THREE halves agree at the same frontier.
    // Defect C's target-only check is subsumed here rather than replaced: the
    // target term is identical, and the draft and pending terms are what make a
    // persistent pair claim meaningful. Frontier agreement alone is not enough --
    // a draft can be frontier-correct and content-wrong -- so pending_pos must
    // name the predecessor of the next suffix, which is the frontier itself.
    {
        const int32_t canonical_frontier = n_past - 1;
        const bool target_ok =
            llama_memory_seq_pos_max(mem_tgt, 0) == canonical_frontier;
        const bool draft_ok =
            llama_memory_seq_pos_max(mem_dft, 0) == canonical_frontier;
        int32_t pend_pos = -1;
        uint64_t pend_fp = 0;
        uint64_t pend_gen = 0;
        const bool pend_ok = common_speculative_pending_state(
            session->spec, 0, &pend_pos, &pend_fp, &pend_gen);
        const bool pending_aligned =
            pend_ok && pend_pos == canonical_frontier && pend_gen > 0;
        const bool identity_ok =
            session->spec != nullptr && session->spec_pinned_ctx_tgt == ctx_tgt;
        session->persistent_pair_untrusted =
            !(target_ok && draft_ok && pending_aligned && identity_ok);
        trace_pair_trust(
            session->persistent_pair_untrusted ? "untrusted" : "trusted",
            canonical_frontier, target_ok, draft_ok, pending_aligned, identity_ok);
    }
    // Snapshot the decoded ids alongside the verdict, so a caller reads both as
    // of this completion rather than querying mutable session state later.
    session->last_generated_tokens = generated;
    // Publish the resident sequence only when the physical frontier agrees with
    // the committed length. If a seq_rm refusal left the frontier long, the
    // vector no longer describes KV, and an empty publication makes the runtime
    // fall back to a cold prompt rather than trust a stale identity.
    if (llama_memory_seq_pos_max(mem_tgt, 0) == n_past - 1 &&
        prompt_tgt.size() == (size_t) n_past) {
        session->last_resident_tokens = prompt_tgt;
    } else {
        session->last_resident_tokens.clear();
    }
    return true;
}

// Whether target, draft and pending_h were all proven canonical at the end of
// the last completion, and the implementation still belongs to the same target
// context. This is NOT the same question as "did this completion reuse resident
// state": a cold completion can end canonical, and a resident completion can end
// poisoned. The runtime must not conflate them.
extern "C" bool orbit_mtp_session_last_pair_canonical(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? !session->persistent_pair_untrusted : false;
}

// Number of token ids the last completion decoded, and a copy of them. The
// caller sizes its buffer with the count and then copies; a short buffer copies
// nothing and reports the required size, so a caller cannot silently truncate.
extern "C" int32_t orbit_mtp_session_last_generated_token_count(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? (int32_t) session->last_generated_tokens.size() : 0;
}

extern "C" int32_t orbit_mtp_session_last_generated_tokens(
        void * handle, int32_t * out, int32_t capacity) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    if (!session) { return 0; }
    const int32_t count = (int32_t) session->last_generated_tokens.size();
    if (!out || capacity < count) { return count; }
    for (int32_t i = 0; i < count; ++i) {
        out[i] = (int32_t) session->last_generated_tokens[i];
    }
    return count;
}

// The tokens physically resident in the target KV after the last completion.
// The runtime publishes committed identity from these, never from
// prompt + generated: see `last_resident_tokens`.
extern "C" int32_t orbit_mtp_session_last_resident_token_count(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? (int32_t) session->last_resident_tokens.size() : 0;
}

extern "C" int32_t orbit_mtp_session_last_resident_tokens(
        void * handle, int32_t * out, int32_t capacity) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    if (!session) { return 0; }
    const int32_t count = (int32_t) session->last_resident_tokens.size();
    if (!out || capacity < count) { return count; }
    for (int32_t i = 0; i < count; ++i) {
        out[i] = (int32_t) session->last_resident_tokens[i];
    }
    return count;
}

extern "C" const char * orbit_mtp_session_last_content(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_content.c_str() : "";
}

extern "C" int32_t orbit_mtp_session_last_output_tokens(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_output_tokens : 0;
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

extern "C" int32_t orbit_mtp_session_last_validate_steps(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_validate_steps : 0;
}

extern "C" int32_t orbit_mtp_session_last_rows_requested_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_rows_requested_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_rows_consumed_estimated_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_rows_consumed_estimated_total : 0;
}

extern "C" int32_t orbit_mtp_session_last_rows_wasted_estimated_total(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_rows_wasted_estimated_total : 0;
}

extern "C" double orbit_mtp_session_last_rows_wasted_estimated_ratio(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    if (!session || session->last_rows_requested_total <= 0) {
        return 0.0;
    }
    return (double) session->last_rows_wasted_estimated_total / (double) session->last_rows_requested_total;
}

extern "C" int32_t orbit_mtp_session_last_accepted_draft_hist_0(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_accepted_draft_hist_0 : 0;
}

extern "C" int32_t orbit_mtp_session_last_accepted_draft_hist_1(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_accepted_draft_hist_1 : 0;
}

extern "C" int32_t orbit_mtp_session_last_accepted_draft_hist_2(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_accepted_draft_hist_2 : 0;
}

extern "C" int32_t orbit_mtp_session_last_accepted_draft_hist_3(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_accepted_draft_hist_3 : 0;
}

extern "C" int32_t orbit_mtp_session_last_accepted_draft_hist_ge4(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_accepted_draft_hist_ge4 : 0;
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

extern "C" const char * orbit_mtp_session_last_validate_equivalence_json(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_validate_equivalence_json.c_str() : "{}";
}

extern "C" const char * orbit_mtp_session_last_output_token_hashes_json(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_output_token_hashes_json.c_str() : "[]";
}

extern "C" const char * orbit_mtp_session_last_first_sample_trace_json(void * handle) {
    auto * session = static_cast<orbit_mtp_session *>(handle);
    return session ? session->last_first_sample_trace_json.c_str() : "{}";
}
