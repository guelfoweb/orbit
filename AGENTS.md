# AGENTS.md

## Ruolo

Questo file guida agenti e sessioni Codex che lavorano su Orbit. Deve aiutare a
continuare il progetto dopo `v0.0.1-rc16` distinguendo fatti consolidati,
decisioni prese, ipotesi non ancora provate e prossimi passi ragionevoli.

## Principi permanenti

- Correctness, stabilita, reliability e semplicita vengono prima della performance.
- Orbit resta Python-first: preferire standard library e codice piccolo, leggibile, debuggabile.
- Target primario: CPU-only con Gemma 4 12B tramite native `orbit server`.
- Runtime owns behavior; backend owns inference.
- Non introdurre fix semantici hardcoded nel routing o nel tool loop.
- Guardrail deterministici sono ammessi solo per sicurezza, validazione, bounded retry e diagnostica.
- Non sacrificare correttezza per speedup teorici.
- Benchmark e test prevalgono sulle intuizioni.
- `workdir/` e una fixture pubblica: non toccare o stagiare `workdir/.miktex/` e `workdir/doc/`.
- Non creare tag o release salvo richiesta esplicita.

## Stato release

### RC13

- Focus: diagnostica MTP.
- Aggiunte diagnostiche MTP per throughput, config, timing e validate efficiency.
- MTP e stabile, ma non e risultato throughput-positive in modo robusto su CPU-only.

### RC14

- Focus: diagnostiche KV/final evidence e compact final evidence.
- `cached=4` su final/retry e stato spiegato come divergenza prompt-view `route -> final`, non come bug backend/cache.
- La slim compact final evidence metadata riduce gli evaluated tokens nei `final_from_tool` piccoli.
- `chat_final` multi-card resta uno stop tecnico senza lineage/intent affidabile.

### RC15

- Focus: evidence lineage.
- `EvidenceRecord` include `evidence_sequence`, `tool_call_id`, `user_turn_id`, `produced_by_phase`.
- `producer_model_call_id` resta `null`.
- Nessuna evidence selection o compaction attiva.
- `dual_shell` conferma che una selection `current_turn`-only non e sicura.
- Gli smoke lineage devono usare workdir temporanei puliti, non store persistenti contaminati.

### RC16

- Budget finale dedicato per `system_info`.
- Documentazione CPU-first e MTP opzionale.
- Metadata header per `orbit bench-core`.
- Guidance di profiling e server profile conservativa.
- Download modello draft MTP spostato fuori dal setup base: e opzionale.

## MTP

- MTP e opzionale/sperimentale.
- Non e default nel quick start.
- Non garantisce speedup, specialmente su CPU-only.
- Il draft model MTP va scaricato solo se si vuole testare intenzionalmente MTP.
- `n_max=3` resta il default migliore tra gli esperimenti osservati.
- `target_validate` e compute-bound; il costo dominante e nel graph compute.
- Two-pass validate e shadow runtime sono stati scartati per rischio correctness: mutazioni KV, stato speculative non clonabile in modo sicuro, sampler/KV/frontier cleanup sensibili.
- MTP strict e timeout/cancel recovery restano gate obbligati quando si valida il path MTP.
- Nei test locali MTP usare MTP attivo, mmproj e multimodal quando si valida quel path.

## KV/cache/final budget

- `cached=4` su `route -> final` e atteso: i prompt divergono subito nel system prompt.
- Non inseguire `cached=4` con redesign rischiosi senza nuova evidenza.
- La strada sicura resta ridurre evaluated tokens nei final/retry.
- `final_from_tool` piccolo e stato migliorato con compact evidence metadata.
- `system_info` ha un cap dedicato a 160 token.
- `shell`, `grep_search` e `unknown` piccoli restano a 96 token.
- `/max-tokens` e un limite user-facing; il runtime applica comunque budget interni per fase.

## Evidence lineage

- `user_turn_id` e utile per provenance, non per relevance.
- `tool_call_id` ed `evidence_sequence` sono utili ma non sufficienti per selection.
- `produced_by_phase` e valorizzato solo nei path noti.
- `producer_model_call_id` resta `null`.
- L'esperimento model-guided shadow evidence selection e stato negativo: extra model call troppo costosa su CPU-only, JSON non affidabile, fragilita su `dual_shell`; patch revertita, non usarlo ora.
- `chat_final` multi-card compaction resta stop tecnico.
- `dual_shell` puo richiedere entrambe le card nel retry/final: non ridurre senza lineage/intent piu forte.

## Benchmarking

- `orbit bench-core` e il benchmark pubblico di regressione.
- Il metadata header di `bench-core` e ON di default.
- Usare `--no-metadata` solo quando serve output minimale.
- Il metadata include commit/tag, `base_url`, `workdir`, timeout, `max_tokens`, env selezionate e `/props` backend best-effort.
- Se `/props` non risponde, `backend_props: unavailable` non deve fallire il benchmark.
- Registrare sempre commit/tag, modello, ctx, threads, MTP, tools e prewarm.
- `scripts/suggest-server-profile.sh` e un punto di partenza conservativo, non garanzia di tuning ottimale.
- GPU va misurata tramite backend esterno compatibile, per esempio `llama-server --base-url`, non come performance nativa di `orbit server`.
- Native `orbit server` e CPU-first con `gpu_layers=0`.

## Gate consigliati

- Pre-PR: unit mirati per l'area modificata.
- Sempre: `compileall` mirato e `git diff --check`.
- Full unit solo per pre-release o cambi ampi.
- Se si toccano budget/final: smoke `system_info`.
- Se si tocca `bench_core`: smoke metadata header e `--no-metadata`.
- Evidence lineage smoke: usare workdir temporanei puliti.
- KV/final smoke: `pwd_followup`.
- MTP gate: `simple_chat --mtp-required` con `/props` healthy.
- Recovery gate: timeout/cancel con `shell20`, poi nuovo `simple_chat --mtp-required`.
- Mai usare store persistente per RC smoke di evidence lineage.

## #124, conversation reuse route guidance

- Problema: il router poteva richiamare tool anche per recap, sintesi, ripetizioni o continuazioni di informazioni gia presenti in conversazione.
- Soluzione: aggiunta una regola generale e model-guided solo in `ROUTE_SYSTEM_PROMPT`.
- La regola preferisce `CHAT` quando l'utente chiede recap/summarize/repeat/continue/explain/compare e il contesto esistente e sufficiente.
- I tool restano consentiti per fresh/current, verify/check, nuove informazioni, changed file/state o contesto missing/stale/ambiguous/insufficient.
- File toccati: `src/orbit/runtime/messages.py`, `tests/test_messages.py`.
- Test eseguiti: `PYTHONPATH=src python3 -m unittest tests.test_messages -q`, `python3 -m compileall -q src/orbit/runtime tests`, `git diff --check`.
- Limiti residui: e una guidance di routing, non una garanzia deterministica; non aggiunge cache, TTL, fast path o logica per-tool.
- Stato: lavoro chiuso. Non aggiungere altre patch su conversation reuse senza regressione osservata.

## Ultimi commit principali

- `1d54e9c` Improve route guidance for conversation reuse (#124)
- `3390059` Clarify optional native MTP support (#123)
- `a05a1e9` Add post-RC16 agent guidance (#122)
- `a6133c35` Add release notes for v0.0.1-rc16
- `767ed6e` Document optional MTP model download (#121)
- `400711e` Document bench core metadata and profile guidance (#120)
- `8e830ed` Add bench core metadata header (#119)
- `b700d74` Clarify CPU-first server and MTP guidance (#118)
- `c03533e` Increase system info final budget (#117)
- `91e84e2` Add release notes for v0.0.1-rc15
- `d4991d4` Add user turn lineage to evidence records (#116)
- `d4ae03a` Add evidence lineage diagnostics (#115)

## Prossimi obiettivi suggeriti

1. Fermarsi e usare RC16 come baseline stabile.
2. Eseguire benchmark CPU controllati con `bench-core` metadata.
3. Analizzare l'output `bench-core` per eventuali regressioni o profili migliori.
4. Eseguire uno smoke end-to-end leggero su conversation reuse solo se emerge una regressione o un comportamento ambiguo.
5. Solo se necessario, investigare `producer_model_call_id` runtime-side.
6. Non riaprire evidence selection senza nuovo segnale affidabile di relevance.
7. Non riaprire MTP algorithm tuning senza nuova evidenza upstream o benchmark forte.
8. Valutare piccoli miglioramenti UX/documentazione solo se misurabili, isolati e supportati da test.

## Anti-obiettivi

- Niente rewrite multi-linguaggio.
- Niente hardcoded semantic routing.
- Niente evidence selection `current_turn`-only.
- Niente MTP default.
- Niente promesse GPU sul native server.
- Niente release senza preflight.
- Niente benchmark senza metadata.
