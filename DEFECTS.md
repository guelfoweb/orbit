# DEFECTS.md

Compact defect log for `orbit`.

Record format:

- `Symptom`: observable behavior.
- `Cause`: likely or confirmed cause.
- `Fix`: mitigation or correction applied.
- `Example`: useful prompt or case.

Rules:

- Add records only for real defects in runtime, routing, tool loop, or tools.
- Prefer small localized fixes covered by tests.
- Record the observed behavior, not only the fix idea.

## Tracked Defects

### 1. Restored sessions kept obsolete system prompts
- Symptom: old sessions continued following outdated instructions.
- Cause: the first `system` message was restored verbatim.
- Fix: regenerate the system prompt on restore while preserving the rest of history.
- Example: old sessions still claimed generic web search did not exist.

### 2. `fetch_url` was misused as a search engine
- Symptom: generic web queries generated guessed Google, Bing, or Wikipedia URLs.
- Cause: there was no dedicated search tool and the `fetch_url` contract was too weak.
- Fix: added `search_web`; restricted `fetch_url` to explicit known URLs.
- Example: `search online for information about Dante Alighieri`.

### 3. DuckDuckGo HTML search used the wrong HTTP method
- Symptom: generic web search returned weak or empty results.
- Cause: the HTML endpoint expected form-style requests.
- Fix: switched to bounded form-encoded requests with structured output.
- Example: `https://html.duckduckgo.com/html/`.

### 4. Thinking-capable mode crashed on unsupported Gemma4 profiles
- Symptom: `does not support thinking (400)` aborted the client.
- Cause: no fallback when `think` was requested on an incompatible Gemma4 profile.
- Fix: catch the error, warn, and retry without thinking.
- Example: `/think on` with a non-thinking Gemma4 profile.

### 5. Pseudo-JSON tool calls were printed instead of executed
- Symptom: the model emitted a textual tool call and Orbit treated it as normal text.
- Cause: fallback parser was too strict.
- Fix: parse fenced JSON and simple relaxed tool-call shapes safely.
- Example: `{"name": list_files, "arguments": {"path": "."}}`.

### 6. Noisy directories inflated context and caused `list_files` loops
- Symptom: repeated workspace listings over `.venv`, `.git`, `node_modules`, and caches.
- Cause: directory listing was too noisy for the target Gemma4 profile.
- Fix: ignore common noise directories and bound listing output.
- Example: `what files are in this folder?`.

### 7. Similar but distinct tool calls were blocked too early
- Symptom: related but different web queries were treated as loops.
- Cause: near-duplicate matching was too aggressive.
- Fix: block only genuinely identical calls after light normalization.
- Example: `Dante Alighieri poet` vs `Dante Alighieri poet Italy`.

### 8. Gemma4 profiles without `tools` capability failed unclearly
- Symptom: some Gemma4 profiles never emitted native tool calls.
- Cause: backend/model capability differences.
- Fix: inspect metadata and degrade to chat-only with a warning.
- Example: startup with a Gemma4 profile that does not advertise `tools`.

### 9. Too many tools confused the target Gemma4 profile
- Symptom: Gemma4 picked the wrong tool for simple requests.
- Cause: all tools were exposed in every turn.
- Fix: two-stage tool routing by category: filesystem, write, shell, web.
- Example: simple filesystem prompts seeing web/write tools.

### 10. Context budget was too static
- Symptom: compaction triggered too late on small contexts.
- Cause: hardcoded thresholds did not use effective context size.
- Fix: context budget engine with soft/hard pressure and context-window awareness.
- Example: CPU-bound small-context sessions slowing sharply.

### 11. Anti-loop logic was mixed into turn policy
- Symptom: repeated-call behavior was hard to extend and test.
- Cause: history, signatures, and retry prompts were coupled.
- Fix: extracted loop guard logic.
- Example: `ToolCallRecord` matching isolated from turn classification.

### 12. Current factual questions routed to local tools
- Symptom: weather/news/person questions went to `bash` or local tools.
- Cause: weak signal for online factual lookup.
- Fix: strengthened `current_factual_lookup` routing and default skill guidance.
- Example: `what is the weather in Rome today?`.

### 13. Routing was too category-centric
- Symptom: unrelated workflows collapsed into the same broad category.
- Cause: missing intent layer.
- Fix: introduced `intent_router` and intent-to-category mapping.
- Example: `codebase_inspection`, `text_document_analysis`, `binary_or_pdf_analysis`.

### 14. Codebase inspection prioritized weak files
- Symptom: the model read docs/config before implementation modules.
- Cause: weak default skill priority hints.
- Fix: prefer entrypoints, core logic, router, registry, runtime, and relevant tests.
- Example: inspection starting from `README.md` instead of `core/agent.py`.

### 15. Codebase inspection ignored explicit subtrees
- Symptom: the model inspected repository root instead of the requested subfolder.
- Cause: subtree focus was not enforced strongly enough.
- Fix: strengthen subtree and hidden/output de-prioritization.
- Example: `analyze the code in the orbit folder`.

### 16. Skill path distracted the model
- Symptom: the model inspected `SKILL.md` instead of the project.
- Cause: absolute skill path was included in the prompt.
- Fix: remove skill path from the system prompt.
- Example: reading `builtins/orbit-default/SKILL.md` during code inspection.

### 17. Repeated `read_file` chunks escaped loop detection
- Symptom: the model kept reading the same file with different `start_line` values.
- Cause: matching used the full signature, not the stable path.
- Fix: path-level repeated-read guard after enough sampling.
- Example: repeated reads of `compact.py`.

### 18. Architecture findings were descriptive, not critical
- Symptom: code review described modules instead of surfacing risks.
- Cause: review guidance was too weak.
- Fix: strengthen critique guidance around coupling, brittle flows, and missing tests.
- Example: "AgentLoop orchestrates flow" instead of a risk finding.

### 19. Multiline JSON tool-call strings failed parsing
- Symptom: `write_file` calls with multiline content were printed.
- Cause: strict JSON parsing rejected literal newlines in strings.
- Fix: try `json.loads(..., strict=False)` before relaxed parsing.
- Example: multiline `content` in a fenced JSON block.

### 20. File edit turns kept editing after completion
- Symptom: `append_file` or `write_file` repeated after a useful edit.
- Cause: no edit-completed stop policy.
- Fix: add file-edit finalization and retry prompts for same-path loops.
- Example: create a report and append a final section.

### 21. Markdown document reads routed as codebase inspection
- Symptom: simple `.md` reads caused read loops.
- Cause: broad matching on words like `repo` inside `report`.
- Fix: tighter matching and local finalization after successful `read_file`.
- Example: `show the content of REPORT.md`.

### 22. Binary analysis accepted wrong tools
- Symptom: binary requests were treated as text and fallback parser accepted off-subset tools.
- Cause: weak binary signals and missing subset validation.
- Fix: stronger binary tokens and validation against exposed tool subset.
- Example: `analyze the binary in this directory`.

### 23. Binary tasks invented paths or read metadata
- Symptom: `read_file` ran on invented binary paths or unrelated docs/config.
- Cause: no discovery-first rule.
- Fix: seed `list_files` and block unconfirmed binary reads.
- Example: `read_file ./binary` before discovery.

### 24. Current directory requests got generic access refusals
- Symptom: the model claimed it could not access the local directory.
- Cause: no minimal workspace discovery seed.
- Fix: seed `list_files` and retry/finalize from existing evidence.
- Example: `what does this directory contain?`.

### 25. Existing files could be overwritten before being read
- Symptom: `write_file` overwrote existing unread files.
- Cause: missing read-before-write guard.
- Fix: block first overwrite and require `read_file` first.
- Example: `fix this file`.

### 26. Pure read-only tools were repeated unnecessarily
- Symptom: identical `read_file`, `list_files`, `search_web`, or `fetch_url` calls repeated.
- Cause: no small session cache for pure calls.
- Fix: add session dedup for read-only tools.
- Example: repeated identical `search_web` query.

### 27. Failing tools kept polluting routing
- Symptom: the model kept choosing a repeatedly failing tool.
- Cause: no per-session trust decay.
- Fix: de-prioritize or drop tools after repeated failures.
- Example: repeated failing `search_web`.

### 28. Binary formats without dot extensions were missed
- Symptom: words like `apk`, `pdf`, or `dex` routed as text.
- Cause: router recognized `.apk` better than `apk`.
- Fix: recognize binary format tokens without leading dots.
- Example: `analyze the apk file in this workspace`.

### 29. Archive containers were treated as raw blobs
- Symptom: APK/ZIP/JAR files were read or string-scanned directly.
- Cause: no container-specific strategy.
- Fix: redirect to listing tools such as `unzip -l` or `zipinfo -1`.
- Example: `strings app.apk | grep activity`.

### 30. Binary tasks without filename repeated listings
- Symptom: after `list_files`, the model listed again instead of choosing a candidate.
- Cause: seeded binary prompt did not surface candidates.
- Fix: derive likely candidates locally and present them explicitly.
- Example: `analyze the apk in this directory`.

### 31. Embedded archive members were treated as real files
- Symptom: `read_file classes.dex` was attempted before extraction.
- Cause: no distinction between archive members and filesystem files.
- Fix: block reads on embedded members and tolerate benign `SIGPIPE` with useful stdout.
- Example: `unzip -l app.apk | head -n 10`.

### 32. Read-then-write workflows ended as pseudo-failures
- Symptom: file was correct but the model kept rereading or returned placeholder text.
- Cause: guarded writes counted too early as completed edits.
- Fix: finalization after useful edits and better placeholder repair.
- Example: update `REPORT.md` after reading `README.md`.

### 33. Web lookup could reopen the same page or fake tool output
- Symptom: after search/fetch success, the model repeated web tools or emitted fake `<tool_response>`.
- Cause: no strong guard after concrete web evidence existed.
- Fix: block redundant web calls and finalize from collected evidence.
- Example: weather lookup after a successful fetch.

### 34. Active analysis skills contaminated chitchat
- Symptom: a greeting could trigger skill bootstrap and writes.
- Cause: bootstrap keyed too much on active skill, not intent.
- Fix: explicit `chitchat` intent and no tools/bootstrap for greetings.
- Example: `hello` with an analysis skill active.

### 35. Metadata defaults and lightweight model profiles regressed
- Symptom: tests failed on `ModelMetadata`; lightweight Gemma4 profiles started with too-heavy defaults.
- Cause: `parameter_size` became required and runtime did not recognize small quantized profiles.
- Fix: make metadata optional and tune lightweight defaults.
- Example: `gemma4:e2b-fast-t6-c8k`.

### 36. Default agent path was too heavy
- Symptom: ordinary turns paid for long prompts and skill context.
- Cause: default path accumulated operational prompt, skill, and transient guardrails.
- Fix: shorten default skill/prompt and use a faster agentic path.
- Example: quick local inspection should stay close to direct Ollama latency.

### 37. Projected context pressure was ignored
- Symptom: compaction happened after the next user prompt had already made the turn expensive.
- Cause: pressure was estimated only on current session state.
- Fix: evaluate context pressure with pending user input.
- Example: long sessions where the next turn suddenly slowed down.

### 38. Overflow compaction kept too much recent context
- Symptom: the first compaction barely reduced prompt size.
- Cause: recent window did not depend on overflow amount.
- Fix: shrink recent window based on projected overflow.
- Example: CPU-bound sessions near context limit.

### 39. Simple turns paid the full operational prompt
- Symptom: greetings and basic questions were slow compared with `ollama run`.
- Cause: no minimal chat path.
- Fix: minimal prompt for chitchat and general knowledge.
- Example: `hi` or `why is the sky blue?`.

### 40. Safe base64 pipelines were blocked
- Symptom: base64 decode failed with `unsupported pipe filter: base64`.
- Cause: `base64` was missing from benign pipeline filter allowlist.
- Fix: allow `base64` in safe bounded pipelines and add tests.
- Example: `echo -n "Y2lhbw==" | base64 -d`.

### 41. Small explicit writes could timeout
- Symptom: simple file creation prompts waited on the model and timed out.
- Cause: no local fast path for explicit small create requests.
- Fix: deterministic preflight for simple `write_file` and read-summary-write workflows.
- Example: `Create TODO.md with two bullets`.

### 42. Italian directory listings were not finalized locally
- Symptom: Italian listing prompts could hit model loops or max loops.
- Cause: missing common Italian directory discovery patterns.
- Fix: expand bilingual listing triggers and local finalization.
- Example: `Which files are in this directory?` and the Italian equivalent.

### 43. Generic online requests could hallucinate after web routing
- Symptom: online lookup routed to web but answered without real results.
- Cause: generic web prompts were not seeded strongly enough.
- Fix: seed `search_web` for broader online lookup requests and finalize from structured evidence.
- Example: `search online for information about Dante Alighieri`.

### 44. "Important files" inspection wasted loops
- Symptom: the model read many files instead of answering from listing evidence.
- Cause: no local path for important-file ranking.
- Fix: recursive listing seed and local priority-file finalization.
- Example: `tell me the 5 most important files to read first`.

### 45. Long document summaries reflected only the first chunk
- Symptom: summaries missed content appearing later in the file.
- Cause: explicit summary path used only the most recent bounded read.
- Fix: progressive bounded multi-chunk reads and merged evidence.
- Example: `summarize LONG_EN.md`.

### 46. Huge-file summaries became sampled quotes
- Symptom: long novels produced titles and fragments instead of a real summary.
- Cause: model saw extractive sampled evidence without enough synthesis guidance.
- Fix: chunk notes with spans/focus/keywords and summary prompt over bounded notes.
- Example: `read promessi_sposi.txt and summarize it`.

### 47. Explicit PDF requests stayed in generic binary triage
- Symptom: PDF read/summarize prompts did not try text extraction first.
- Cause: no bounded explicit PDF extraction path.
- Fix: `pdftotext` head extraction with `strings` fallback.
- Example: `summarize docs/report.pdf`.

### 48. Code review prompts were weakly recognized
- Symptom: review requests produced generic or hallucinated output.
- Cause: codebase inspection lacked review-specific reads and finalization.
- Fix: seed central implementation reads and add cautious local findings.
- Example: `review this codebase for risks`.

### 49. Review missed risks beyond the first chunk
- Symptom: TODOs or broad `except:` blocks later in a file were missed.
- Cause: review looked mostly at first chunks.
- Fix: progressive bounded reads for priority review files.
- Example: risk appears after line 200.

### 50. Review findings were too per-file
- Symptom: coupling risks across `agent`, `runtime`, `router`, and `registry` were missed.
- Cause: findings were derived mostly from single-file patterns.
- Fix: add cross-file integration findings.
- Example: review of orchestration code.

### 51. Create/delete filesystem operations lacked bounded tools
- Symptom: folder/file create/delete prompts did not close cleanly.
- Cause: missing `make_directory` and `delete_path`.
- Fix: add bounded tools and deterministic inference for create/remove.
- Example: `create a directory named test_workspace`.

### 52. Colloquial PDF paths with spaces were overcaptured
- Symptom: nearby words were included in the filename.
- Cause: PDF path token parser lacked enough stopwords.
- Fix: add bilingual stopwords around PDF object names.
- Example: `read this file sample report.pdf`.

### 53. PDF read hints contaminated text file summaries
- Symptom: `Summarize README.md` returned raw content.
- Cause: PDF read/open hints were placed in shared text show hints.
- Fix: split PDF-only read hints from text-document hints.
- Example: `Summarize README.md`.

### 54. Machine configuration prompts were slow
- Symptom: model improvised several shell commands for local hardware info.
- Cause: no bounded system-info path.
- Fix: deterministic OS/kernel/CPU/RAM collection via safe commands.
- Example: `what is the configuration of this machine?`.

### 55. Tool-concept prompts collided
- Symptom: arithmetic or simulated-tool prompts hit the wrong conceptual branch.
- Cause: close conceptual guardrails had weak priority.
- Fix: order arithmetic/simulation cases before generic tool explanations.
- Example: `calculate 12345/345 then multiply`.

### 56. Config-file prompts were confused with machine configuration
- Symptom: config file reads routed as system-info prompts.
- Cause: `configuration` alone was treated as machine signal.
- Fix: require machine/system/device context for system-info.
- Example: `read config.json`.

### 57. English documentation searches went local
- Symptom: online documentation search read local `README.md`.
- Cause: `documentation/docs` were not online-lookup signals when combined with search verbs.
- Fix: include documentation tokens in online lookup detection.
- Example: `Search for documentation on tool calling with Ollama`.

### 58. Deterministic preflights overrode capable tool choice
- Symptom: model-first flows could not reuse tool evidence naturally.
- Cause: local seeds ran before capable models could plan.
- Fix: model-first profile for `gemma4:e2b`.
- Example: machine config prompt followed by CPU follow-up.

### 59. Model-first machine prompts went to workspace tools
- Symptom: local resource questions called `list_files`.
- Cause: prompt did not separate machine vs workspace tool categories strongly enough.
- Fix: strengthen model-first tool category guidance and retry correction.
- Example: `check local resources of this machine`.

### 60. Model-first replies were too verbose after tools
- Symptom: simple listings became broad project analyses.
- Cause: post-tool finalization guidance was weak.
- Fix: add lightweight post-tool guidance by intent class.
- Example: `what does this working directory contain?`.

### 61. Workspace discovery used overly broad recursive listings
- Symptom: simple directory prompts produced huge recursive output.
- Cause: initial scope was not bounded for workspace discovery.
- Fix: redirect recursive top-level discovery to bounded non-recursive listing.
- Example: `what does this working directory contain?`.

### 62. Compaction summary was not operational enough
- Symptom: compacted memory mixed requests, findings, and tool activity linearly.
- Cause: fixed thresholds and generic summary structure.
- Fix: context-window-aware thresholds and structured `Working memory` / `Durable memory`.
- Example: `/compact` on sessions rich in tool output.

### 63. Transient prompts and verbose tool schemas inflated prefill
- Symptom: simple model-first turns remained slow.
- Cause: transient system prompts persisted and tool definitions were too verbose.
- Fix: prune transient system messages and compact model-first tool definitions.
- Example: `hi`, `list workspace`.

### 64. Large tool results could saturate context
- Symptom: large `list_files` or `read_file` results made the next model call slow or impossible.
- Cause: compaction only ran before the turn, not before appending large tool output.
- Fix: project pressure before appending tool results; hard-compact first when needed.
- Example: large listings in long sessions.

### 65. Vision attached only the first explicit image
- Symptom: image comparison saw only one image.
- Cause: resolver returned only first image path.
- Fix: collect and deduplicate all explicit image paths.
- Example: `compare cmp-blue.png and cmp-red.png`.

### 66. Raw images could crash the Ollama runner
- Symptom: multimodal comparison returned runner error 500.
- Cause: fragile raw image inputs such as alpha PNGs or large heterogeneous files.
- Fix: normalize images with Pillow: RGB, alpha flattening, conservative resize, PNG re-encode.
- Example: `compare cmp-blue.png and vision-test.png`.

### 67. Long pasted text broke prompt readability
- Symptom: base64 or long multiline paste made REPL history unusable.
- Cause: terminal rendered full input after submit.
- Fix: visual-only collapse to `[text N chars]`; real prompt remains unchanged.
- Example: long base64 decode prompt.

### 68. Short creative prompts used the heavy ambiguous path
- Symptom: `make up a story in 5 lines` took too long.
- Cause: creative no-file prompts routed as ambiguous.
- Fix: route creative text without save/path signals as chitchat.
- Example: `write a poem in 4 lines`.

### 69. Client-side Ollama env did not configure systemd server
- Symptom: exported `OLLAMA_NUM_PARALLEL` before `orbit` had no effect.
- Cause: Ollama server was already running under systemd.
- Fix: document server-side env configuration.
- Example: `env OLLAMA_NUM_PARALLEL=1 orbit ...`.

### 70. Exact-answer prompts missed the minimal path
- Symptom: `Say exactly: OK` used a large prompt.
- Cause: exact-answer variants were incomplete.
- Fix: add English and Italian exact-answer hints.
- Example: `Say exactly: OK`.

### 71. Explicit workspace listings still called the model
- Symptom: `list all files and directories` fetched listing then asked the model to format it.
- Cause: local directory finalizer missed workspace/list/directories signals.
- Fix: finalize explicit bounded listings locally.
- Example: `list all files and directories in the current workspace`.

### 72. Simple filesystem metadata did unnecessary model work
- Symptom: size/mtime/newest-file requests went through model loops.
- Cause: no model-first seed/finalizer for metadata.
- Fix: direct bounded `stat_path` and local formatting.
- Example: `what is the size and modified time of README.md?`.

### 73. Code review attempted `read_file` without a path
- Symptom: review read a file, then called `read_file` with missing `path`.
- Cause: invalid structured call was executed despite existing evidence.
- Fix: finalize locally from valid evidence instead of executing invalid call.
- Example: `review agent.py for vulnerabilities and security issues`.

### 74. Multi-evidence prompts lost sources or chose wrong side effects
- Symptom: no-modification prompts routed to edit; local+web prompts claimed unread evidence.
- Cause: missing side-effect negation, presence-check, and local+web finalizer.
- Fix: detect explicit no-write constraints, add presence check, and finalize from actual local/web evidence.
- Example: `Read summary.txt, then search online...`.

### 75. Workspace security scan read too many chunks
- Symptom: security scan read multiple chunks of the same large file without concrete security evidence.
- Cause: scan reused deeper code-review chunk policy.
- Fix: limit security scan to one chunk per candidate.
- Example: workspace security prompt over `agent.py`.

### 76. Static malware triage was misrouted as filesystem metadata
- Symptom: APK/static-analysis prompts that mentioned metadata did not collect sample evidence.
- Cause: generic metadata routing won over binary/static sample intent.
- Fix: detect static triage signals before metadata routing and seed bounded evidence collection.
- Example: `collect initial metadata for malware/Questionario BNL.apk`.

### 77. Static sample evidence required unnecessary model synthesis
- Symptom: hash/type/container evidence was collected but the model could still loop or summarize weakly.
- Cause: no local finalizer for sufficient static-analysis evidence.
- Fix: finalize concise evidence locally from `file`, hashes, and bounded archive listings.
- Example: `Perform static analysis for all malware samples in the malware directory`.

### 78. Metadata answers omitted provenance when asked
- Symptom: newest-file prompts answered the file and timestamp but not how the answer was determined.
- Cause: local `stat_path` formatter ignored method/provenance wording.
- Fix: add bounded provenance text when the prompt asks how the metadata was determined.
- Example: `what is the newest file ... and how did you determine it?`.

### 79. Generic web search could surface sponsored redirects or over-compress results
- Symptom: generic search returned weak synthesis or DuckDuckGo sponsored `/y.js` redirects.
- Cause: model summarized search results too aggressively and the extractor did not skip ad redirects.
- Fix: provide bounded local search-result summaries for simple information searches and filter ad redirects at extraction time.
- Example: `search online for information about Dante Alighieri`.

### 80. Static malware skill stopped at hashes or entered slow scaffolding loops
- Symptom: malware/static-analysis prompts either stopped after file type and hashes or spent long model loops creating/listing the case directory.
- Cause: initial evidence was treated as sufficient, while the full skill prompt was too heavy for the small model before useful reverse-engineering evidence was gathered.
- Fix: distinguish evidence-only requests from static/reverse analysis, seed bounded manifest/container/script/string inspection, and locally summarize bounded reverse-inspection evidence unless a case/report update is explicitly requested.
- Example: `Perform static reverse engineering for the samples in the malware directory`.

### 81. Multi-image comparison ignored later images or text
- Symptom: image comparisons could describe only the first image or miss text visible in another image.
- Cause: passing multiple images in one Ollama message was unstable for the target model/runtime.
- Fix: inspect each image separately, request visible text and subject, then synthesize the comparison from per-image evidence.
- Example: `compare two images: images/vision-test-1.png and images/vision-test-2.jpg`.

### 82. Audio prompts tried unsupported raw audio paths
- Symptom: long audio attachments crashed or timed out the runner, while `audio`/`audios` fields were ignored.
- Cause: Ollama/Gemma4 handled short WAV audio only through the attachment path and was unstable with long raw audio.
- Fix: require `ffmpeg`/`ffprobe`, normalize to WAV PCM 16 kHz mono, split into 5s chunks, transcribe chunks separately, then synthesize.
- Example: `transcribe audio/voice-sample-16k-mono.wav`.

### 83. Security discussion prompts triggered local malware triage
- Symptom: conversational prompts mentioning malware analysis, C2, IoC, or APKs could trigger `list_files` and bounded static triage.
- Cause: binary/static routing treated security terminology as operational intent.
- Fix: classify discursive and learning prompts as chat, add a model YES/NO intent gate for ambiguous high-impact binary routes, and keep explicit sample/path prompts operational.
- Example: `ask me something about malware analysis, C2 or IoC`.

### 84. Web/search discussion prompts triggered web lookup
- Symptom: prompts discussing web search as a concept could expose web tools instead of answering conversationally.
- Cause: broad `web`, `search`, and `about` hints were enough to classify the turn as current factual lookup.
- Fix: classify discursive web/search statements as chat, keep explicit/current lookup phrases operational, and add a model YES/NO gate for weak web lookup prompts on the target model.
- Example: `what do you think about web search in LLMs?`.

### 85. Discursive tool words exposed shell or filesystem tools
- Symptom: prompts mentioning `grep`, `find`, `file systems`, `base64`, or `tools` conceptually could expose shell/filesystem/write/web tools.
- Cause: command and path hints were matched as operational intent even when the prompt asked for explanation or unsupported external action.
- Fix: route discursive command/file/encoding prompts to knowledge/chat, make path-extension detection stricter, recognize CPU count as bounded machine inspection, and gate fully ambiguous prompts before exposing tools on Gemma.
- Example: `show me how grep works`.

### 86. PDF document analysis shared the binary-analysis intent
- Symptom: normal PDF reading/summarization and binary/static analysis shared the same `binary_or_pdf_analysis` route.
- Cause: PDF and binary files both needed bounded shell/filesystem handling, so they were historically grouped under one intent.
- Fix: split routing into `pdf_analysis` for document extraction and `binary_analysis` for static/container inspection, while keeping a compatibility helper for older guardrails.
- Example: `Summarize docs/Project Overview.pdf.`.

### 87. Code review synthesis drifted into generic security advice
- Symptom: explicit code-review prompts could return plausible but unanchored findings such as generic prompt-injection or tool-execution risks.
- Cause: after `read_file`, model-first synthesis was still allowed to answer without concrete line or pattern evidence.
- Fix: add bounded local security-pattern extraction and replace unanchored generic security reviews with evidence-based local findings when possible.
- Example: `review agent.py for vulnerabilities and security issues`.

### 88. Long document summaries over-weighted noisy sampled fragments
- Symptom: summaries of very large narrative documents could focus on isolated quotes, questions, footnotes, or weak chunk fragments.
- Cause: sampled chunk notes lacked a document map and candidate scoring allowed low-information literary fragments.
- Fix: add `document_map`, synthesis guidance, better Italian stopwords, footnote filtering, and narrative-aware candidate scoring for long sampled summaries.
- Example: `Read promessi_sposi.txt, summarize it in exactly 4 lines`.

## Recurring Guidelines

- Keep the base prompt short.
- Move behavior into bounded guardrails and runtime logic, not permanent prompt bulk.
- Expose the smallest useful tool subset for each turn.
- Prefer local finalization when tool evidence is sufficient.
- Compact early and aggressively for long text or large contexts.
- Keep web fetching bounded; do not add browser automation unless the project scope changes.
