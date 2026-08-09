# Atomic Text Artifact Generation

## Problem

Non-trivial files do not fit reliably inside a normal structured tool call.
Qwen 3.6 may spend the tool output budget inside XML, JSON, shell quoting, or a
heredoc and leave an incomplete envelope. Orbit must not execute or repair that
truncated output. A separate failure mode is premature completion after the
model creates and inspects only the destination directory.

## Protocol

Orbit keeps the workflow model-driven while separating the small control
decision from the potentially long file body:

1. The route model selects `write_artifact` and supplies one relative path,
   the overwrite decision, and whether missing parents may be created.
2. The canonical contract validates that bounded request before generation.
3. One native content phase receives the original request and validated target
   and emits UTF-8 file content only. It has no tools and does not parse its
   output as JSON, XML, shell, a heredoc, or a tool call.
4. A stopped, non-empty result of at most 64 KiB is validated as UTF-8 and
   published to the exact target with an atomic file operation.
5. Publication evidence records the exact path, byte count, SHA-256,
   overwrite state, explicit `publication_action` (`created` or `replaced`),
   and any parents created by this request.
6. The next model round exposes only `verify_artifact`. The capability is
   intrinsically bound to the exact published path, and the model selects one
   bounded read-only check without supplying a replacement path.
7. The final response remains model-authored from separate publication and
   verification evidence.

`verify_artifact` is an ephemeral internal capability. It is not part of the
normal tools-on registry and cannot be selected outside a pending artifact
request. Ordinary chat and non-artifact tool routes gain no model call and do
not receive the artifact schemas.

Publication and verification are intentionally separate. A verification
failure does not undo or replace an already published target. Orbit reports
the file as published but unverified and does not claim that its content passed
the selected check.

## Checks

The model may select `text_integrity` for UTF-8, byte-count, and SHA-256
verification, or `content` when the bounded body must also be projected as
evidence. These are the same checks for every text format. Runtime does not
select a parser from an extension, execute content, or interpret the body as a
program, markup, configuration, or tool protocol.

The runtime validates only the selected structural check. It does not infer a
file type, choose a check, repair content, or judge whether the artifact solves
the user's semantic task. Format-specific correctness checks in validation,
such as parsing source or configuration text, run outside Orbit after the
artifact workflow completes.

The verified Qwen3-Coder profile uses a model-specific reversible JSON-string
transport for this content phase because its generic chat response wrapped file
bodies in a Markdown presentation fence. The backend pre-opens the string,
constrains only its structural grammar, decodes the model-generated value with
strict UTF-8, and rejects malformed or incomplete framing. It does not trim,
normalize, repair, or semantically alter the decoded value. This does not
change the shared generative artifact contract or publication lifecycle. See
`docs/QWEN3_CODER_COMPATIBILITY.md`.

The JSON string is transport, not an exact-copy contract. The model remains
responsible for the decoded semantic content. Orbit guarantees that valid
generated string characters and escapes are decoded reversibly; it does not
guarantee that a generative model reproduces externally authoritative Unicode
or newline bytes on request.

## Publication And Parent Directories

The path stays confined to the active workdir. Absolute paths, traversal,
control characters, symlink components, symlink destinations, FIFOs, devices,
and other non-regular targets fail closed. Existing files require explicit
`overwrite=true` and are attested before and during publication.

Missing parents require the model-selected `create_parents=true` argument.
After content validation, Orbit creates each absent directory relative to an
attested descriptor and records its exact inode. Parent directories are not
published or rolled back as one tree. On failure Orbit attempts `rmdir` only
for an exact directory created by this request and only while it remains empty.
A directory containing concurrent files or directories stays at its original
visible path. Pre-existing directories are never removed, renamed, or moved
into private Orbit state.

For an existing parent, Orbit uses an unnamed or private same-filesystem file
and atomically links or exchanges it into place. Overwrite races are detected;
the previous or concurrent destination is preserved when the requested
identity can no longer be proven. Filesystems that reject Linux `O_TMPFILE`
with `EOPNOTSUPP` or `EINVAL` during open or anonymous-link publication fall
back to a mode-`0600`, exclusive, randomly named private file in that same
directory. If that filesystem also lacks `RENAME_NOREPLACE` for regular files,
Orbit uses an atomic no-replace hard link for the private inode and removes only
the private name. Fsync and post-publication identity/hash attestation remain
unchanged.

The destination file, rather than a shared parent tree, is the atomic unit.
New files use no-replace link or rename semantics and replacing an existing file
requires atomic exchange. Orbit fails closed when those primitives are
unavailable; it does not fall back to a pathname check followed by a racy
ordinary rename.

Parent creation is not a successful artifact publication by itself. The
mutation epoch advances exactly once after atomic file publication; the
workflow remains structurally incomplete until the model selects and passes a
read-only verification.

## Lifecycle

`finish_reason=length`, cancellation or timeout during content generation,
generation error, UTF-8 failure, size failure, or a pre-commit path race
publishes no destination file. Content is buffered before any temporary
filesystem object is created, so interruption during the long generation phase
leaves no file, parent, or private artifact entry. Once atomic publication
succeeds, a later malformed, cancelled, timed-out, or failed verification does
not mutate or remove the published file.

The final atomic filesystem section is intentionally short and synchronous.
Normal failures clean its private objects. Named private files are paired with
bounded manifests in `.orbit-artifact-state`. The manifest binds the random
name, parent, device, inode, uid, boot identity, PID, and process start time.
Startup cleanup stays inside the active workdir, never follows symlinks, keeps
entries owned by a live process, and removes an entry only after positively
proving that its owner process is inactive and every manifest, ownership, mode,
link-count, and inode check still matches. Unknown owner state is always
preserved; stale age alone never authorizes deletion. Malformed, replaced,
linked, or otherwise ambiguous entries are preserved for manual inspection.

Zero residual private state after an uncatchable crash or power loss cannot be
proved for every filesystem. A crash between private-file creation and manifest
registration can leave an untracked private name. A crash during hard-link
publication or overwrite exchange can leave an entry whose identity is
ambiguous or may contain displaced user data; recovery preserves it rather than
guessing. A crash after explicitly authorized parent creation but before file
publication can also leave empty visible directories because Orbit cannot later
distinguish them safely from directories retained or reused by another process.
These narrow cases may require manual inspection. They never justify deleting a
user-visible target or moving concurrent content.

## Bounds And Limits

- one UTF-8 file per request;
- maximum generated content: 64 KiB;
- dedicated content budget: 4,096 output tokens;
- one model-selected verification action;
- native Orbit backend only;
- no semantic repair, chunking, map-reduce, hidden retry, or deterministic
  task content;
- multi-file requests require separate model-selected artifact requests and
  are not planned by the runtime.

Route selection remains model-owned. The dedicated protocol applies only when
the model selects `write_artifact`; one measured very small JavaScript request
selected chat and returned prose without creating a file. Orbit does not force
or infer an artifact route to conceal that model-selection limit.

The 4,096-token phase is dedicated to artifact content and does not enlarge
route, normal tool, chat, or final budgets. CPU generation of a large artifact
can still take minutes. Atomic publication improves reliability, not model
generation speed.

## Validation

The Qwen 3.6 35B-A3B Q4_K_M validation covered standalone JavaScript, one-file
HTML/CSS/JavaScript, Markdown, JSON, Python, and plain text. Five clean runs of
the original standalone Snake request produced the same 4,590-byte artifact
with SHA-256
`3dac570c5e1c7b24bd304bc776651eea0901db2a423bd68b00550301f723ab45`,
selected the generic text-integrity check, and stopped normally. Each run used
four model calls, published before the read-only verification, passed
`node --check`, and left no private artifact entry. Separately, validation
extracted the script from the 10,290-byte self-contained browser artifact and
confirmed its syntax with `node --check`; that external check is not an Orbit
verifier capability.

A 2,048-token content probe ended at `length` and correctly published nothing.
The retained 4,096-token bound completed the measured browser artifact.
Cancellation during content generation and an actual server restart left no
file, parent, or temporary entry. Timeout behavior is covered by injected
backend timeouts; the CLI HTTP timeout is inactivity-based and does not expire
while tokens continue to stream.

Measured CPU wall time is descriptive. It ranged from about two minutes for
small JSON and Markdown artifacts to about nine minutes for the 10 KiB browser
game on the validation host.
