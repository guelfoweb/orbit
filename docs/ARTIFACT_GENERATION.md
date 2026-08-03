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
4. A stopped, non-empty result of at most 64 KiB remains pending in bounded
   memory. No destination file or parent directory exists yet.
5. The next model round exposes only `verify_artifact`. The model selects the
   exact pending path and one bounded check.
6. A passing check and atomic publication complete one mutation epoch. The
   final response remains model-authored from exact path, byte-count, SHA-256,
   overwrite, created-parent, and verification evidence.

`verify_artifact` is an ephemeral internal capability. It is not part of the
normal tools-on registry and cannot be selected outside a pending artifact
request. Ordinary chat and non-artifact tool routes gain no model call and do
not receive the artifact schemas.

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

## Publication And Parent Directories

The path stays confined to the active workdir. Absolute paths, traversal,
control characters, symlink components, symlink destinations, FIFOs, devices,
and other non-regular targets fail closed. Existing files require explicit
`overwrite=true` and are attested before and during publication.

Missing parents are part of the same atomic operation. Orbit records exactly
which parents were absent when the request was prepared. After content and
model-selected verification pass, it builds those parents and the file in one
private directory below the deepest existing parent, fsyncs the private tree,
and publishes the top missing parent with one no-replace rename. If another
process creates that parent first, publication fails and preserves the
concurrent tree. Failure cleanup removes only Orbit's private empty hierarchy;
it never removes pre-existing or concurrently created paths.

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

Publishing a new parent tree still requires filesystem support for atomic
no-replace directory rename, and replacing an existing file requires atomic
exchange. Orbit fails closed when those primitives are unavailable; it does not
fall back to a pathname check followed by a racy ordinary rename. Creation in
an existing directory remains supported through the no-replace hard-link
fallback described above.

Parent creation is not a successful mutation by itself. The mutation epoch
advances exactly once, only after the selected check passes and publication is
complete.

## Lifecycle

`finish_reason=length`, malformed verification, cancellation, timeout, reset,
generation error, UTF-8 failure, size failure, path race, or check failure
publishes nothing. Pending state is turn-local and process-local. Content is
buffered before any temporary filesystem object is created, so cancellation or
restart during the long generation phase leaves no file, parent, or stale
temporary artifact.

The final atomic filesystem section is intentionally short and synchronous.
Normal failures clean its private objects. As with any userspace atomic-write
protocol, an uncatchable process or power loss in the narrow interval before a
private staging directory is renamed can leave a hidden `.orbit-artifact-*`
entry; it never exposes a partial destination or removes unrelated data.

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
HTML/CSS/JavaScript, Markdown, JSON, and Python. Five clean standalone Snake
runs produced the same 3,058-byte artifact and SHA-256, selected the generic
text-integrity check, and stopped normally. Separately, validation extracted
the script from the 10,290-byte self-contained browser artifact and confirmed
its syntax with `node --check`; that external check is not an Orbit verifier
capability.

A 2,048-token content probe ended at `length` and correctly published nothing.
The retained 4,096-token bound completed the measured browser artifact.
Cancellation during content generation and an actual server restart left no
file, parent, or temporary entry. Timeout behavior is covered by injected
backend timeouts; the CLI HTTP timeout is inactivity-based and does not expire
while tokens continue to stream.

Measured CPU wall time is descriptive. It ranged from about two minutes for
small JSON and Markdown artifacts to about nine minutes for the 10 KiB browser
game on the validation host.
