# Download recovery

Rerun the same `orbit download <repo>/<file.gguf>` command after interruption.
Single-file downloads and split shards use the same transfer path. The model
store layout and model qualification are unchanged.

- Received bytes persist in `<filename>.part`, including after Ctrl-C, SIGTERM,
  connection reset or timeout. An exclusive destination lock prevents concurrent
  Orbit writers; the operating system releases it when a process exits.
- A nonempty partial requests `Range: bytes=N-`. Resume requires HTTP 206 with
  matching start, end, total and segment length. A saved strong ETag is sent in
  `If-Range`; an explicitly conflicting validator is rejected.
- If the server returns HTTP 200, Orbit restarts from byte zero into
  `.part.restart`, preserving the original partial. An interrupted replacement
  is resumed first on the next attempt or invocation. If the server ignores
  Range again, the response must match that replacement's existing prefix before
  extending it. A second incompatible representation fails closed and keeps both
  files. This path can require space for the old partial plus the new full file.
- Socket operations have a 30-second inactivity timeout. Reads persist available
  bytes without waiting to fill a large buffer. Resets, short responses and
  transient HTTP 408/429/500/502/503/504 responses receive at most four attempts
  total, separated by 1, 2 and 4 seconds. A permanent 404, inconsistent range,
  local write error or cancellation is not retried.
- Publication requires a known size and exactly that many bytes on disk. A
  complete partial may be finalized on HTTP 416 only when its declared total
  matches. Unknown sizes, encoded representations and inconsistent responses
  fail closed. Rename is atomic within the destination directory; incomplete
  data is never published under the final filename. Obsolete partials are
  removed only after successful publication.

Timeouts measure socket inactivity, not a total time limit for a large file.
Operating-system DNS resolution is outside that timeout. There is no infinite
retry loop. A server continually delivering small amounts of data may take a
long time; Ctrl-C preserves the partial.

Transfer size checks are not a cryptographic content checksum. Without a strong
ETag, resume relies on the server preserving the ranged representation. Split
GGUF header/set validation remains in place; existing final-file reuse retains
its previous behavior. An invalid partial is preserved for diagnosis rather than
silently deleted. Do not edit partials while an Orbit download owns the lock.

The transfer follows the HTTP [Range and If-Range contract](https://www.rfc-editor.org/rfc/rfc9110.html).
Recovery tests use local HTTP fixtures and subprocess interruptions; they do not
download model weights.
