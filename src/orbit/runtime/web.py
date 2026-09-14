from __future__ import annotations

import errno
import socket
import time
from html.parser import HTMLParser
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from orbit.runtime.analysis_network_policy import (
    ANALYSIS_NETWORK_DENIED_REASON,
    network_retrieval_denied,
)


MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT_SECONDS = 10
# Bounded retry for transient provider/network failures only (WEB-SEARCH-RELIABILITY-1).
# A single ECONNRESET/timeout/429/5xx from the search provider is usually transient;
# retry a small, fixed number of times with short backoff, then surface the real error.
SEARCH_MAX_ATTEMPTS = 3
# Backoff (seconds) applied BEFORE the 2nd and 3rd attempts. Never applied to a
# permanent failure, so a permanent error is not materially delayed.
SEARCH_RETRY_BACKOFF_SECONDS = (0.5, 1.0)
# HTTP statuses treated as transient (retryable). All other 4xx are permanent.
_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
# OSError errno values treated as transient at the socket layer.
_TRANSIENT_ERRNOS = frozenset(
    {errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT, errno.EPIPE}
)
DEFAULT_FETCH_TIMEOUT_SECONDS = 10
MAX_FETCH_TIMEOUT_SECONDS = 15
DEFAULT_FETCH_MAX_BYTES = 128_000
MAX_FETCH_MAX_BYTES = 256_000
DEFAULT_FETCH_MAX_TEXT_CHARS = 4_000
_TEXTUAL_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-javascript",
    "image/svg+xml",
)


def search_web(query: str, *, max_results: int = MAX_SEARCH_RESULTS) -> str:
    if network_retrieval_denied():
        # A search is an outbound request; under the static-analysis contract it
        # is refused before any socket is opened. No network occurs.
        return f"error: {ANALYSIS_NETWORK_DENIED_REASON}"
    query = query.strip()
    if not query:
        return "error: search query must be non-empty"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        },
    )
    html: str | None = None
    status = 200
    for attempt in range(1, SEARCH_MAX_ATTEMPTS + 1):
        try:
            status, html = _fetch_search_html(request)
            break
        except OSError as exc:
            # A retry is another attempt at the SAME fixed provider request; it opens
            # no new URL and adds no model call. Only genuinely transient failures are
            # retried, and only while attempts remain; everything else (and the final
            # attempt) surfaces the real error immediately, with the attempt count.
            if attempt < SEARCH_MAX_ATTEMPTS and _is_transient_search_error(exc):
                # Defensive index: stays valid even if SEARCH_MAX_ATTEMPTS is later
                # raised without extending the backoff tuple.
                backoff = SEARCH_RETRY_BACKOFF_SECONDS[min(attempt - 1, len(SEARCH_RETRY_BACKOFF_SECONDS) - 1)]
                time.sleep(backoff)
                continue
            return f"error: web search failed after {attempt} attempt(s): {exc}"
    assert html is not None
    if _is_ddg_challenge(status, html):
        # WEB-SEARCH-PROVIDER-CHALLENGE-1: the provider returned its anti-bot challenge
        # page (HTTP 202 + a duckduckgo.com/anomaly.js form), not search results. This
        # is NOT a legitimate zero-result query, so it must not be reported as an empty
        # success. No clean keyless fallback exists (DDG Lite serves the same challenge;
        # Mojeek gates with a captcha), so this surfaces an informative provider-challenge
        # error. Bounded: the challenge is detected once and never re-fetched in a loop.
        return (
            "error: web search unavailable: the search provider (duckduckgo) returned an "
            "anti-bot challenge instead of results; no results were retrieved"
        )
    results = _parse_duckduckgo_html(html, max_results=max_results)
    if not results:
        return "web_search_results: true\nresults: none"
    lines = ["web_search_results: true", f"query: {query}", "results:"]
    for index, result in enumerate(results, 1):
        lines.extend(
            [
                f"{index}. title: {result['title']}",
                f"   url: {result['url']}",
                f"   snippet: {result['snippet']}",
            ]
        )
    return "\n".join(lines)


# DuckDuckGo anti-bot challenge signature (P3: HTTP status AND a provider-specific body
# marker). The challenge page is served with HTTP 202 and embeds a form posting to
# `duckduckgo.com/anomaly.js`. BOTH are required: the 202 status distinguishes it from a
# normal 200 results page (which may legitimately mention "anomaly.js" in a result URL or
# snippet), and the provider-anchored marker distinguishes it from any other 202. This
# never keys on "0 results" alone.
_DDG_CHALLENGE_STATUS = 202
_DDG_CHALLENGE_MARKER = "duckduckgo.com/anomaly"


def _is_ddg_challenge(status: int, html: str) -> bool:
    return status == _DDG_CHALLENGE_STATUS and _DDG_CHALLENGE_MARKER in html.lower()


def _fetch_search_html(request: Request) -> tuple[int, str]:
    """Perform one search HTTP request, returning (http_status, body). Isolated so the
    retry loop can drive it and tests can stub the network with a single patch point."""
    with urlopen(request, timeout=SEARCH_TIMEOUT_SECONDS) as response:
        status = response.getcode() if hasattr(response, "getcode") else 200
        body = response.read(512_000).decode("utf-8", errors="replace")
        return status, body


def _is_transient_search_error(exc: OSError) -> bool:
    """True only for genuinely transient provider/network failures that are safe to
    retry: connection reset/abort/broken pipe, timeout, temporary DNS failure, and a
    narrow set of HTTP statuses (429 and selected 5xx). Permanent 4xx, permanent DNS,
    connection refused, and any non-network error are NOT retried."""
    if isinstance(exc, HTTPError):
        # HTTPError is a subclass of URLError, so it must be checked first.
        return exc.code in _RETRYABLE_HTTP_STATUSES
    if isinstance(exc, URLError):
        return _transient_reason(exc.reason)
    return _transient_reason(exc)


def _transient_reason(reason: object) -> bool:
    if isinstance(
        reason,
        (TimeoutError, socket.timeout, ConnectionResetError, ConnectionAbortedError, BrokenPipeError),
    ):
        return True
    if isinstance(reason, socket.gaierror):
        # Only a *temporary* name-resolution failure is retryable; a permanent
        # NXDOMAIN/unknown-host is not.
        return reason.errno == getattr(socket, "EAI_AGAIN", object())
    if isinstance(reason, OSError):
        return reason.errno in _TRANSIENT_ERRNOS
    return False


def fetch_url_definition() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a URL and return normalized textual evidence: final URL, HTTP status, content type, title, readable text, "
                "or a real observed fetch failure. Prefer this for explicit read/fetch/explain/summarize/analyze URL requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1},
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_FETCH_TIMEOUT_SECONDS,
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_FETCH_MAX_BYTES,
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    }


def execute_fetch_url(arguments: dict[str, object]) -> str:
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return "error: fetch_url requires a non-empty url"
    if network_retrieval_denied():
        # Execution-time denial: the mandatory second layer. Whatever routed a
        # fetch_url request here -- a stale tool listing, a malformed call, a
        # future miswiring -- no socket is opened and no redirect is followed.
        # Reported as a policy denial (its own status), never a network failure,
        # and never a success: the URL stays a reportable IOC and the caller can
        # truthfully say the remote resource was not retrieved.
        return _format_fetch_failure("policy_denied", url=url.strip(), error=ANALYSIS_NETWORK_DENIED_REASON)
    timeout = _bounded_int(arguments.get("timeout"), default=DEFAULT_FETCH_TIMEOUT_SECONDS, maximum=MAX_FETCH_TIMEOUT_SECONDS)
    max_bytes = _bounded_int(arguments.get("max_bytes"), default=DEFAULT_FETCH_MAX_BYTES, maximum=MAX_FETCH_MAX_BYTES)
    request = Request(
        url.strip(),
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return _format_fetch_response(
                requested_url=url.strip(),
                final_url=response.geturl(),
                status_code=response.getcode(),
                content_type=response.headers.get_content_type() or "",
                encoding=response.headers.get_content_charset() or "utf-8",
                body=response.read(max_bytes + 1),
                max_bytes=max_bytes,
                reason=None,
            )
    except HTTPError as exc:
        body = exc.read(max_bytes + 1)
        return _format_fetch_response(
            requested_url=url.strip(),
            final_url=exc.geturl() or url.strip(),
            status_code=exc.code,
            content_type=exc.headers.get_content_type() if exc.headers is not None else "",
            encoding=(exc.headers.get_content_charset() if exc.headers is not None else None) or "utf-8",
            body=body,
            max_bytes=max_bytes,
            reason=exc.reason if hasattr(exc, "reason") else exc.msg,
        )
    except (socket.timeout, TimeoutError):
        return _format_fetch_failure("timeout", url=url.strip(), error=f"timed out after {timeout}s")
    except URLError as exc:
        reason = exc.reason if getattr(exc, "reason", None) else exc
        if isinstance(reason, socket.timeout):
            return _format_fetch_failure("timeout", url=url.strip(), error=f"timed out after {timeout}s")
        return _format_fetch_failure("network_error", url=url.strip(), error=str(reason))
    except OSError as exc:
        return _format_fetch_failure("network_error", url=url.strip(), error=str(exc))


def fetch_url_result_status(content: str | None) -> str | None:
    if not content:
        return None
    match = re.search(r"^status:\s*(\w+)\s*$", content, flags=re.MULTILINE)
    return match.group(1) if match else None


def fetch_url_result_has_text(content: str | None) -> bool:
    if not content or "text:\n" not in content:
        return False
    _prefix, body = content.split("text:\n", 1)
    return bool(body.strip())


def fetch_url_result_text(content: str | None) -> str | None:
    if not fetch_url_result_has_text(content):
        return None
    assert content is not None
    return content.split("text:\n", 1)[1]


def fetch_url_result_error(content: str | None) -> str | None:
    if not content:
        return None
    match = re.search(r"^error:\s*(.+)$", content, flags=re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def html_to_text(html: str) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(html)
    parser.close()
    return _normalize_text("\n".join(parser.blocks))


def _format_fetch_response(
    *,
    requested_url: str,
    final_url: str,
    status_code: int,
    content_type: str,
    encoding: str,
    body: bytes,
    max_bytes: int,
    reason: object | None,
) -> str:
    text_truncated = len(body) > max_bytes
    if text_truncated:
        body = body[:max_bytes]
    status = "ok" if 200 <= status_code < 400 else "http_error"
    if not _is_textual_content_type(content_type):
        return _format_fetch_result(
            status="unsupported_content" if status == "ok" else status,
            url=requested_url,
            final_url=final_url,
            http_status=status_code,
            content_type=content_type,
            encoding=encoding,
            title=None,
            text=None,
            text_truncated=text_truncated,
            error=f"unsupported content type: {content_type or 'unknown'}" if status == "ok" else _http_error_text(status_code, reason),
        )
    decoded = body.decode(encoding or "utf-8", errors="replace")
    title = _extract_html_title(decoded) if _looks_like_html(decoded) else None
    readable = html_to_text(decoded) if _looks_like_html(decoded) else _normalize_text(decoded)
    readable, text_truncated = _truncate_fetch_text(readable, already_truncated=text_truncated)
    if status != "ok":
        return _format_fetch_result(
            status="http_error",
            url=requested_url,
            final_url=final_url,
            http_status=status_code,
            content_type=content_type,
            encoding=encoding,
            title=title,
            text=readable or None,
            text_truncated=text_truncated,
            error=_http_error_text(status_code, reason),
        )
    if not readable.strip():
        return _format_fetch_result(
            status="empty_body",
            url=requested_url,
            final_url=final_url,
            http_status=status_code,
            content_type=content_type,
            encoding=encoding,
            title=title,
            text=None,
            text_truncated=text_truncated,
            error="response body was empty after text extraction",
        )
    return _format_fetch_result(
        status="ok",
        url=requested_url,
        final_url=final_url,
        http_status=status_code,
        content_type=content_type,
        encoding=encoding,
        title=title,
        text=readable,
        text_truncated=text_truncated,
        error=None,
    )


def _format_fetch_failure(status: str, *, url: str, error: str) -> str:
    return _format_fetch_result(
        status=status,
        url=url,
        final_url=url,
        http_status=None,
        content_type=None,
        encoding=None,
        title=None,
        text=None,
        text_truncated=False,
        error=error,
    )


def _format_fetch_result(
    *,
    status: str,
    url: str,
    final_url: str,
    http_status: int | None,
    content_type: str | None,
    encoding: str | None,
    title: str | None,
    text: str | None,
    text_truncated: bool,
    error: str | None,
) -> str:
    lines = [
        "url_fetch: true",
        f"status: {status}",
        f"url: {url}",
        f"final_url: {final_url}",
        f"http_status: {http_status if http_status is not None else 'null'}",
        f"content_type: {content_type or 'null'}",
        f"encoding: {encoding or 'null'}",
        f"title: {title or 'null'}",
        f"text_truncated: {'true' if text_truncated else 'false'}",
    ]
    if error:
        lines.append(f"error: {error}")
    if text:
        lines.extend(["text:", text])
    return "\n".join(lines)


def _is_textual_content_type(content_type: str | None) -> bool:
    if not content_type:
        return True
    normalized = content_type.split(";", 1)[0].strip().lower()
    return any(normalized.startswith(prefix) or normalized == prefix for prefix in _TEXTUAL_CONTENT_TYPES)


def _looks_like_html(text: str) -> bool:
    lowered = text[:2048].lower()
    return "<html" in lowered or "<body" in lowered or "<title" in lowered or "<!doctype html" in lowered


def _extract_html_title(text: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = _normalize_inline_text(_strip_tags(match.group(1)))
    return title or None


def _http_error_text(status_code: int, reason: object | None) -> str:
    suffix = f": {reason}" if reason else ""
    return f"HTTP {status_code}{suffix}"


def _bounded_int(value: object, *, default: int, maximum: int) -> int:
    if not isinstance(value, int):
        return default
    if value <= 0:
        return default
    return min(value, maximum)


def _truncate_fetch_text(text: str, *, already_truncated: bool) -> tuple[str, bool]:
    if len(text) <= DEFAULT_FETCH_MAX_TEXT_CHARS:
        return text, already_truncated
    clipped = text[:DEFAULT_FETCH_MAX_TEXT_CHARS].rstrip()
    last_break = max(clipped.rfind("\n"), clipped.rfind(". "), clipped.rfind(" "))
    if last_break >= DEFAULT_FETCH_MAX_TEXT_CHARS // 2:
        clipped = clipped[:last_break].rstrip()
    return clipped, True


def _parse_duckduckgo_html(html: str, *, max_results: int) -> list[dict[str, str]]:
    blocks = re.split(r'class=["\']result(?:\s|["\'])', html)
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in blocks[1:]:
        title_match = re.search(r'class=["\']result__a["\'][^>]*href=["\'](?P<url>[^"\']+)["\'][^>]*>(?P<title>.*?)</a>', block, re.DOTALL)
        if not title_match:
            continue
        url = _clean_result_url(_strip_tags(title_match.group("url")))
        title = _normalize_inline_text(_strip_tags(title_match.group("title")))
        snippet_match = re.search(r'class=["\']result__snippet["\'][^>]*>(?P<snippet>.*?)</a>', block, re.DOTALL)
        snippet = _normalize_inline_text(_strip_tags(snippet_match.group("snippet"))) if snippet_match else ""
        if not url or not title or url in seen:
            continue
        seen.add(url)
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _clean_result_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path == "/l/" and parsed.query:
        match = re.search(r"(?:^|&)uddg=([^&]+)", parsed.query)
        if match:
            return unquote(match.group(1))
    return url


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value)


def _normalize_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._current: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in {"p", "div", "section", "article", "header", "footer", "li", "br", "h1", "h2", "h3", "h4"}:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in {"p", "div", "section", "article", "header", "footer", "li", "h1", "h2", "h3", "h4"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._current.append(text)

    def close(self) -> None:
        self._flush()
        super().close()

    def _flush(self) -> None:
        if not self._current:
            return
        block = re.sub(r"\s+", " ", " ".join(self._current)).strip()
        if block:
            self.blocks.append(block)
        self._current = []


def _normalize_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
