from __future__ import annotations

import socket
from html.parser import HTMLParser
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT_SECONDS = 10
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
    try:
        with urlopen(request, timeout=SEARCH_TIMEOUT_SECONDS) as response:
            html = response.read(512_000).decode("utf-8", errors="replace")
    except OSError as exc:
        return f"error: web search failed: {exc}"
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
                    "url": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["url"],
            },
        },
    }


def execute_fetch_url(arguments: dict[str, object]) -> str:
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return "error: fetch_url requires a non-empty url"
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
