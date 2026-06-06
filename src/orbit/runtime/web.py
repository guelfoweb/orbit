from __future__ import annotations

from html.parser import HTMLParser
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


FETCH_TIMEOUT_SECONDS = 15
SEARCH_TIMEOUT_SECONDS = 15
MAX_FETCH_BYTES = 512 * 1024
MAX_SEARCH_BYTES = 256 * 1024
MAX_FETCH_EXTRACTED_CHARS = 256_000
DEFAULT_FETCH_CHUNK_CHARS = 6_000
MAX_FETCH_CHUNK_CHARS = 24_000
MAX_FETCH_CHUNK_CALLS_PER_TURN = 3
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 8
VALID_SEARCH_TIMELIMITS = {"d", "w", "m", "y"}
SITE_PATTERN = re.compile(r"^(?!-)(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,63}$")
SITE_QUERY_PATTERN = re.compile(r"(?i)(?:^|\s)site:([A-Za-z0-9.-]+\.[A-Za-z]{2,63})(?=\s|$)")
DUCKDUCKGO_HTML_URL = "https://duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

TEXT_CONTENT_TYPES = {
    "application/json",
    "application/ld+json",
    "application/rss+xml",
    "application/xml",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/xml",
}


def search_web(
    query: Any,
    *,
    max_results: Any = DEFAULT_SEARCH_RESULTS,
    site: Any = None,
    timelimit: Any = None,
) -> str:
    if not isinstance(query, str) or not query.strip():
        return "error: query must be a non-empty string"
    if not isinstance(max_results, int):
        return "error: max_results must be an integer"
    if max_results < 1 or max_results > MAX_SEARCH_RESULTS:
        return f"error: max_results must be between 1 and {MAX_SEARCH_RESULTS}"
    if site is not None and not _valid_search_site(site):
        return "error: site must be a bare domain like example.com"
    if timelimit is not None and timelimit not in VALID_SEARCH_TIMELIMITS:
        return "error: timelimit must be one of: d, w, m, y"

    query = query.strip()
    query, extracted_site = _extract_site_filter(query)
    site = site if site is not None else extracted_site
    if site is not None and not _valid_search_site(site):
        return "error: site must be a bare domain like example.com"
    effective_query = f"site:{site.strip().lower()} {query}" if isinstance(site, str) and site.strip() else query
    params = {"q": effective_query}
    if timelimit:
        params["df"] = timelimit
    url = f"{DUCKDUCKGO_HTML_URL}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=SEARCH_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", 200)
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(MAX_SEARCH_BYTES + 1)
    except HTTPError as exc:
        return f"error: HTTP {exc.code} while searching web"
    except URLError as exc:
        return f"error: cannot search web: {exc.reason}"
    except TimeoutError:
        return f"error: search_web timed out after {SEARCH_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return f"error: cannot search web: {exc}"

    if content_type not in {"text/html", "application/xhtml+xml"}:
        return f"error: unsupported search response content type: {content_type}"
    try:
        html = raw[:MAX_SEARCH_BYTES].decode(charset, errors="replace")
    except LookupError:
        html = raw[:MAX_SEARCH_BYTES].decode("utf-8", errors="replace")
    results = parse_search_results(html, max_results=max_results)
    lines = [
        f"query: {query}",
        f"effective_query: {effective_query}",
        f"status: {status}",
        f"results: {len(results)}",
    ]
    if site:
        lines.append(f"site: {site.strip().lower()}")
    if timelimit:
        lines.append(f"timelimit: {timelimit}")
    if len(raw) > MAX_SEARCH_BYTES:
        lines.append(f"download_truncated: true at {MAX_SEARCH_BYTES} bytes")
    if not results:
        lines.append("error: no structured results extracted")
        return "\n".join(lines)
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"{index}. title: {result['title']}",
                f"   url: {result['url']}",
                f"   snippet: {result['snippet'] or '(none)'}",
            ]
        )
    return "\n".join(lines)


def fetch_url(url: Any, *, chunk_index: Any = None, chunk_chars: Any = DEFAULT_FETCH_CHUNK_CHARS) -> str:
    if not isinstance(url, str) or not url.strip():
        return "error: url must be a non-empty string"
    if chunk_index is not None and (not isinstance(chunk_index, int) or chunk_index < 0):
        return "error: chunk_index must be a non-negative integer"
    if not isinstance(chunk_chars, int) or chunk_chars <= 0:
        return "error: chunk_chars must be a positive integer"
    if chunk_chars > MAX_FETCH_CHUNK_CHARS:
        return f"error: chunk_chars too large: {chunk_chars}, max {MAX_FETCH_CHUNK_CHARS}"

    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "error: fetch_url accepts only explicit http/https URLs"

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", 200)
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(MAX_FETCH_BYTES + 1)
    except HTTPError as exc:
        return f"error: HTTP {exc.code} while fetching URL"
    except URLError as exc:
        return f"error: cannot fetch URL: {exc.reason}"
    except TimeoutError:
        return f"error: fetch_url timed out after {FETCH_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return f"error: cannot fetch URL: {exc}"

    truncated = len(raw) > MAX_FETCH_BYTES
    raw = raw[:MAX_FETCH_BYTES]
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")

    if content_type == "text/html" or content_type == "application/xhtml+xml":
        body = _html_to_text(text)
    elif content_type in TEXT_CONTENT_TYPES or content_type.startswith("text/"):
        body = _normalize_text(text)
    else:
        return "\n".join(
            [
                f"url: {final_url}",
                f"status: {status}",
                f"content_type: {content_type}",
                "error: unsupported content type for text extraction",
            ]
        )

    extracted_truncated = len(body) > MAX_FETCH_EXTRACTED_CHARS
    if extracted_truncated:
        body = body[:MAX_FETCH_EXTRACTED_CHARS].rstrip()

    chunk = _chunk_text(body, chunk_index=chunk_index or 0, chunk_chars=chunk_chars)
    if isinstance(chunk, str):
        return chunk
    start, end, total_chunks, chunk_text = chunk

    lines = [
        f"url: {final_url}",
        f"status: {status}",
        f"content_type: {content_type}",
        f"chunk_index: {chunk_index or 0}",
        f"total_chunks: {total_chunks}",
        f"chars: {start}-{end} of {len(body)}",
    ]
    if extracted_truncated:
        lines.append(f"extracted_truncated: true at {MAX_FETCH_EXTRACTED_CHARS} chars")
    if truncated:
        lines.append(f"download_truncated: true at {MAX_FETCH_BYTES} bytes")
    lines.extend(["text:", chunk_text if chunk_text else "(empty document)"])
    return "\n".join(lines)


def parse_search_results(html: str, *, max_results: int = DEFAULT_SEARCH_RESULTS) -> list[dict[str, str]]:
    parser = _DuckDuckGoHTMLParser(max_results=max_results)
    parser.feed(html)
    parser.close()
    return parser.results


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self, *, max_results: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_results = max_results
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs if key}
        classes = set(values.get("class", "").split())
        if tag == "a" and "result__a" in classes and len(self.results) < self.max_results:
            self._store_current()
            self._current = {"title": "", "url": _clean_search_url(values.get("href", "")), "snippet": ""}
            self._capture = "title"
            self._parts = []
            return
        if self._current is not None and tag in {"a", "div"} and "result__snippet" in classes:
            self._capture = "snippet"
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._current is None or self._capture is None:
            return
        if self._capture == "title" and tag == "a":
            self._current["title"] = _normalize_text(" ".join(self._parts))
            self._capture = None
            self._parts = []
            return
        if self._capture == "snippet" and tag in {"a", "div"}:
            self._current["snippet"] = _normalize_text(" ".join(self._parts))
            self._capture = None
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture and data.strip():
            self._parts.append(data)

    def close(self) -> None:
        self._store_current()
        super().close()

    def _store_current(self) -> None:
        if self._current is None:
            return
        if len(self.results) >= self.max_results:
            self._current = None
            return
        if self._current["title"] and _is_search_result_url(self._current["url"]):
            if all(existing["url"] != self._current["url"] for existing in self.results):
                self.results.append(dict(self._current))
        self._current = None


class _ReadableHTMLParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "form", "nav", "footer", "header"}
    _BLOCK_TAGS = {"title", "h1", "h2", "h3", "p", "li", "blockquote", "pre", "td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._active_block: str | None = None
        self._active_parts: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "meta":
            self._handle_meta(attrs)
            return
        if tag in self._BLOCK_TAGS:
            self._flush_block()
            self._active_block = tag
            self._active_parts = []
        elif tag == "br" and self._active_block:
            self._active_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == self._active_block:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._active_block:
            return
        if data.strip():
            self._active_parts.append(data)

    def close(self) -> None:
        self._flush_block()
        super().close()

    def _handle_meta(self, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value for key, value in attrs if key and value}
        name = (values.get("name") or values.get("property") or "").lower()
        if name in {"description", "og:description", "twitter:description"} and values.get("content"):
            self.blocks.append(_normalize_text(values["content"]))

    def _flush_block(self) -> None:
        if not self._active_block:
            return
        block = _normalize_text(" ".join(self._active_parts))
        if block:
            self.blocks.append(f"- {block}" if self._active_block == "li" else block)
        self._active_block = None
        self._active_parts = []


def _html_to_text(html: str) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(html)
    parser.close()
    return _dedupe_blocks(parser.blocks)


def _dedupe_blocks(blocks: list[str]) -> str:
    seen: set[str] = set()
    cleaned: list[str] = []
    for block in blocks:
        normalized = _normalize_text(block)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return "\n".join(cleaned)


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _clean_search_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return url


def _valid_search_site(site: Any) -> bool:
    if not isinstance(site, str) or not site.strip():
        return False
    value = site.strip().lower()
    if "://" in value or "/" in value or ":" in value or " " in value:
        return False
    return SITE_PATTERN.fullmatch(value) is not None


def _extract_site_filter(query: str) -> tuple[str, str | None]:
    match = SITE_QUERY_PATTERN.search(query)
    if not match:
        return query, None
    site = match.group(1).strip().lower()
    cleaned = SITE_QUERY_PATTERN.sub(" ", query, count=1)
    return _normalize_text(cleaned), site


def _is_search_result_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.endswith("/y.js"):
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _chunk_text(text: str, *, chunk_index: int, chunk_chars: int) -> tuple[int, int, int, str] | str:
    total_chunks = max(1, (len(text) + chunk_chars - 1) // chunk_chars)
    if chunk_index >= total_chunks:
        return f"error: chunk_index out of range: {chunk_index}, total_chunks {total_chunks}"
    start = chunk_index * chunk_chars
    end = min(start + chunk_chars, len(text))
    return start, end, total_chunks, text[start:end]
