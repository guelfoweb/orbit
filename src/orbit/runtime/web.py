from __future__ import annotations

from html.parser import HTMLParser
import re
from urllib.parse import quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT_SECONDS = 10


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


def html_to_text(html: str) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(html)
    parser.close()
    return _normalize_text("\n".join(parser.blocks))


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
