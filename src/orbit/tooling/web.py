from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import math
import re
from typing import Any
from urllib import error, request
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, unquote

from .common import ToolError, coerce_int


MAX_FETCH_TEXT_CHARS = 12_000
MAX_FETCH_LINKS = 25
MAX_SEARCH_RESULTS = 10
MAX_FETCH_QUERY_MATCHES = 8
DEFAULT_FETCH_QUERY_CONTEXT_CHARS = 260
DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
BROWSER_LIKE_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
LINK_RE = re.compile(r"""<a\s+[^>]*href=["']([^"'#]+)["']""", re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
HIGHLIGHT_PATTERNS = (
    re.compile(r".{0,60}temperatura minima.{0,120}?massima.{0,80}?(?:°|&deg;|&#176;)\s*C.{0,40}", re.IGNORECASE),
    re.compile(r".{0,60}temperatura massima.{0,120}?minima.{0,80}?(?:°|&deg;|&#176;)\s*C.{0,40}", re.IGNORECASE),
    re.compile(r".{0,40}(?:sereno|nuvoloso|pioggia|temporale|sole).{0,120}?(?:°|&deg;|&#176;)\s*C.{0,40}", re.IGNORECASE),
)
RESULT_LINK_RE = re.compile(
    r"""<a[^>]*class=["'][^"']*result__a[^"']*["'][^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)
RESULT_BLOCK_RE = re.compile(
    r"""<div[^>]*class=["'][^"']*result[^"']*["'][^>]*>(.*?)</div>\s*</div>""",
    re.IGNORECASE | re.DOTALL,
)
RESULT_SNIPPET_RE = re.compile(
    r"""<(?:a|div)[^>]*class=["'][^"']*result__snippet[^"']*["'][^>]*>(.*?)</(?:a|div)>""",
    re.IGNORECASE | re.DOTALL,
)
RESULT_URL_RE = re.compile(
    r"""<a[^>]*class=["'][^"']*result__url[^"']*["'][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)


def web_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": (
                    "Search the web for an open-ended query and return bounded results with title, URL, and snippet. "
                    "Use this for generic online research when you do not already have a concrete URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": (
                    "Fetch one explicit web page URL and return text, title, final URL, and a bounded link list. "
                    "Use this only when you already have a concrete http/https URL. "
                    "Use query/query_mode to check whether the page mentions a term or discusses a concept without reading the whole page. "
                    "Use start_char to continue a long page in chunks. "
                    "Do not use it as a general web search tool and do not guess search-engine or encyclopedia URLs from a name alone."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "start_char": {"type": "integer", "minimum": 0},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": MAX_FETCH_TEXT_CHARS},
                        "max_links": {"type": "integer", "minimum": 0, "maximum": MAX_FETCH_LINKS},
                        "query": {"type": "string"},
                        "query_mode": {"type": "string", "enum": ["literal", "concept"]},
                        "max_matches": {"type": "integer", "minimum": 1, "maximum": MAX_FETCH_QUERY_MATCHES},
                        "context_chars": {"type": "integer", "minimum": 80, "maximum": 800},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
                    },
                    "required": ["url"],
                },
            },
        }
    ]


@dataclass
class WebTools:
    search_endpoint: str = DEFAULT_SEARCH_ENDPOINT

    def search_web(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query is required")
        max_results = min(MAX_SEARCH_RESULTS, max(1, coerce_int(arguments.get("max_results"), 5)))
        timeout = min(120, max(1, coerce_int(arguments.get("timeout"), 20)))
        html = self._request_text(
            url=self.search_endpoint,
            timeout=timeout,
            form_data={"q": query.strip()},
        )
        results = self._extract_search_results(html, max_results=max_results)
        return {
            "ok": True,
            "query": query.strip(),
            "provider": "duckduckgo-html",
            "results": results,
        }

    def fetch_url(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_url = arguments.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise ToolError("url is required")
        parsed = urlparse(raw_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ToolError("url must be a valid http or https URL")
        start_char = max(0, coerce_int(arguments.get("start_char"), 0))
        max_chars = min(MAX_FETCH_TEXT_CHARS, max(200, coerce_int(arguments.get("max_chars"), 8000)))
        max_links = min(MAX_FETCH_LINKS, max(0, coerce_int(arguments.get("max_links"), 10)))
        query = arguments.get("query")
        query = query.strip() if isinstance(query, str) and query.strip() else None
        query_mode = arguments.get("query_mode")
        query_mode = query_mode if query_mode in {"literal", "concept"} else "literal"
        max_matches = min(MAX_FETCH_QUERY_MATCHES, max(1, coerce_int(arguments.get("max_matches"), 5)))
        context_chars = min(800, max(80, coerce_int(arguments.get("context_chars"), DEFAULT_FETCH_QUERY_CONTEXT_CHARS)))
        timeout = min(120, max(1, coerce_int(arguments.get("timeout"), 20)))
        final_url, status_code, content_type, text = self._fetch_page(raw_url, timeout=timeout)
        full_text = self._extract_text(text)
        total_chars = len(full_text)
        matches = self._search_text(full_text, query=query, mode=query_mode, max_matches=max_matches, context_chars=context_chars) if query else []
        if query:
            start_char = 0
            chunk_text = "\n\n".join(item["context"] for item in matches)
            end_char = min(total_chars, len(chunk_text))
            highlights = []
            truncated = False
        else:
            if start_char > total_chars:
                start_char = total_chars
            end_char = min(total_chars, start_char + max_chars)
            chunk_text = full_text[start_char:end_char]
            highlights = self._extract_highlights(chunk_text)
            truncated = end_char < total_chars
        chunk_count = max(1, math.ceil(total_chars / max_chars)) if total_chars else 1
        chunk_index = 1 if not total_chars else (start_char // max_chars) + 1
        result = {
            "ok": True,
            "url": raw_url,
            "final_url": final_url,
            "status_code": status_code,
            "content_type": content_type,
            "title": self._extract_title(text),
            "text": chunk_text,
            "highlights": highlights,
            "links": self._extract_links(text, final_url, max_links),
            "start_char": start_char,
            "end_char": end_char,
            "total_chars": total_chars,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "next_start_char": end_char if truncated else None,
            "has_more": truncated,
            "truncated": truncated,
        }
        if query:
            result.update(
                {
                    "query": query,
                    "query_mode": query_mode,
                    "match_count": len(matches),
                    "matches": matches,
                    "has_query_matches": bool(matches),
                }
            )
        return result

    def _fetch_page(
        self,
        raw_url: str,
        *,
        timeout: int,
        form_data: dict[str, str] | None = None,
    ) -> tuple[str, int | None, str | None, str]:
        data = None
        headers = {
            "User-Agent": BROWSER_LIKE_USER_AGENT,
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        if form_data:
            data = urlencode(form_data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = request.Request(
            raw_url,
            data=data,
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                final_url = response.geturl()
                status_code = getattr(response, "status", None)
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read()
        except error.HTTPError as exc:
            raise ToolError(f"http error: {exc.code}") from exc
        except error.URLError as exc:
            raise ToolError(f"connection failed: {exc.reason}") from exc
        try:
            text = body.decode(charset, errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        return final_url, status_code, content_type, text

    def _request_text(self, url: str, *, timeout: int, form_data: dict[str, str] | None = None) -> str:
        _, _, _, text = self._fetch_page(url, timeout=timeout, form_data=form_data)
        return text

    @classmethod
    def _extract_search_results(cls, html: str, *, max_results: int) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for block in RESULT_BLOCK_RE.findall(html):
            link_match = RESULT_LINK_RE.search(block)
            if not link_match:
                continue
            raw_href = unescape(link_match.group(1)).strip()
            url = cls._normalize_result_url(raw_href)
            if not url or url in seen:
                continue
            seen.add(url)
            title = cls._clean_fragment(link_match.group(2))
            snippet_match = RESULT_SNIPPET_RE.search(block)
            snippet = cls._clean_fragment(snippet_match.group(1)) if snippet_match else ""
            display_match = RESULT_URL_RE.search(block)
            display_url = cls._clean_fragment(display_match.group(1)) if display_match else ""
            result = {
                "title": title or url,
                "url": url,
                "snippet": snippet,
            }
            if display_url:
                result["display_url"] = display_url
            results.append(result)
            if len(results) >= max_results:
                break
        if results:
            return results
        for href, title_html in RESULT_LINK_RE.findall(html):
            url = cls._normalize_result_url(unescape(href).strip())
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                {
                    "title": cls._clean_fragment(title_html) or url,
                    "url": url,
                    "snippet": "",
                }
            )
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _normalize_result_url(raw_url: str) -> str | None:
        if not raw_url:
            return None
        parsed = urlparse(raw_url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/y.js"):
                return None
            if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                target = parse_qs(parsed.query).get("uddg")
                if target:
                    decoded = unquote(target[0]).strip()
                    nested = urlparse(decoded)
                    if nested.scheme in {"http", "https"} and nested.netloc:
                        return decoded
            return raw_url
        return None

    @staticmethod
    def _extract_title(text: str) -> str | None:
        match = TITLE_RE.search(text)
        if not match:
            return None
        title = WHITESPACE_RE.sub(" ", unescape(match.group(1))).strip()
        return title or None

    @staticmethod
    def _extract_links(text: str, base_url: str, max_links: int) -> list[str]:
        if max_links <= 0:
            return []
        links: list[str] = []
        seen: set[str] = set()
        for href in LINK_RE.findall(text):
            resolved = urljoin(base_url, unescape(href).strip())
            parsed = urlparse(resolved)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or resolved in seen:
                continue
            seen.add(resolved)
            links.append(resolved)
            if len(links) >= max_links:
                break
        return links

    @staticmethod
    def _extract_text(text: str) -> str:
        without_scripts = SCRIPT_STYLE_RE.sub(" ", text)
        without_tags = TAG_RE.sub(" ", without_scripts)
        return WHITESPACE_RE.sub(" ", unescape(without_tags)).strip()

    @staticmethod
    def _extract_highlights(text: str, *, limit: int = 5) -> list[str]:
        if not text:
            return []
        highlights: list[str] = []
        seen: set[str] = set()
        for pattern in HIGHLIGHT_PATTERNS:
            for match in pattern.findall(text):
                cleaned = WHITESPACE_RE.sub(" ", unescape(match)).strip(" -")
                if len(cleaned) < 20 or cleaned in seen:
                    continue
                seen.add(cleaned)
                highlights.append(cleaned)
                if len(highlights) >= limit:
                    return highlights
        return highlights

    @classmethod
    def _search_text(
        cls,
        text: str,
        *,
        query: str,
        mode: str,
        max_matches: int,
        context_chars: int,
    ) -> list[dict[str, Any]]:
        if not text.strip() or not query.strip():
            return []
        if mode == "concept":
            return cls._search_text_by_concept(text, query=query, max_matches=max_matches, context_chars=context_chars)
        return cls._search_text_literal(text, query=query, max_matches=max_matches, context_chars=context_chars)

    @staticmethod
    def _search_text_literal(text: str, *, query: str, max_matches: int, context_chars: int) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for match in pattern.finditer(text):
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            context = text[start:end].strip()
            if start > 0:
                context = "..." + context
            if end < len(text):
                context += "..."
            matches.append(
                {
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "term": text[match.start():match.end()],
                    "context": context,
                }
            )
            if len(matches) >= max_matches:
                break
        return matches

    @classmethod
    def _search_text_by_concept(cls, text: str, *, query: str, max_matches: int, context_chars: int) -> list[dict[str, Any]]:
        concepts = cls._query_concepts(query)
        if len(concepts) > 1:
            return cls._search_text_by_concepts(
                text,
                concepts=concepts,
                max_matches=max_matches,
                context_chars=context_chars,
            )
        keywords = cls._query_keywords(query)
        if not keywords:
            return cls._search_text_literal(text, query=query, max_matches=max_matches, context_chars=context_chars)
        return cls._search_text_by_keywords(
            text,
            keywords=keywords,
            max_matches=max_matches,
            context_chars=context_chars,
        )

    @classmethod
    def _search_text_by_concepts(
        cls,
        text: str,
        *,
        concepts: list[str],
        max_matches: int,
        context_chars: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_contexts: set[str] = set()
        per_concept_limit = max(1, min(2, max_matches // max(1, len(concepts)) + 1))
        for concept in concepts:
            keywords = cls._query_keywords(concept)
            if not keywords:
                continue
            for item in cls._search_text_by_keywords(
                text,
                keywords=keywords,
                max_matches=per_concept_limit,
                context_chars=context_chars,
            ):
                context = str(item.get("context") or "")
                if not context or context in seen_contexts:
                    continue
                seen_contexts.add(context)
                enriched = dict(item)
                enriched["query_part"] = concept
                results.append(enriched)
                if len(results) >= max_matches:
                    return results
        return results

    @staticmethod
    def _query_concepts(query: str) -> list[str]:
        parts = [
            part.strip(" .:;!?")
            for part in re.split(r"\s*(?:,|\band\b|\bor\b|\be\b|\bo\b)\s*", query, flags=re.IGNORECASE)
            if part.strip(" .:;!?")
        ]
        return parts if len(parts) > 1 else []

    @staticmethod
    def _search_text_by_keywords(
        text: str,
        *,
        keywords: list[str],
        max_matches: int,
        context_chars: int,
    ) -> list[dict[str, Any]]:
        if not keywords:
            return []
        candidates: list[tuple[int, int, int, str, list[str]]] = []
        for match in re.finditer(r"[^.!?;:]{40,900}(?:[.!?;:]|$)", text):
            paragraph = match.group(0).strip()
            lowered = paragraph.lower()
            hits = [keyword for keyword in keywords if keyword in lowered]
            if not hits:
                continue
            score = len(set(hits))
            candidates.append((score, -len(paragraph), match.start(), paragraph, sorted(set(hits))))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        results: list[dict[str, Any]] = []
        seen_contexts: set[str] = set()
        for score, _length, start, paragraph, hits in candidates:
            context = paragraph[: context_chars * 2].strip()
            if len(paragraph) > len(context):
                context = context.rstrip() + "..."
            if context in seen_contexts:
                continue
            seen_contexts.add(context)
            results.append(
                {
                    "start_char": start,
                    "end_char": start + len(paragraph),
                    "score": score,
                    "keywords": hits,
                    "context": context,
                }
            )
            if len(results) >= max_matches:
                break
        return results

    @staticmethod
    def _query_keywords(query: str) -> list[str]:
        expansions = {
            "artificial": ("intelligenza", "artificiale"),
            "intelligence": ("intelligenza", "artificiale"),
            "dignity": ("dignità", "dignita"),
            "freedom": ("libertà", "liberta"),
            "human": ("umano", "umana", "persona"),
            "humans": ("umano", "umana", "persona"),
            "worth": ("dignità", "dignita", "persona", "dignity"),
            "free": ("libertà", "liberta", "freedom"),
            "will": ("libertà", "liberta", "freedom", "coscienza"),
            "system": ("sistema", "sistemi"),
            "systems": ("sistema", "sistemi"),
            "technology": ("tecnica", "tecnologia"),
            "digital": ("digitale",),
            "justice": ("giustizia",),
            "work": ("lavoro",),
            "peace": ("pace",),
            "war": ("guerra",),
            "truth": ("verità", "verita"),
            "family": ("famiglia",),
            "dignità": ("dignity",),
            "dignita": ("dignity",),
            "libertà": ("freedom",),
            "liberta": ("freedom",),
            "umano": ("human", "person"),
            "umana": ("human", "person"),
            "persona": ("human", "person"),
            "intelligenza": ("intelligence",),
            "artificiale": ("artificial",),
        }
        stopwords = {
            "about", "does", "mention", "mentions", "talk", "talks", "this", "that", "there", "with", "from", "page",
            "article", "text", "what", "where", "when", "which", "whether", "verify", "check", "concept", "topic",
            "parla", "parlano", "menziona", "menzionano", "questo", "questa", "pagina", "articolo", "testo",
            "concetto", "tema", "verifica", "controlla", "dice", "riguardo", "qualcosa", "particolare",
        }
        words = [word.lower() for word in re.findall(r"[\wÀ-ÿ-]{4,}", query)]
        if re.search(r"\bai\b", query, flags=re.IGNORECASE):
            words.extend(["artificial", "intelligence"])
        unique: list[str] = []
        for word in words:
            if word in stopwords or word in unique:
                continue
            unique.append(word)
            for expanded in expansions.get(word, ()):
                if expanded not in unique:
                    unique.append(expanded)
        return unique[:12]

    @classmethod
    def _clean_fragment(cls, text: str) -> str:
        return cls._extract_text(text)
