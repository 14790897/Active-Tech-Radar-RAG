from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

LOGGER = logging.getLogger("active_radar")
INVESTIGATION_PROMPT_CHAR_LIMIT = 8000


class RadarState(TypedDict):
    question: str
    queries: List[str]
    results: List[Dict[str, Any]]
    page_cache: Dict[str, Dict[str, Any]]
    filtered: List[Dict[str, Any]]
    investigation_targets: List[Dict[str, Any]]
    researched: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    summary: str
    sufficient: bool
    followup_queries: List[str]
    iterations: int
    max_iterations: int
    max_deep_links: int


PLANNER_PROMPT = (
    "You are a query planner for technical research. "
    "Turn the user question into precise search queries for web sources like "
    "GitHub issues, arXiv, docs, and technical blogs. "
    "Return ONLY valid JSON with this schema: "
    "{\"queries\": [\"...\"]}. "
    "Make 3-6 queries. Include time constraints if the user asks for recency."
)

FILTER_PROMPT = (
    "You are a relevance grader. "
    "Given a user question and search results, decide which results are highly relevant. "
    "Return ONLY valid JSON as a list of objects: "
    "[{\"id\": 0, \"keep\": true, \"score\": 0.0, \"reason\": \"...\"}]. "
    "Score is 0-1. Keep only strong matches to the question."
)

TRIAGE_PROMPT = (
    "You are a web research triage assistant. "
    "Given a user question and already relevant search results, decide which links merit a deeper read "
    "by fetching the full page content. "
    "Prefer primary or authoritative sources, links likely to contain technical detail, and links where "
    "the snippet alone is not enough to answer the question. "
    "Avoid duplicates and low-value summary pages unless they are the best available source. "
    "Return ONLY valid JSON as a list of objects: "
    "[{\"id\": 0, \"investigate\": true, \"priority\": 0.0, \"reason\": \"...\"}]. "
    "Priority is 0-1. Select at most the provided max_links items."
)

SYNTH_PROMPT = (
    "You are a technical synthesizer. "
    "Use ONLY the provided evidence to answer the question. "
    "Return ONLY valid JSON with this schema: "
    "{\"summary\": \"...\", \"sufficient\": true, \"followup_queries\": [\"...\"]}. "
    "Evidence may include a short snippet and optional full page content extracted from the source page. "
    "The summary must be concise, actionable, and include inline citations like [1]. "
    "Citations must refer to the evidence IDs. "
    "If evidence is insufficient, say what is missing and propose 2-4 follow-up queries."
)


def _ensure_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")


def _parse_json(text: str) -> Any:
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty response", text or "", 0)
    s = text.strip()
    decoder = json.JSONDecoder()
    try:
        return decoder.decode(s)
    except json.JSONDecodeError:
        pass

    if "```" in s:
        parts = s.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if "{" in candidate or "[" in candidate:
                s = candidate
                break

    try:
        obj, _ = decoder.raw_decode(s)
        return obj
    except json.JSONDecodeError:
        pass

    first_obj = s.find("{")
    first_arr = s.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    for start in sorted(starts):
        try:
            obj, _ = decoder.raw_decode(s[start:])
            return obj
        except json.JSONDecodeError:
            continue

    for idx, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(s[idx:])
                return obj
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("No valid JSON found", s, 0)


def _unique_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in results:
        key = (item.get("url") or "", item.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _shorten(text: str, limit: int = 200) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _clip_for_prompt(text: str, limit: int = 4000) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _store_extracted_text(text: str, limit: int = 20000) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _preview_text(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _text_length(text: Any) -> int:
    return len(" ".join(str(text or "").split()))


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def _extract_page_cache_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    url = item.get("url", "")
    entry = {
        "url": url,
        "fetch_url": url,
        "cache_source": "",
        "fetched": False,
        "error": "",
        "page_title": "",
        "site_name": "",
        "published_at": "",
        "content": "",
        "raw_result": item.get("raw_result", {}),
    }
    if not url:
        entry["error"] = "missing_url"
        return entry

    raw_content = str(item.get("raw_content", "") or "").strip()
    if item.get("source") == "tavily" and raw_content:
        entry.update(
            {
                "cache_source": "tavily_raw",
                "fetched": True,
                "error": "",
                "page_title": str(item.get("title", "") or ""),
                "site_name": _domain_from_url(url),
                "published_at": str(item.get("published_at", "") or ""),
                "content": _store_extracted_text(raw_content),
            }
        )
        return entry

    snippet_content = str(item.get("snippet", "") or "").strip()
    if snippet_content:
        source_name = str(item.get("source", "") or "search")
        entry.update(
            {
                "cache_source": f"{source_name}_snippet",
                "fetched": True,
                "error": "",
                "page_title": str(item.get("title", "") or ""),
                "site_name": _domain_from_url(url),
                "published_at": str(item.get("published_at", "") or ""),
                "content": _store_extracted_text(snippet_content),
            }
        )
        return entry

    entry["error"] = "no_search_content"
    return entry


def _format_evidence_lines(evidence: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if not evidence:
        return ["[no evidence]"]
    for item in evidence:
        eid = item.get("id", "?")
        title = _shorten(item.get("title", ""), 140)
        url = item.get("url", "")
        lines.append(f"[{eid}] {title} | {url}")
    return lines


def _save_run_outputs(
    output_dir: str,
    question: str,
    summary: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    answer_path = target_dir / f"{run_id}_answer.md"
    evidence_path = target_dir / f"{run_id}_evidence.json"

    answer_text = (
        f"# Question\n\n{question}\n\n"
        f"# Answer\n\n{summary or '[no summary generated]'}\n"
    )
    answer_path.write_text(answer_text, encoding="utf-8")

    evidence_payload = {
        "run_id": run_id,
        "question": question,
        "summary": summary,
        "evidence": evidence,
    }
    evidence_path.write_text(
        json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "answer_path": str(answer_path),
        "evidence_path": str(evidence_path),
    }


class SearchClient:
    def __init__(self, backend: str, max_results: int) -> None:
        self.backend = backend
        self.max_results = max_results
        self._tavily: TavilyClient | None = None
        if backend == "tavily":
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise RuntimeError("TAVILY_API_KEY is required for tavily backend")
            self._tavily = TavilyClient(api_key=api_key)

    def search(self, query: str) -> List[Dict[str, Any]]:
        LOGGER.info("search | backend=%s | query=%s", self.backend, query)
        if self._tavily:
            response = self._tavily.search(
                query,
                max_results=self.max_results,
                include_raw_content="text",
            )
            results = response.get("results", [])
            normalized = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "raw_content": item.get("raw_content", "") or "",
                    "published_at": item.get("published_date", "") or item.get("published_at", "") or "",
                    "raw_result": item,
                    "source": "tavily",
                }
                for item in results
            ]
            for idx, item in enumerate(normalized, start=1):
                LOGGER.info(
                    "search | result | %d | title=%s | url=%s | snippet_chars=%d | raw_chars=%d | snippet=%s",
                    idx,
                    _shorten(item.get("title", ""), 120),
                    item.get("url", ""),
                    _text_length(item.get("snippet", "")),
                    _text_length(item.get("raw_content", "")),
                    _shorten(item.get("snippet", ""), 200),
                )
            return normalized
        with DDGS() as ddgs:
            items = list(ddgs.text(query, max_results=self.max_results))
        normalized = []
        for item in items:
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href") or item.get("url") or "",
                    "snippet": item.get("body") or item.get("snippet") or "",
                    "source": "ddgs",
                }
            )
        for idx, item in enumerate(normalized, start=1):
            LOGGER.info(
                "search | result | %d | title=%s | url=%s | snippet_chars=%d | raw_chars=%d | snippet=%s",
                idx,
                _shorten(item.get("title", ""), 120),
                item.get("url", ""),
                _text_length(item.get("snippet", "")),
                _text_length(item.get("raw_content", "")),
                _shorten(item.get("snippet", ""), 200),
            )
        return normalized


def build_graph(llm: ChatOpenAI, search_client: SearchClient) -> Any:
    def plan_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("plan | start")
        question = state["question"].strip()
        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=question),
        ]
        response = llm.invoke(messages)
        try:
            data = _parse_json(response.content)
        except json.JSONDecodeError:
            LOGGER.exception("plan | parse failed")
            data = {}
        queries = data.get("queries", []) if isinstance(data, dict) else []
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            queries = [question]
        LOGGER.info("plan | queries=%s", queries)
        return {
            "queries": queries,
            "results": [],
            "page_cache": {},
            "filtered": [],
            "investigation_targets": [],
            "researched": [],
            "followup_queries": [],
        }

    def search_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("search | start")
        queries = state.get("queries", [])
        gathered: List[Dict[str, Any]] = []
        for query in queries:
            gathered.extend(search_client.search(query))
        combined = _unique_results(state.get("results", []) + gathered)
        LOGGER.info("search | total_results=%d", len(combined))
        return {"results": combined}

    def cache_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("cache | start")
        results = state.get("results", [])
        existing_cache = dict(state.get("page_cache", {}))
        updated_cache = dict(existing_cache)
        fetched_count = 0
        for item in results:
            url = item.get("url", "")
            if not url:
                continue
            if url in updated_cache:
                LOGGER.info("cache | hit | url=%s", url)
                continue
            LOGGER.info("cache | fetch | url=%s", url)
            cache_entry = _extract_page_cache_entry(item)
            updated_cache[url] = cache_entry
            fetched_count += 1
            LOGGER.info(
                "cache | stored | ok=%s | source=%s | date=%s | snippet_chars=%d | raw_chars=%d | cached_chars=%d | title=%s | url=%s | fetch_url=%s | error=%s",
                cache_entry.get("fetched", False),
                cache_entry.get("cache_source", ""),
                cache_entry.get("published_at", ""),
                _text_length(item.get("snippet", "")),
                _text_length(item.get("raw_content", "")),
                _text_length(cache_entry.get("content", "")),
                _shorten(cache_entry.get("page_title", "") or item.get("title", ""), 120),
                url,
                cache_entry.get("fetch_url", url),
                cache_entry.get("error", ""),
            )
        LOGGER.info("cache | total=%d | new=%d", len(updated_cache), fetched_count)
        return {"page_cache": updated_cache}

    def filter_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("filter | start")
        results = state.get("results", [])
        page_cache = state.get("page_cache", {})
        if not results:
            LOGGER.info("filter | no results")
            return {"filtered": []}
        eligible_results = [
            item
            for item in results
            if page_cache.get(item.get("url", ""), {}).get("content", "")
        ]
        LOGGER.info(
            "filter | eligible_with_content=%d | dropped_without_content=%d",
            len(eligible_results),
            len(results) - len(eligible_results),
        )
        if not eligible_results:
            LOGGER.info("filter | no results with cached content")
            return {"filtered": []}
        evidence = [
            {
                "id": idx,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
            }
            for idx, item in enumerate(eligible_results)
        ]
        messages = [
            SystemMessage(content=FILTER_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {"question": state["question"], "results": evidence},
                    ensure_ascii=False,
                )
            ),
        ]
        response = llm.invoke(messages)
        try:
            data = _parse_json(response.content)
        except json.JSONDecodeError:
            LOGGER.exception("filter | parse failed")
            data = []
        keep_ids = set()
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                if item.get("keep") is True or (item.get("score") or 0) >= 0.6:
                    keep_ids.add(item.get("id"))
        filtered = [
            eligible_results[i]
            for i in keep_ids
            if isinstance(i, int) and i < len(eligible_results)
        ]
        LOGGER.info("filter | kept=%d", len(filtered))
        return {"filtered": filtered}

    def triage_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("triage | start")
        filtered = state.get("filtered", [])
        page_cache = state.get("page_cache", {})
        max_deep_links = max(state.get("max_deep_links", 0), 0)
        if not filtered or max_deep_links == 0:
            LOGGER.info("triage | skipped | filtered=%d | max_deep_links=%d", len(filtered), max_deep_links)
            return {"investigation_targets": [], "researched": []}

        candidates = [
            {
                "id": idx,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "domain": _domain_from_url(item.get("url", "")),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "page_title": page_cache.get(item.get("url", ""), {}).get("page_title", ""),
                "published_at": page_cache.get(item.get("url", ""), {}).get("published_at", ""),
                "cached_preview": _preview_text(page_cache.get(item.get("url", ""), {}).get("content", ""), 300),
                "has_cached_content": bool(page_cache.get(item.get("url", ""), {}).get("content", "")),
            }
            for idx, item in enumerate(filtered)
        ]
        messages = [
            SystemMessage(content=TRIAGE_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "question": state["question"],
                        "max_links": max_deep_links,
                        "results": candidates,
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        response = llm.invoke(messages)
        try:
            data = _parse_json(response.content)
        except json.JSONDecodeError:
            LOGGER.exception("triage | parse failed")
            data = []

        picked: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                idx = item.get("id")
                if not isinstance(idx, int) or idx < 0 or idx >= len(filtered):
                    continue
                priority = item.get("priority", 0.0)
                try:
                    priority = float(priority)
                except (TypeError, ValueError):
                    priority = 0.0
                if item.get("investigate") is not True and priority < 0.6:
                    continue
                picked.append(
                    {
                        "id": idx,
                        "priority": priority,
                        "reason": str(item.get("reason", "")).strip(),
                    }
                )

        seen_ids = set()
        targets: List[Dict[str, Any]] = []
        for item in sorted(picked, key=lambda x: x["priority"], reverse=True):
            idx = item["id"]
            if idx in seen_ids:
                continue
            seen_ids.add(idx)
            base = dict(filtered[idx])
            base.update(item)
            targets.append(base)
            if len(targets) >= max_deep_links:
                break

        LOGGER.info("triage | selected=%d", len(targets))
        for item in targets:
            LOGGER.info(
                "triage | target | id=%s | priority=%.2f | url=%s | reason=%s",
                item.get("id", "?"),
                item.get("priority", 0.0),
                item.get("url", ""),
                _shorten(item.get("reason", ""), 180),
            )
        return {"investigation_targets": targets, "researched": []}

    def investigate_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("investigate | start")
        targets = state.get("investigation_targets", [])
        page_cache = state.get("page_cache", {})
        if not targets:
            LOGGER.info("investigate | no targets")
            return {"researched": []}

        researched: List[Dict[str, Any]] = []
        for item in targets:
            url = item.get("url", "")
            if not url:
                continue
            LOGGER.info(
                "investigate | load | id=%s | url=%s | reason=%s",
                item.get("id", "?"),
                url,
                _shorten(item.get("reason", ""), 180),
            )
            cache_entry = page_cache.get(url, {})
            if not cache_entry:
                LOGGER.info("investigate | cache miss | url=%s", url)
                continue
            if not cache_entry.get("fetched") or not cache_entry.get("content"):
                LOGGER.info(
                    "investigate | no cached content | url=%s | error=%s",
                    url,
                    cache_entry.get("error", ""),
                )
                continue

            raw_content = str(cache_entry.get("content", "") or "")
            content = _clip_for_prompt(raw_content, INVESTIGATION_PROMPT_CHAR_LIMIT)
            researched_item = {
                "id": item.get("id"),
                "title": item.get("title", ""),
                "url": url,
                "fetch_url": cache_entry.get("fetch_url", url),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "priority": item.get("priority", 0.0),
                "investigation_reason": item.get("reason", ""),
                "page_title": str(cache_entry.get("page_title", "") or ""),
                "site_name": str(cache_entry.get("site_name", "") or ""),
                "published_at": str(cache_entry.get("published_at", "") or ""),
                "content": content,
                "cache_source": cache_entry.get("cache_source", ""),
                "raw_result": cache_entry.get("raw_result", {}),
            }
            researched.append(researched_item)
            LOGGER.info(
                "investigate | cached | id=%s | source=%s | date=%s | cached_chars=%d | prompt_chars=%d | title=%s | fetch_url=%s | preview=%s",
                researched_item.get("id", "?"),
                researched_item.get("cache_source", ""),
                researched_item.get("published_at", ""),
                len(raw_content),
                len(researched_item.get("content", "")),
                _shorten(researched_item.get("page_title", "") or researched_item.get("title", ""), 120),
                researched_item.get("fetch_url", url),
                _preview_text(researched_item.get("content", "")),
            )
        return {"researched": researched}

    def synth_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("synthesize | start")
        filtered = state.get("filtered", [])
        researched_by_id = {
            item["id"]: item
            for item in state.get("researched", [])
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        }
        evidence = [
            {
                "id": idx + 1,
                "title": researched_by_id.get(idx, {}).get("page_title") or item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "investigated": idx in researched_by_id,
                "investigation_reason": researched_by_id.get(idx, {}).get("investigation_reason", ""),
                "site_name": researched_by_id.get(idx, {}).get("site_name", ""),
                "published_at": researched_by_id.get(idx, {}).get("published_at", ""),
                "content": researched_by_id.get(idx, {}).get("content", ""),
            }
            for idx, item in enumerate(filtered)
        ]
        LOGGER.info("synthesize | evidence_count=%d", len(evidence))
        for item in evidence:
            LOGGER.info(
                "evidence | id=%s | title=%s | investigated=%s | snippet_chars=%d | content_chars=%d | url=%s | snippet=%s",
                item.get("id", "?"),
                _shorten(item.get("title", ""), 120),
                item.get("investigated", False),
                _text_length(item.get("snippet", "")),
                _text_length(item.get("content", "")),
                item.get("url", ""),
                _shorten(item.get("snippet", ""), 200),
            )
        messages = [
            SystemMessage(content=SYNTH_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {"question": state["question"], "evidence": evidence},
                    ensure_ascii=False,
                )
            ),
        ]
        response = llm.invoke(messages)
        try:
            data = _parse_json(response.content)
        except json.JSONDecodeError:
            LOGGER.exception("synthesize | parse failed")
            data = {}
        summary = ""
        sufficient = False
        followups: List[str] = []
        if isinstance(data, dict):
            summary = data.get("summary", "") or ""
            sufficient = bool(data.get("sufficient"))
            followups = [q for q in data.get("followup_queries", []) if isinstance(q, str)]
        LOGGER.info("synthesize | sufficient=%s | followups=%d", sufficient, len(followups))
        return {
            "evidence": evidence,
            "summary": summary,
            "sufficient": sufficient,
            "followup_queries": followups,
        }

    def refine_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("refine | start")
        followups = state.get("followup_queries", [])
        if not followups:
            followups = [state["question"]]
        LOGGER.info("refine | queries=%s", followups)
        return {
            "queries": followups,
            "iterations": state.get("iterations", 0) + 1,
        }

    def should_continue(state: RadarState) -> str:
        if state.get("sufficient"):
            return END
        if state.get("iterations", 0) >= state.get("max_iterations", 1):
            return END
        return "refine"

    builder = StateGraph(RadarState)
    builder.add_node("plan", plan_node)
    builder.add_node("search", search_node)
    builder.add_node("cache", cache_node)
    builder.add_node("filter", filter_node)
    builder.add_node("triage", triage_node)
    builder.add_node("investigate", investigate_node)
    builder.add_node("synthesize", synth_node)
    builder.add_node("refine", refine_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "cache")
    builder.add_edge("cache", "filter")
    builder.add_edge("filter", "triage")
    builder.add_edge("triage", "investigate")
    builder.add_edge("investigate", "synthesize")
    builder.add_conditional_edges("synthesize", should_continue, ["refine", END])
    builder.add_edge("refine", "search")

    return builder.compile()


def setup_logging(log_file: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def main(argv: List[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Active Tech-Radar RAG")
    parser.add_argument("question", nargs="?", help="Your technical question")
    parser.add_argument("--backend", choices=["ddgs", "tavily"], default="ddgs")
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-deep-links", type=int, default=2)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--log-file", default="active_radar.log")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--diagram", choices=["none", "mermaid", "png"], default="none")
    parser.add_argument("--diagram-file", default="")
    args = parser.parse_args(argv)

    setup_logging(args.log_file)
    LOGGER.info(
        "run | backend=%s | max_results=%d | max_iterations=%d | max_deep_links=%d | model=%s",
        args.backend,
        args.max_results,
        args.max_iterations,
        max(args.max_deep_links, 0),
        args.model,
    )

    question = args.question or ""
    if not question:
        LOGGER.error("missing question")
        print("Please provide a question.", file=sys.stderr)
        return 2

    _ensure_openai_key()

    llm = ChatOpenAI(model=args.model, temperature=0.2)
    search_client = SearchClient(args.backend, args.max_results)
    graph = build_graph(llm, search_client)

    if args.diagram != "none":
        if args.diagram == "mermaid":
            diagram_path = args.diagram_file or "graph.mmd"
            mermaid = graph.get_graph().draw_mermaid()
            Path(diagram_path).write_text(mermaid, encoding="utf-8")
            LOGGER.info("diagram | type=mermaid | file=%s", diagram_path)
        else:
            diagram_path = args.diagram_file or "graph.png"
            png_bytes = graph.get_graph().draw_mermaid_png()
            Path(diagram_path).write_bytes(png_bytes)
            LOGGER.info("diagram | type=png | file=%s", diagram_path)

    state: RadarState = {
        "question": question,
        "queries": [],
        "results": [],
        "page_cache": {},
        "filtered": [],
        "investigation_targets": [],
        "researched": [],
        "evidence": [],
        "summary": "",
        "sufficient": False,
        "followup_queries": [],
        "iterations": 0,
        "max_iterations": args.max_iterations,
        "max_deep_links": max(args.max_deep_links, 0),
    }

    result = graph.invoke(state)
    summary = result.get("summary", "") if isinstance(result, dict) else ""
    evidence = result.get("evidence", []) if isinstance(result, dict) else []
    saved_outputs = _save_run_outputs(args.output_dir, question, summary, evidence)
    LOGGER.info(
        "output | run_id=%s | answer_file=%s | evidence_file=%s",
        saved_outputs["run_id"],
        saved_outputs["answer_path"],
        saved_outputs["evidence_path"],
    )
    if not summary:
        LOGGER.error("no summary generated")
        print("No summary generated.")
        return 1

    print("=== Evidence ===")
    for line in _format_evidence_lines(evidence):
        print(line)
    print("")
    print(summary)
    print("")
    print(f"Answer file: {saved_outputs['answer_path']}")
    print(f"Evidence file: {saved_outputs['evidence_path']}")
    LOGGER.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
