from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from ddgs import DDGS
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

LOGGER = logging.getLogger("active_radar")


class RadarState(TypedDict):
    question: str
    queries: List[str]
    results: List[Dict[str, Any]]
    filtered: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    summary: str
    sufficient: bool
    followup_queries: List[str]
    iterations: int
    max_iterations: int


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

SYNTH_PROMPT = (
    "You are a technical synthesizer. "
    "Use ONLY the provided evidence to answer the question. "
    "Return ONLY valid JSON with this schema: "
    "{\"summary\": \"...\", \"sufficient\": true, \"followup_queries\": [\"...\"]}. "
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
            response = self._tavily.search(query, max_results=self.max_results)
            results = response.get("results", [])
            normalized = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "tavily",
                }
                for item in results
            ]
            for idx, item in enumerate(normalized, start=1):
                LOGGER.info(
                    "search | result | %d | title=%s | url=%s | snippet=%s",
                    idx,
                    _shorten(item.get("title", ""), 120),
                    item.get("url", ""),
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
                "search | result | %d | title=%s | url=%s | snippet=%s",
                idx,
                _shorten(item.get("title", ""), 120),
                item.get("url", ""),
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
            "filtered": [],
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

    def filter_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("filter | start")
        results = state.get("results", [])
        if not results:
            LOGGER.info("filter | no results")
            return {"filtered": []}
        evidence = [
            {
                "id": idx,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
            }
            for idx, item in enumerate(results)
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
        filtered = [results[i] for i in keep_ids if isinstance(i, int) and i < len(results)]
        LOGGER.info("filter | kept=%d", len(filtered))
        return {"filtered": filtered}

    def synth_node(state: RadarState) -> Dict[str, Any]:
        LOGGER.info("synthesize | start")
        filtered = state.get("filtered", [])
        evidence = [
            {
                "id": idx + 1,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
            }
            for idx, item in enumerate(filtered)
        ]
        LOGGER.info("synthesize | evidence_count=%d", len(evidence))
        for item in evidence:
            LOGGER.info(
                "evidence | id=%s | title=%s | url=%s | snippet=%s",
                item.get("id", "?"),
                _shorten(item.get("title", ""), 120),
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
    builder.add_node("filter", filter_node)
    builder.add_node("synthesize", synth_node)
    builder.add_node("refine", refine_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "filter")
    builder.add_edge("filter", "synthesize")
    builder.add_conditional_edges("synthesize", should_continue, ["refine", END])
    builder.add_edge("refine", "search")

    return builder.compile()


def setup_logging(log_file: str) -> None:
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
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--log-file", default="active_radar.log")
    parser.add_argument("--diagram", choices=["none", "mermaid", "png"], default="none")
    parser.add_argument("--diagram-file", default="")
    args = parser.parse_args(argv)

    setup_logging(args.log_file)
    LOGGER.info("run | backend=%s | max_results=%d | max_iterations=%d | model=%s",
                args.backend, args.max_results, args.max_iterations, args.model)

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
        "filtered": [],
        "evidence": [],
        "summary": "",
        "sufficient": False,
        "followup_queries": [],
        "iterations": 0,
        "max_iterations": args.max_iterations,
    }

    result = graph.invoke(state)
    summary = result.get("summary", "") if isinstance(result, dict) else ""
    evidence = result.get("evidence", []) if isinstance(result, dict) else []
    if not summary:
        LOGGER.error("no summary generated")
        print("No summary generated.")
        return 1

    print("=== Evidence ===")
    for line in _format_evidence_lines(evidence):
        print(line)
    print("")
    print(summary)
    LOGGER.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
