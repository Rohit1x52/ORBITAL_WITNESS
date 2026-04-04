from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agent import SatelliteAgentConfig, RAGChainBuilder
from .classifier import classify_image
from .graph_state import AnalysisGraphState
from .image_utils import detect_changes, preprocess_image
from .nasa_api import fetch_imagery

logger = logging.getLogger(__name__)


class GraphOrchestrator:
    def __init__(self, config: Optional[SatelliteAgentConfig] = None):
        self.config = config or SatelliteAgentConfig()
        self.rag_builder = RAGChainBuilder(self.config)
        self.rag_chain = self.rag_builder.build_rag_chain()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self._build_checkpointer())

    def _build_checkpointer(self):
        backend = getattr(self.config, "checkpoint_backend", "memory")
        if backend != "memory":
            logger.warning("Checkpoint backend '%s' is not configured yet; using in-memory checkpointing.", backend)
        return MemorySaver()

    def _timestamp(self) -> str:
        return datetime.now().isoformat()

    def _ensure_state(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = dict(state)
        state.setdefault("errors", [])
        state.setdefault("timestamps", {})
        state.setdefault("metadata", {})
        state["metadata"].setdefault("node_latency_ms", {})
        return state

    def _record_node_latency(self, state: AnalysisGraphState, node_name: str, start_time: float) -> None:
        elapsed_ms = (perf_counter() - start_time) * 1000
        state["metadata"]["node_latency_ms"][node_name] = round(elapsed_ms, 2)

    def fetch_imagery_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            location = state["location"]
            before_date = state["before_date"]
            after_date = state["after_date"]
            state["before_image"] = fetch_imagery(location, before_date)
            state["after_image"] = fetch_imagery(location, after_date)
            state["timestamps"]["fetch_imagery"] = self._timestamp()
            state["status"] = "imagery_fetched"
        except Exception as exc:
            state["errors"].append(f"fetch_imagery: {exc}")
            state["status"] = "failed"
        self._record_node_latency(state, "fetch_imagery", start)
        return state

    def preprocess_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            state["before_image"] = preprocess_image(state["before_image"])
            state["after_image"] = preprocess_image(state["after_image"])
            state["timestamps"]["preprocess"] = self._timestamp()
            state["status"] = "preprocessed"
        except Exception as exc:
            state["errors"].append(f"preprocess: {exc}")
            state["status"] = "failed"
        self._record_node_latency(state, "preprocess", start)
        return state

    def detect_changes_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            state["diff_map"] = detect_changes(state["before_image"], state["after_image"])
            state["timestamps"]["detect_changes"] = self._timestamp()
            state["status"] = "changes_detected"
        except Exception as exc:
            state["errors"].append(f"detect_changes: {exc}")
            state["status"] = "failed"
        self._record_node_latency(state, "detect_changes", start)
        return state

    def classify_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            result = classify_image(state["diff_map"])
            state["classification"] = result
            state["classification_label"] = result.get("label", "unknown")
            state["confidence"] = float(result.get("confidence", 0.0))
            state["timestamps"]["classify"] = self._timestamp()
            state["status"] = "classified"
        except Exception as exc:
            state["errors"].append(f"classify: {exc}")
            state["classification"] = {"label": "Error", "confidence": 0.0}
            state["classification_label"] = "Error"
            state["confidence"] = 0.0
            state["status"] = "failed"
        self._record_node_latency(state, "classify", start)
        return state

    def fallback_classify_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            result = classify_image(state["diff_map"], backbone="densenet121")
            state["classification"] = result
            state["classification_label"] = result.get("label", "unknown")
            state["confidence"] = float(result.get("confidence", 0.0))
            state["metadata"]["fallback_classifier"] = "densenet121"
            state["timestamps"]["fallback_classify"] = self._timestamp()
            state["status"] = "fallback_classified"
        except Exception as exc:
            state["errors"].append(f"fallback_classify: {exc}")
            state["status"] = "failed"
        self._record_node_latency(state, "fallback_classify", start)
        return state

    def retrieve_context_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            label = state.get("classification", {}).get("label", "")
            retriever = self.rag_builder.get_vectorstore().as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(label)
            state["metadata"]["retrieved_context"] = "\n\n".join(
                doc.page_content[:300] for doc in docs
            )
            state["timestamps"]["retrieve_context"] = self._timestamp()
            state["status"] = "context_retrieved"
        except Exception as exc:
            state["errors"].append(f"retrieve_context: {exc}")
            state["status"] = "failed"
        self._record_node_latency(state, "retrieve_context", start)
        return state

    def summarize_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            classification = state.get("classification", {})
            label = classification.get("label", "unknown")
            confidence = classification.get("confidence", 0.0)
            state["summary"] = f"Detected {label} with confidence {confidence:.2f}."
            state["timestamps"]["summarize"] = self._timestamp()
            state["status"] = "summarized"
        except Exception as exc:
            state["errors"].append(f"summarize: {exc}")
            state["status"] = "failed"
        self._record_node_latency(state, "summarize", start)
        return state

    def generate_solutions_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        try:
            label = state.get("classification", {}).get("label", "Error")
            if label in {"No Significant Change", "Error"}:
                state["solutions"] = (
                    "No significant change detected. No immediate action required."
                    if label == "No Significant Change"
                    else "Unable to generate solutions due to classification error."
                )
            else:
                payload = {
                    "classification": state.get("classification", {}),
                    "summary": state.get("summary", ""),
                    "context": state.get("metadata", {}).get("retrieved_context", ""),
                    "input_params": {
                        "location": state.get("location"),
                        "before_date": state.get("before_date"),
                        "after_date": state.get("after_date"),
                    },
                }
                state["solutions"] = self.rag_chain.invoke(payload)
            state["timestamps"]["generate_solutions"] = self._timestamp()
            state["status"] = "solutions_generated"
        except Exception as exc:
            state["errors"].append(f"generate_solutions: {exc}")
            state["solutions"] = f"Error generating solutions: {exc}"
            state["status"] = "failed"
        self._record_node_latency(state, "generate_solutions", start)
        return state

    def finalize_node(self, state: AnalysisGraphState) -> AnalysisGraphState:
        state = self._ensure_state(state)
        start = perf_counter()
        state["timestamps"]["finalize"] = self._timestamp()
        state["metadata"]["confidence_threshold"] = self.config.confidence_threshold
        state["metadata"]["workflow_mode"] = "graph"
        state["metadata"]["pipeline_latency_ms"] = round(
            sum(state["metadata"].get("node_latency_ms", {}).values()), 2
        )
        state["result"] = {
            "classification": state.get("classification", {}),
            "summary": state.get("summary", ""),
            "solutions": state.get("solutions", ""),
            "timestamp": self._timestamp(),
            "metadata": state.get("metadata", {}),
        }
        state["status"] = "completed"
        self._record_node_latency(state, "finalize", start)
        return state

    def route_after_classify(self, state: AnalysisGraphState) -> str:
        confidence = float(state.get("confidence", 0.0))
        if confidence >= self.config.confidence_threshold:
            return "summarize"
        return "fallback_classify"

    def _build_graph(self):
        graph = StateGraph(AnalysisGraphState)
        graph.add_node("fetch_imagery", self.fetch_imagery_node)
        graph.add_node("preprocess", self.preprocess_node)
        graph.add_node("detect_changes", self.detect_changes_node)
        graph.add_node("classify", self.classify_node)
        graph.add_node("fallback_classify", self.fallback_classify_node)
        graph.add_node("summarize", self.summarize_node)
        graph.add_node("retrieve_context", self.retrieve_context_node)
        graph.add_node("generate_solutions", self.generate_solutions_node)
        graph.add_node("finalize", self.finalize_node)

        graph.set_entry_point("fetch_imagery")
        graph.add_edge("fetch_imagery", "preprocess")
        graph.add_edge("preprocess", "detect_changes")
        graph.add_edge("detect_changes", "classify")
        graph.add_conditional_edges(
            "classify",
            self.route_after_classify,
            {
                "summarize": "summarize",
                "fallback_classify": "fallback_classify",
            },
        )
        graph.add_edge("fallback_classify", "summarize")
        graph.add_edge("summarize", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_solutions")
        graph.add_edge("generate_solutions", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def invoke(self, input_data: Dict[str, Any], thread_id: Optional[str] = None) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id or "orbital-witness-analysis"}}
        state = self.app.invoke(input_data, config=config)
        return state.get("result", {})
