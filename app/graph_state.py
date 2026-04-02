from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict


class AnalysisGraphState(TypedDict, total=False):
    location: Tuple[float, float]
    before_date: str
    after_date: str
    before_image: Any
    after_image: Any
    diff_map: Any
    classification_label: str
    confidence: float
    classification: Dict[str, Any]
    summary: str
    solutions: str
    status: str
    errors: List[str]
    timestamps: Dict[str, str]
    metadata: Dict[str, Any]
    task_id: str
    result: Dict[str, Any]
