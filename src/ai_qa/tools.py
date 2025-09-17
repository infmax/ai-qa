"""Helpers for rule-based instruction generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

JSONDict = Dict[str, Any]


def _slugify(value: str) -> str:
    """Convert a field name to a lowercase slug without spaces."""

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


@dataclass
class ToolResult:
    """Container describing an instruction produced by a tool."""

    instruction: JSONDict
    tool_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StepTool(Protocol):
    """Protocol describing a step-aware instruction helper."""

    name: str

    def matches(self, step: str, phase: str) -> bool:
        """Return ``True`` if the tool can handle the provided step."""

    async def arun(
        self,
        *,
        step: str,
        html: str,
        history: Sequence[JSONDict],
        feedback: Optional[str],
        phase: str,
    ) -> Optional[ToolResult]:
        """Produce an instruction or ``None`` if the tool cannot help."""


class FormFillTool:
    """A heuristic tool that fills textual inputs based on labels and names."""

    name = "form_fill"

    def matches(self, step: str, phase: str) -> bool:  # pragma: no cover - simple predicate
        lowered = step.lower()
        return phase == "ACT" and ("заполн" in lowered or "fill" in lowered)

    async def arun(
        self,
        *,
        step: str,
        html: str,
        history: Sequence[JSONDict],
        feedback: Optional[str],
        phase: str,
    ) -> Optional[ToolResult]:
        fields = self._extract_fields(step)
        if not fields:
            return None
        actions: List[JSONDict] = []
        metadata: Dict[str, Any] = {"fields": fields}
        for field in fields:
            selector = self._guess_selector(field, html)
            actions.append(
                {
                    "type": "fill",
                    "selector": selector,
                    "value": self._default_value(field, history),
                }
            )
        instruction = {
            "actions": actions,
            "comment": "Heuristic form fill by FormFillTool",
        }
        return ToolResult(instruction=instruction, tool_name=self.name, metadata=metadata)

    @staticmethod
    def _extract_fields(step: str) -> List[str]:
        # Look for formats like "Заполнить поля: - A; - B" or "fill: A, B"
        if ":" not in step:
            return []
        _, tail = step.split(":", 1)
        raw_fields = re.split(r"[\n;•\-]+", tail)
        cleaned = [field.strip(" .;-") for field in raw_fields]
        return [field for field in cleaned if field]

    @staticmethod
    def _guess_selector(field: str, html: str) -> str:
        # Try to find a <label for="...">Field</label>
        label_pattern = re.compile(
            r"<label[^>]*for=\"([^\"]+)\"[^>]*>\s*" + re.escape(field) + r"[\s<]*",
            re.IGNORECASE,
        )
        match = label_pattern.search(html)
        if match:
            return f"#{match.group(1)}"

        attr_pattern = re.compile(
            r"<(input|textarea|select)[^>]*(id|name|placeholder)=\"([^\"]*" + re.escape(field) + r"[^\"]*)\"",
            re.IGNORECASE,
        )
        match = attr_pattern.search(html)
        if match:
            tag, attr, value = match.groups()
            value = value.strip()
            if attr == "id":
                return f"#{value}"
            return f"{tag}[{attr}=\"{value}\"]"

        slug = _slugify(field)
        return f"input[name*={slug!r}]"

    @staticmethod
    def _default_value(field: str, history: Sequence[JSONDict]) -> str:
        slug = _slugify(field)
        return f"auto_{slug or 'value'}"


class ToolRegistry:
    """Registry dispatching steps to matching tools."""

    def __init__(self, tools: Optional[Iterable[StepTool]] = None) -> None:
        self._tools: List[StepTool] = list(tools or [])

    def register(self, tool: StepTool) -> None:
        self._tools.append(tool)

    async def arun(
        self,
        *,
        step: str,
        html: str,
        history: Sequence[JSONDict],
        feedback: Optional[str],
        phase: str,
    ) -> Optional[ToolResult]:
        for tool in self._tools:
            if not tool.matches(step, phase):
                continue
            result = await tool.arun(
                step=step,
                html=html,
                history=history,
                feedback=feedback,
                phase=phase,
            )
            if result:
                return result
        return None
