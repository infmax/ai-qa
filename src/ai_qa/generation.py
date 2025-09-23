"""Instruction generation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate

from .tools import ToolRegistry

JSONDict = Dict[str, Any]


class InstructionGenerator:
    """Wraps an ``LLMChain`` to produce JSON Playwright instructions."""

    def __init__(self, llm: BaseChatModel) -> None:
        template = PromptTemplate(
            input_variables=["step", "html", "history", "feedback", "phase"],
            template=(
                "You generate JSON Playwright instructions.\n"
                "Follow this schema exactly: {\n"
                "  \"actions\": [\n"
                "    {\n"
                "      \"type\": one of [\"goto\", \"click\", \"fill\", \"expect\", \"assert_text\"],\n"
                "      \"selector\": CSS selector when applicable,\n"
                "      \"value\": text for \"fill\",\n"
                "      \"text\": assertion text for \"assert_text\",\n"
                "      \"url\": absolute or relative URL for \"goto\"\n"
                "    }\n"
                "  ],\n"
                "  \"comment\": short reasoning string\n"
                "}.\n"
                "Use information from the current HTML to pick selectors.\n"
                "Keep actions minimal yet sufficient.\n"
                "Phase for current step: {phase}.\n"
                "If the phase is VERIFY, only include assertion-style actions (types \"expect\" or \"assert_text\") and do not perform interactions such as clicks, fills, or navigation.\n"
                "If the phase is ACT, include at least one interaction action (types \"goto\", \"click\", or \"fill\") and avoid emitting only assertion actions.\n"
                "History of prior steps and actions:\n{history}\n\n"
                "Additional feedback, prior errors, or selector hints:\n{feedback}\n\n"
                "Current page HTML snippet:\n{html}\n\n"
                "Current manual step: {step}\n\n"
                "Return only valid JSON."
            ),
        )
        self._chain = LLMChain(llm=llm, prompt=template)
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    async def agenerate(
        self,
        *,
        step: str,
        html: str,
        history: List[JSONDict],
        feedback: Optional[str] = None,
        phase: str = "ACT",
    ) -> JSONDict:
        """Generate a JSON instruction for the given step."""

        history_str = json.dumps(history, ensure_ascii=False, indent=2)
        feedback_value = feedback or "None"
        raw = await self._chain.apredict(
            step=step,
            html=html,
            history=history_str,
            feedback=feedback_value,
            phase=phase,
        )
        return self._clean_json(raw)

    @staticmethod
    def _clean_json(raw: str) -> JSONDict:
        snippet = raw.strip()
        start = snippet.find("{")
        end = snippet.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"Model response is not JSON: {raw}")
        snippet = snippet[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:  # pragma: no cover - validation
            raise ValueError(f"Invalid JSON from model: {raw}") from exc


@dataclass
class PlannedInstruction:
    """Instruction plus metadata about how it was produced."""

    instruction: JSONDict
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelection:
    """LLM decision about whether to use a specialised tool."""

    tool_name: Optional[str]
    reason: str
    raw: JSONDict


class ToolDecider:
    """LLM-backed helper that decides which tool to invoke for a step."""

    def __init__(self, llm: BaseChatModel) -> None:
        template = PromptTemplate(
            input_variables=["step", "history", "feedback", "phase", "tools", "html"],
            template=(
                "You help choose specialised tools for generating Playwright instructions.\n"
                "Each request contains a manual step, optional feedback, the current phase, and HTML context.\n"
                "Available tools are provided as a JSON array with fields `name` and `description`:\n{tools}\n\n"
                "Respond with a JSON object using this schema:{\n"
                "  \"tool\": string name of the selected tool or null if none applies,\n"
                "  \"reason\": brief rationale for your decision\n"
                "}.\n"
                "Choose a tool only if it clearly matches the step and phase. Otherwise return null.\n"
                "Never invent tool names beyond the provided list.\n"
                "Current phase: {phase}.\n"
                "Manual step: {step}.\n"
                "User/system feedback: {feedback}.\n"
                "Prior executed steps and actions: {history}.\n"
                "Relevant HTML snippet: {html}.\n"
                "Return only valid JSON."
            ),
        )
        self._chain = LLMChain(llm=llm, prompt=template)

    async def aselect(
        self,
        *,
        step: str,
        html: str,
        history: List[JSONDict],
        feedback: Optional[str],
        phase: str,
        tools: List[Dict[str, str]],
    ) -> ToolSelection:
        history_str = json.dumps(history, ensure_ascii=False, indent=2)
        feedback_value = feedback or "None"
        tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
        raw = await self._chain.apredict(
            step=step,
            history=history_str,
            feedback=feedback_value,
            phase=phase,
            tools=tools_str,
            html=html,
        )
        data = InstructionGenerator._clean_json(raw)
        tool_value = data.get("tool")
        if isinstance(tool_value, str):
            selected_tool = tool_value.strip() or None
            if selected_tool and selected_tool.lower() in {"none", "null"}:
                selected_tool = None
        else:
            selected_tool = None
        reason = data.get("reason")
        if not isinstance(reason, str):
            reason = ""
        return ToolSelection(tool_name=selected_tool, reason=reason, raw=data)


class ReactiveInstructionPlanner:
    """Chooses between specialised tools and the LLM for each step."""

    def __init__(self, generator: InstructionGenerator, tools: Optional[ToolRegistry] = None) -> None:
        self._generator = generator
        self._tools = tools or ToolRegistry()
        self._decider = ToolDecider(generator.llm) if self._tools.specs() else None

    async def agenerate(
        self,
        *,
        step: str,
        html: str,
        history: List[JSONDict],
        feedback: Optional[str] = None,
        phase: str = "ACT",
        use_tools: bool = True,
    ) -> PlannedInstruction:
        """Generate instructions, preferring specialised tools when available."""

        metadata: Dict[str, Any] = {}

        if use_tools and self._decider:
            selection = await self._decider.aselect(
                step=step,
                html=html,
                history=history,
                feedback=feedback,
                phase=phase,
                tools=self._tools.specs(),
            )
            if selection.tool_name:
                tool = self._tools.get(selection.tool_name)
                tool_result = None
                if tool:
                    tool_result = await self._tools.call(
                        selection.tool_name,
                        step=step,
                        html=html,
                        history=history,
                        feedback=feedback,
                        phase=phase,
                    )
                if tool_result:
                    instruction = tool_result.instruction
                    instruction.setdefault(
                        "comment",
                        f"Generated with tool {tool_result.tool_name}",
                    )
                    metadata = dict(tool_result.metadata)
                    metadata.update(
                        {
                            "selection_reason": selection.reason,
                            "selected_tool": selection.tool_name,
                            "selection": selection.raw,
                        }
                    )
                    return PlannedInstruction(
                        instruction=instruction,
                        source=f"tool:{tool_result.tool_name}",
                        metadata=metadata,
                    )
                metadata = {
                    "selection": selection.raw,
                    "selection_reason": selection.reason,
                    "selected_tool": selection.tool_name,
                    "selection_error": "tool_unavailable" if not tool else "tool_declined",
                }
            else:
                metadata = {
                    "selection": selection.raw,
                    "selection_reason": selection.reason,
                    "selected_tool": None,
                }

        instruction = await self._generator.agenerate(
            step=step,
            html=html,
            history=history,
            feedback=feedback,
            phase=phase,
        )
        instruction.setdefault("comment", "Generated by LLM")
        return PlannedInstruction(
            instruction=instruction,
            source="llm",
            metadata=metadata,
        )
