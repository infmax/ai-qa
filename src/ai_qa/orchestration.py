"""LangGraph workflow orchestrating generation, validation, and execution."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from langgraph.graph import END, StateGraph
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from .generation import PlannedInstruction, ReactiveInstructionPlanner
from .executor import PlaywrightExecutor

JSONDict = Dict[str, Any]

logger = logging.getLogger(__name__)


class InstructionAgent:
    """Coordinates the LangGraph workflow, generator, and Playwright executor."""

    def __init__(
        self,
        planner: ReactiveInstructionPlanner,
        executor_factory: Callable[[], PlaywrightExecutor],
    ) -> None:
        self._planner = planner
        self._executor_factory = executor_factory

    async def arun(self, steps: Iterable[str]) -> List[JSONDict]:
        history: List[JSONDict] = []
        async with self._executor_factory() as executor:
            current_html = await executor.html()
            for index, step in enumerate(steps, 1):
                logger.info("Processing step %s: %s", index, step)
                phase = "VERIFY" if index % 2 == 1 else "ACT"
                state = await self._run_single_step(
                    executor=executor,
                    step=step,
                    step_index=index,
                    html=current_html,
                    history=history,
                    phase=phase,
                )
                instruction = state["instruction"]
                current_html = state["html"]
                history.append(
                    {
                        "step_index": index,
                        "step": step,
                        "instruction": instruction,
                        "retries": state.get("retries", 0),
                        "source": state.get("instruction_source", "unknown"),
                        "phase": phase,
                    }
                )
        return history

    async def _run_single_step(
        self,
        *,
        executor: PlaywrightExecutor,
        step: str,
        step_index: int,
        html: str,
        history: List[JSONDict],
        phase: str,
    ) -> JSONDict:
        StepState = Dict[str, Any]

        graph = StateGraph(dict)

        async def generate_node(state: StepState) -> StepState:
            feedback_parts: List[str] = []
            if state.get("error"):
                feedback_parts.append(f"Previous error: {state['error']}")
            if state.get("user_feedback"):
                feedback_parts.append(f"User feedback: {state['user_feedback']}")
            feedback_message = "\n".join(feedback_parts)
            plan: PlannedInstruction = await self._planner.agenerate(
                step=step,
                html=state.get("html", html),
                history=history,
                feedback=feedback_message or None,
                phase=phase,
                use_tools=not state.get("force_llm", False),
            )
            logger.debug(
                "Generated instruction for step %s via %s: %s",
                step_index,
                plan.source,
                plan.instruction,
            )
            return {
                "instruction": plan.instruction,
                "instruction_source": plan.source,
                "instruction_metadata": plan.metadata,
                "error": None,
                "review_decision": None,
                "validation_decision": None,
                "execution_status": None,
                "retries": state.get("retries", 0),
                "user_feedback": None,
                "force_llm": False,
                "phase": state.get("phase", phase),
            }

        def replace_selector(instruction: JSONDict, original: str, new: str) -> JSONDict:
            updated = json.loads(json.dumps(instruction))  # deep copy
            actions = updated.get("actions", [])
            for action in actions:
                if action.get("selector") == original:
                    action["selector"] = new
            return updated

        def prompt_user(prompt: str) -> str:
            try:
                return input(prompt)
            except EOFError:
                return ""

        def pretty_instruction(instruction: JSONDict) -> str:
            return json.dumps(instruction, ensure_ascii=False, indent=2)

        def user_review_node(state: StepState) -> StepState:
            instruction = state["instruction"]
            retries = state.get("retries", 0)
            source = state.get("instruction_source", "unknown")
            metadata = state.get("instruction_metadata", {})
            while True:
                print("\n=== Step", step_index, "===")
                print(step)
                print(f"Expected phase: {phase}")
                print(f"Instruction source: {source}")
                if metadata:
                    print(f"Tool metadata: {json.dumps(metadata, ensure_ascii=False)}")
                print("Generated instruction:")
                print(pretty_instruction(instruction))
                response = prompt_user(
                    "Approve instructions? [Enter=accept / r=regenerate / llm=force LLM / JSON / old=>new selector]: "
                ).strip()
                if not response or response.lower() in {"y", "yes"}:
                    decision = "validated"
                    user_feedback = None
                    force_llm = False
                    break
                if response.lower() == "r":
                    decision = "regenerate"
                    user_feedback = "User requested regeneration"
                    retries += 1
                    force_llm = False
                    break
                if response.lower() == "llm":
                    decision = "regenerate"
                    user_feedback = "User forced LLM"
                    retries += 1
                    force_llm = True
                    break
                if "=>" in response:
                    old, new = (part.strip() for part in response.split("=>", 1))
                    instruction = replace_selector(instruction, old, new)
                    print("Updated instruction with new selector.\n")
                    continue
                if response.startswith("{"):
                    try:
                        instruction = json.loads(response)
                        print("Instruction overridden by user-provided JSON.\n")
                    except json.JSONDecodeError as exc:
                        print(f"Invalid JSON: {exc}. Try again.\n")
                    continue
                print("Unrecognized input. Please respond again.\n")
            return {
                "instruction": instruction,
                "review_decision": decision,
                "user_feedback": user_feedback,
                "retries": retries,
                "force_llm": force_llm,
                "phase": state.get("phase", phase),
            }

        def user_review_router(state: StepState) -> str:
            return "generate" if state.get("review_decision") == "regenerate" else "validate"

        async def validate_node(state: StepState) -> StepState:
            instruction = state["instruction"]
            actions = instruction.get("actions", [])
            if not isinstance(actions, Iterable):
                message = "Instruction must contain iterable 'actions'"
                print(f"Validation error: {message}\n")
                return {
                    "validation_decision": "regenerate",
                    "error": message,
                    "retries": state.get("retries", 0) + 1,
                    "phase": state.get("phase", phase),
                }
            errors: List[str] = []
            for position, action in enumerate(actions, 1):
                action_type = action.get("type")
                if action_type == "goto":
                    continue
                selector = action.get("selector")
                if not isinstance(selector, str) or not selector.strip():
                    errors.append(f"Action {position} missing selector")
                    continue
                failure = await executor.validate_selector(selector)
                if failure:
                    errors.append(f"Action {position} selector error: {failure}")
            if errors:
                message = "; ".join(errors)
                print(f"Selector validation failed: {message}\n")
                return {
                    "validation_decision": "regenerate",
                    "error": message,
                    "retries": state.get("retries", 0) + 1,
                    "phase": state.get("phase", phase),
                }

            phase_name = state.get("phase", phase)
            action_types = [action.get("type") for action in actions if isinstance(action, dict)]
            if phase_name == "VERIFY":
                allowed = {"expect", "assert_text"}
                if not action_types:
                    message = "VERIFY steps must contain at least one assertion action"
                    print(f"Validation error: {message}\n")
                    return {
                        "validation_decision": "regenerate",
                        "error": message,
                        "retries": state.get("retries", 0) + 1,
                        "phase": phase_name,
                    }
                invalid = [action for action in action_types if action not in allowed]
                if invalid:
                    message = "VERIFY steps cannot include interaction actions"
                    print(f"Validation error: {message}\n")
                    return {
                        "validation_decision": "regenerate",
                        "error": message,
                        "retries": state.get("retries", 0) + 1,
                        "phase": phase_name,
                    }
            else:
                interactive = {"goto", "click", "fill"}
                if not any(action in interactive for action in action_types):
                    message = "ACT steps must include at least one interaction action"
                    print(f"Validation error: {message}\n")
                    return {
                        "validation_decision": "regenerate",
                        "error": message,
                        "retries": state.get("retries", 0) + 1,
                        "phase": phase_name,
                    }
            return {
                "validation_decision": "execute",
                "error": None,
                "phase": state.get("phase", phase),
            }

        def validation_router(state: StepState) -> str:
            return "generate" if state.get("validation_decision") == "regenerate" else "execute"

        async def execute_node(state: StepState) -> StepState:
            instruction = state["instruction"]
            snapshot_html = await executor.html()
            try:
                await executor.run_instruction(instruction)
            except PlaywrightTimeoutError as exc:
                message = f"Playwright timeout: {exc}"
                print(f"Execution failed with timeout: {exc}\n")
                await executor.restore_html(snapshot_html)
                return {
                    "execution_status": "retry",
                    "error": message,
                    "retries": state.get("retries", 0) + 1,
                    "instruction": instruction,
                    "html": snapshot_html,
                    "phase": state.get("phase", phase),
                }
            except Exception as exc:  # pragma: no cover - runtime execution failure
                message = f"Execution error: {exc}"
                print(f"Execution failed: {exc}\n")
                await executor.restore_html(snapshot_html)
                return {
                    "execution_status": "retry",
                    "error": message,
                    "retries": state.get("retries", 0) + 1,
                    "instruction": instruction,
                    "html": snapshot_html,
                    "phase": state.get("phase", phase),
                }
            current_html = await executor.html()
            return {
                "execution_status": "success",
                "html": current_html,
                "error": None,
                "instruction": instruction,
                "phase": state.get("phase", phase),
            }

        def execution_router(state: StepState) -> str:
            return "generate" if state.get("execution_status") == "retry" else "end"

        graph.add_node("generate", generate_node)
        graph.add_node("user_review", user_review_node)
        graph.add_node("validate", validate_node)
        graph.add_node("execute", execute_node)

        graph.set_entry_point("generate")
        graph.add_edge("generate", "user_review")
        graph.add_conditional_edges(
            "user_review",
            user_review_router,
            {"generate": "generate", "validate": "validate"},
        )
        graph.add_conditional_edges(
            "validate",
            validation_router,
            {"generate": "generate", "execute": "execute"},
        )
        graph.add_conditional_edges("execute", execution_router, {"generate": "generate", "end": END})

        app = graph.compile()

        initial_state: StepState = {
            "html": html,
            "instruction": {},
            "retries": 0,
            "force_llm": False,
            "phase": phase,
        }

        final_state = await app.ainvoke(initial_state)
        return final_state


__all__ = ["InstructionAgent"]