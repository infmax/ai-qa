"""Factory helpers for constructing the interactive QA agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

from .executor import PlaywrightExecutor
from .generation import InstructionGenerator, ReactiveInstructionPlanner
from .llm import create_chat_llm
from .orchestration import InstructionAgent
from .test_case import parse_test_case
from .tools import FormFillTool, ToolRegistry


def build_default_agent(
    *,
    base_url: str,
    credentials: str,
    headless: bool = True,
    slow_mo: Optional[float] = None,
    tools_factory: Optional[Callable[[], ToolRegistry]] = None,
) -> InstructionAgent:
    """Create an ``InstructionAgent`` with sensible defaults."""

    llm = create_chat_llm(credentials=credentials)
    generator = InstructionGenerator(llm)
    tools = tools_factory() if tools_factory else _default_tools()
    planner = ReactiveInstructionPlanner(generator=generator, tools=tools)
    return InstructionAgent(
        planner=planner,
        executor_factory=lambda: PlaywrightExecutor(
            base_url=base_url,
            headless=headless,
            slow_mo=slow_mo,
        ),
    )


def _default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(FormFillTool())
    return registry


def _load_steps_from_args(args: argparse.Namespace) -> List[str]:
    if args.test_case:
        source = args.test_case
    else:
        source = Path(args.test_case_file).read_text(encoding="utf-8")
    return parse_test_case(source)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for executing the agent over a test case."""

    parser = argparse.ArgumentParser(
        description="Generate and execute Playwright instructions with an LLM agent",
    )
    parser.add_argument("--base-url", required=True, help="Base URL of the application under test")
    parser.add_argument("--test-case", help="Raw test case description as a single string")
    parser.add_argument("--test-case-file", help="Path to a file containing the test case description")
    parser.add_argument(
        "--credentials",
        required=True,
        help="GigaChat API credentials token",
    )
    parser.add_argument("--headless", action="store_true", help="Run Playwright in headless mode (default)")
    parser.add_argument("--no-headless", action="store_true", help="Run Playwright with a visible browser window")
    parser.add_argument("--slow-mo", type=float, help="Slow down Playwright actions by the given milliseconds")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")

    parsed = parser.parse_args(argv)
    if parsed.test_case and parsed.test_case_file:
        parser.error("Specify either --test-case or --test-case-file, not both")
    if not (parsed.test_case or parsed.test_case_file):
        parser.error("Either --test-case or --test-case-file must be provided")

    headless = not parsed.no_headless if parsed.no_headless else True
    if parsed.headless:
        headless = True

    logging.basicConfig(level=getattr(logging, parsed.log_level.upper(), logging.INFO))

    steps = _load_steps_from_args(parsed)
    agent = build_default_agent(
        base_url=parsed.base_url,
        credentials=parsed.credentials,
        headless=headless,
        slow_mo=parsed.slow_mo,
    )

    results = asyncio.run(agent.arun(steps))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()


__all__ = ["InstructionAgent", "PlaywrightExecutor", "build_default_agent", "parse_test_case"]
