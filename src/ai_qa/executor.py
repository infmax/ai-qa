"""Playwright execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urljoin

from playwright.async_api import (
    Browser,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

JSONDict = Dict[str, Any]


@dataclass
class PlaywrightExecutor:
    """Executes JSON instructions on a Playwright browser page."""

    base_url: Optional[str] = None
    headless: bool = True
    slow_mo: Optional[float] = None

    browser: Optional[Browser] = None
    page: Optional[Page] = None
    _playwright: Any = None

    async def __aenter__(self) -> "PlaywrightExecutor":
        self._playwright = await async_playwright().start()
        launch_kwargs: Dict[str, Any] = {"headless": self.headless}
        if self.slow_mo:
            launch_kwargs["slow_mo"] = self.slow_mo
        self.browser = await self._playwright.chromium.launch(**launch_kwargs)
        self.page = await self.browser.new_page()
        if self.base_url:
            await self.page.goto(self.base_url)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.browser:
            await self.browser.close()
        if getattr(self, "_playwright", None):
            await self._playwright.stop()

    async def html(self) -> str:
        if not self.page:
            raise RuntimeError("Page is not initialized")
        return await self.page.content()

    async def restore_html(self, html: str) -> None:
        if not self.page:
            raise RuntimeError("Page is not initialized")
        await self.page.set_content(html)

    async def run_instruction(self, instruction: JSONDict) -> None:
        actions = instruction.get("actions", [])
        if not isinstance(actions, Iterable):
            raise ValueError("Instruction JSON must contain an iterable 'actions' list")
        for action in actions:
            await self._execute_action(action)

    async def validate_selector(self, selector: str) -> Optional[str]:
        if not self.page:
            raise RuntimeError("Page is not initialized")
        try:
            locator = self.page.locator(selector)
            count = await locator.count()
        except PlaywrightError as exc:  # pragma: no cover - direct Playwright failure
            return str(exc)
        if count == 0:
            return f"Selector {selector!r} matched 0 elements"
        return None

    async def _execute_action(self, action: JSONDict) -> None:
        if not self.page:
            raise RuntimeError("Page is not initialized")
        action_type = action.get("type")
        if action_type == "goto":
            url = action.get("url")
            if not isinstance(url, str):
                raise ValueError("'goto' action requires a 'url' string")
            target = url if not self.base_url else urljoin(self.base_url, url)
            await self.page.goto(target)
            return
        selector = action.get("selector")
        if not isinstance(selector, str):
            raise ValueError(f"Action '{action_type}' requires a 'selector' string")
        if action_type == "click":
            await self.page.click(selector)
        elif action_type == "fill":
            value = action.get("value", "")
            await self.page.fill(selector, str(value))
        elif action_type == "expect":
            timeout = action.get("timeout", 5000)
            await self.page.wait_for_selector(selector, timeout=timeout)
        elif action_type == "assert_text":
            expected = str(action.get("text", ""))
            locator = self.page.locator(selector)
            content = await locator.text_content()
            if content is None or expected not in content:
                raise AssertionError(
                    f"Selector {selector!r} text {content!r} does not contain expected {expected!r}"
                )
        else:
            raise ValueError(f"Unsupported action type: {action_type}")


__all__ = ["PlaywrightExecutor", "PlaywrightTimeoutError"]
