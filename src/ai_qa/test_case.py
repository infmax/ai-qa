"""Utilities for working with textual test cases."""

from __future__ import annotations

import re
from typing import List


def parse_test_case(text: str) -> List[str]:
    """Split the raw test case description into numbered steps.

    The function accepts either a multiline description or a single string where
    steps are separated by integers followed by a dot (``1.``, ``2.``, ...).
    Empty items and surrounding whitespace are stripped from the result.
    """

    parts = re.split(r"\s*\d+[\.)]?\s*", text.strip())
    return [part.strip() for part in parts if part.strip()]
