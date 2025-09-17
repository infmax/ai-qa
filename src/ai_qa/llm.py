"""Helpers for constructing the GigaChat language model used by the agent."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_gigachat.chat_models import GigaChat

_DEFAULT_SCOPE = "GIGACHAT_API_PERS"
_DEFAULT_MODEL = "GigaChat"


def create_chat_llm(
    *,
    credentials: Optional[str] = None,
    scope: str = _DEFAULT_SCOPE,
    model: str = _DEFAULT_MODEL,
    verify_ssl_certs: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a LangChain ``GigaChat`` chat model.

    Parameters
    ----------
    credentials:
        API token for accessing the GigaChat service. If omitted, the
        ``GIGACHAT_CREDENTIALS`` environment variable is used.
    scope:
        OAuth scope passed to the SDK. Defaults to ``GIGACHAT_API_PERS``.
    model:
        Name of the model variant. Defaults to ``GigaChat``.
    verify_ssl_certs:
        Whether to verify TLS certificates. Disabled by default as in the example configuration.
    **kwargs:
        Дополнительные параметры, передаваемые конструктору ``GigaChat``.
    """

    if not credentials:
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError(
            "GigaChat credentials must be provided via the credentials argument or the GIGACHAT_CREDENTIALS environment variable.",
        )

    return GigaChat(
        credentials=credentials,
        scope=scope,
        model=model,
        verify_ssl_certs=verify_ssl_certs,
        **kwargs,
    )


__all__ = ["BaseChatModel", "GigaChat", "create_chat_llm"]
