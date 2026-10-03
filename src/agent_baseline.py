from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Simplified baseline agent with thread-local memory only."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def _ensure_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a deterministic reply and token accounting for this thread only."""

        if not self.force_offline and self.langchain_agent is not None:
            try:
                result = self.langchain_agent.invoke({"input": message})
                text = str(getattr(result, "content", result))
                session = self._ensure_session(thread_id)
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": text})
                return {
                    "response": text,
                    "answer": text,
                    "content": text,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "token_usage": self.token_usage(thread_id),
                    "prompt_tokens_processed": self.prompt_token_usage(thread_id),
                }
            except Exception:
                self.langchain_agent = None

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._ensure_session(thread_id).token_usage if thread_id in self.sessions else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._ensure_session(thread_id).prompt_tokens_processed if thread_id in self.sessions else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Thread-scoped, deterministic offline behavior without persistent memory."""

        session = self._ensure_session(thread_id)
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        session.prompt_tokens_processed = prompt_tokens

        lowered = (message or "").strip().lower()
        if "tên" in lowered or "name" in lowered:
            text = "Tôi là BaselineAgent. Tôi chỉ nhớ trong cùng thread hiện tại; tôi không lưu long-term memory giữa các thread khác nhau."
        elif "nhớ" in lowered or "memory" in lowered:
            text = "Tôi chỉ nhớ các tin nhắn trong thread này và không có User.md hay memory dài hạn qua session mới."
        elif "mình" in lowered and ("ở" in lowered or "đang" in lowered or "làm" in lowered):
            text = "Tôi sẽ ghi nhớ thông tin trong thread hiện tại nhưng sẽ quên khi đổi thread mới."
        else:
            text = "Tôi đang trả lời theo chế độ baseline: nhớ trong thread hiện tại, không lưu lâu dài và không có User.md."

        response_tokens = estimate_tokens(text)
        session.token_usage += response_tokens

        session.messages.append({"role": "assistant", "content": text})

        result = {
            "response": text,
            "answer": text,
            "content": text,
            "thread_id": thread_id,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
            "memory_mode": "baseline_thread_only",
            "session_size": len(session.messages),
        }
        return result

    def _maybe_build_langchain_agent(self):
        """Optional live LangChain agent for environments where model dependencies are installed."""

        if self.force_offline or self.langchain_agent is not None:
            return self.langchain_agent

        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
        return self.langchain_agent
