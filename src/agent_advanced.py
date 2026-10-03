from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Advanced agent with persistent user memory + compact long-thread memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to the live mode if available, otherwise use deterministic offline behavior."""

        if not self.force_offline and self.langchain_agent is not None:
            try:
                response = self.langchain_agent.invoke({"input": message})
                text = str(getattr(response, "content", response))
                self.compact_memory.append(thread_id, "user", message)
                self.compact_memory.append(thread_id, "assistant", text)
                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(text)
                self.thread_prompt_tokens[thread_id] = self._estimate_prompt_context_tokens(user_id, thread_id)
                return {
                    "response": text,
                    "answer": text,
                    "content": text,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "token_usage": self.thread_tokens.get(thread_id, 0),
                    "prompt_tokens_processed": self.thread_prompt_tokens.get(thread_id, 0),
                }
            except Exception:
                self.langchain_agent = None

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path: extract facts, persist them, compact, then answer with recall."""

        updates = extract_profile_updates(message)
        if updates:
            for key, value in updates.items():
                self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = prompt_tokens

        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)

        response_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens

        return {
            "response": answer,
            "answer": answer,
            "content": answer,
            "user_id": user_id,
            "thread_id": thread_id,
            "token_usage": self.thread_tokens.get(thread_id, 0),
            "prompt_tokens_processed": prompt_tokens,
            "memory_file_size": self.memory_file_size(user_id),
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate context cost: profile file + compact summary + recent messages."""

        profile_text = self.profile_store.read_text(user_id)
        profile_tokens = estimate_tokens(profile_text)

        compact_state = self.compact_memory.context(thread_id)
        summary_text = str(compact_state.get("summary", "") or "")
        recent_messages = compact_state.get("messages", [])
        summary_tokens = estimate_tokens(summary_text)
        recent_tokens = sum(estimate_tokens(str(msg.get("content", ""))) for msg in recent_messages)

        return profile_tokens + summary_tokens + recent_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Use the persisted facts and compact memory to answer recall style questions."""

        profile = self.profile_store.facts(user_id)
        lower = (message or "").lower()

        fact_pairs: list[str] = []

        if profile.get("name") and any(token in lower for token in ["tên mình là gì", "tên là gì", "mình tên gì", "bạn biết tên", "tên của mình", "tên", "who"]):
            fact_pairs.append(f"tên bạn là {profile['name']}")

        if profile.get("location") and any(token in lower for token in ["đang ở đâu", "ở đâu", "nơi ở", "sống ở", "mình ở", "hiện tại mình ở", "ở hiện tại", "nơi ở hiện tại"]):
            fact_pairs.append(f"bạn đang ở {profile['location']}")

        if profile.get("profession") and any(token in lower for token in ["làm nghề gì", "nghề nghiệp", "hiện tại mình làm", "mình làm nghề", "nghề của mình", "mình làm gì", "làm gì"]):
            fact_pairs.append(f"nghề hiện tại là {profile['profession']}")

        if profile.get("response_style") and any(token in lower for token in ["style trả lời", "trả lời mình thích", "muốn trả lời", "cách trả lời", "reply style", "trả lời như thế nào", "style"]):
            fact_pairs.append(f"style trả lời bạn thích là {profile['response_style']}")

        if profile.get("favorite_drink") and any(token in lower for token in ["đồ uống", "uống gì", "thích uống", "favorite drink", "cà phê", "uống"]):
            fact_pairs.append(f"đồ uống yêu thích là {profile['favorite_drink']}")

        if profile.get("interests") and any(token in lower for token in ["mình thích gì", "sở thích", "interests", "favorite", "món ăn", "ăn gì", "python", "ai", "benchmark"]):
            fact_pairs.append(f"mối quan tâm chính là {profile['interests']}")

        if fact_pairs:
            if len(fact_pairs) == 1:
                return f"Mình biết {fact_pairs[0]}."
            return "Mình nhớ: " + "; ".join(fact_pairs) + "."

        if profile:
            summary_bits = []
            if profile.get("name"):
                summary_bits.append(f"tên là {profile['name']}")
            if profile.get("location"):
                summary_bits.append(f"ở {profile['location']}")
            if profile.get("profession"):
                summary_bits.append(f"làm {profile['profession']}")
            if profile.get("response_style"):
                summary_bits.append(f"ưu tiên trả lời {profile['response_style']}")
            if summary_bits:
                return "Tôi nhớ bạn " + ", ".join(summary_bits) + "."

        return "Tôi chưa có đủ thông tin vững chắc về bạn trong User.md để trả lời chính xác." 

    def _maybe_build_langchain_agent(self):
        """Optional live LangChain integration; no-op when the dependencies are absent."""

        if self.force_offline:
            self.langchain_agent = None
            return self.langchain_agent

        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
        return self.langchain_agent
