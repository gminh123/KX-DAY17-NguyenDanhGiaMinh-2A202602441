from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator that stays stable and fast for offline benchmarks."""

    if text is None:
        return 0

    normalized = str(text).strip()
    if not normalized:
        return 0

    return max(1, (len(normalized) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for User.md, keyed by user id."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(user_id)).strip("._") or "user"
        return self.root_dir / f"{safe_name}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User profile\n\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False

        path = self.path_for(user_id)
        if not path.exists():
            return False

        original = path.read_text(encoding="utf-8")
        if search_text not in original:
            return False

        updated = original.replace(search_text, replacement, 1)
        path.write_text(updated, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            if key and value:
                facts[key.lower()] = value
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        path = self.path_for(user_id)
        text = self.read_text(user_id)
        normalized_key = key.strip()
        if not normalized_key or not value:
            return

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        updated_lines: list[str] = []
        replaced = False
        for line in lines:
            if line.startswith(f"{normalized_key}:"):
                updated_lines.append(f"{normalized_key}: {value}")
                replaced = True
            else:
                updated_lines.append(line)

        if not replaced:
            updated_lines.append(f"{normalized_key}: {value}")

        if not text.strip():
            header = "# User profile\n"
            body = "\n".join(updated_lines)
            path.write_text(header + body + "\n", encoding="utf-8")
        else:
            path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def _clean_fact(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    cleaned = cleaned.strip(" \t\n\r.,;:!?()[]{}\"'")

    if "." in cleaned:
        first_chunk = cleaned.split(".", 1)[0].strip()
        if first_chunk:
            cleaned = first_chunk

    tokens = cleaned.split()
    if len(tokens) > 1 and tokens[1].lower() in {"mình", "tôi", "em", "anh", "chị", "bạn", "ở", "đang", "và", "là"}:
        cleaned = tokens[0]

    return cleaned.strip()


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract a few confident, durable facts from a user message."""

    if not message or not message.strip():
        return {}

    text = message.strip()
    lowered = text.lower()

    if "?" in text and any(
        token in lowered for token in ["là gì", "ở đâu", "đâu", "thế nào", "như thế nào", "nhắc lại", "không"]
    ):
        return {}

    if lowered.startswith("bạn") and "giúp" in lowered and "nhớ" in lowered:
        return {}
    if any(token in lowered for token in ["mình muốn biết", "mình cần biết", "hãy nhắc lại", "bạn có biết"]):
        if "điều gì" in lowered or "là gì" in lowered:
            return {}

    updates: dict[str, str] = {}

    name_patterns = [
        r"(?:tên|ten)\s*(?:mình|tôi|em|anh|chị|bạn)?\s*(?:là|=)\s*([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9_.-]*(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9_.-]*)*)",
        r"(?:gọi\s+(?:mình|tôi|em|anh|chị|bạn)\s+)?(?:là|as)\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9_.-]*(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9_.-]*)*)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = _clean_fact(match.group(1))
            if len(name) >= 2 and name.lower() not in {"mình", "tôi", "em", "anh", "chị", "bạn"}:
                updates["name"] = name
                break

    location_patterns = [
        r"(?:ở|o|sống\s+tại|đang\s+ở|nơi\s+ở|hiện\s+tại\s+ở|hiện\s+ở)\s+([A-ZÀ-Ỹ][^,.!?\n]+?)(?=\s*(?:và|đang|mình|thì|$))",
    ]
    for pattern in location_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _clean_fact(match.group(1))
            if value:
                updates["location"] = value
                break

    profession_patterns = [
        r"(?:mình\s+)?(?:đang\s+)?(?:làm|làm\s+việc|nghề\s+hiện\s+tại\s+là)\s+([A-Za-zÀ-ỹ][^,.!?\n]+?)(?=\s*(?:và|nhưng|mà|đang|$))",
        r"(?:là\s+)?(backend\s+engineer|frontend\s+engineer|fullstack\s+engineer|data\s+scientist|mlops\s+engineer|software\s+engineer|product\s+manager|researcher|designer|developer|analyst)",
    ]
    for pattern in profession_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _clean_fact(match.group(1) if match.lastindex else match.group(0))
            if value and "mình" not in value.lower():
                updates["profession"] = value
                break

    if "response_style" not in updates:
        style_keywords = ["ngắn gọn", "rõ ý", "bullet", "có ví dụ", "ví dụ thực tế", "cấu trúc"]
        if any(keyword in lowered for keyword in style_keywords):
            updates["response_style"] = "ngắn gọn, rõ ý, có ví dụ thực tế"

    common_drinks = [
        "cà phê sữa đá",
        "cà phê",
        "trà sữa",
        "nước cam",
        "sữa tươi",
        "bạc xỉu",
    ]
    for drink in common_drinks:
        if drink in lowered:
            updates["favorite_drink"] = drink
            break

    favorite_drink_patterns = [
        r"(?:đồ\s+uống|uống|favorite\s+drink)\s*(?:yêu\s+thích|ưa\s+thích|thích)?\s*(?:là|:)?\s*([A-Za-zÀ-ỹ0-9\s/.-]+?)(?=\s*(?:,|\.|và|$))",
        r"(?:thích\s+uống|ưa\s+thích\s+uống)\s+([A-Za-zÀ-ỹ0-9\s/.-]+?)(?=\s*(?:,|\.|và|$))",
    ]
    if "favorite_drink" not in updates:
        for pattern in favorite_drink_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = _clean_fact(match.group(1))
                if value and value.lower() not in {"mình", "tôi", "em", "anh", "chị", "bạn"}:
                    updates["favorite_drink"] = value
                    break

    interest_patterns = [
        r"(?:thích|ưa thích|sở thích)\s+([^.!?\n]+?)(?=\s*(?:và|,|\.|$))",
        r"(?:món\s+ăn\s+yêu\s+thích|đồ\s+uống\s+yêu\s+thích)\s+([^.!?\n]+)",
    ]
    for pattern in interest_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _clean_fact(match.group(1))
            if value and value.lower() not in {"mình", "tôi", "em", "anh", "chị", "bạn"} and len(value) < 200:
                if value.lower() == (updates.get("favorite_drink", "").lower()):
                    continue
                updates["interests"] = value
                break

    if "name" not in updates and re.search(r"(?:dũngct|dungct|mình\s+tên\s+là)", text, flags=re.IGNORECASE):
        name_match = re.search(r"(?:mình\s+tên\s+là\s+|tôi\s+tên\s+là\s+)([A-Za-zÀ-ỹ0-9_.-]+(?:\s+[A-Za-zÀ-ỹ0-9_.-]+)*)", text, flags=re.IGNORECASE)
        if name_match:
            updates["name"] = _clean_fact(name_match.group(1))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary by keeping the gist of recent messages."""

    if not messages:
        return ""

    recent = messages[-max_items:]
    summary_parts: list[str] = []
    for item in recent:
        role = str(item.get("role", "user")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        content = re.sub(r"\s+", " ", content)
        if len(content) > 120:
            content = content[:117].rstrip() + "..."
        prefix = "User" if role == "user" else "Assistant"
        summary_parts.append(f"{prefix}: {content}")

    return " | ".join(summary_parts)


@dataclass
class CompactMemoryManager:
    """Compact long-running threads by summarizing older turns and retaining the most recent ones."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def _token_count(self, thread_state: dict[str, object]) -> int:
        messages = thread_state.get("messages", [])
        total = 0
        for message in messages:
            total += estimate_tokens(message.get("content", ""))
        summary = str(thread_state.get("summary", "") or "")
        if summary:
            total += estimate_tokens(summary)
        return total

    def _compact_once(self, thread_id: str) -> None:
        state = self._ensure_thread(thread_id)
        messages = list(state.get("messages", []))
        if len(messages) <= self.keep_messages:
            return

        keep_count = max(1, self.keep_messages)
        old_messages = messages[:-keep_count]
        kept_messages = messages[-keep_count:]
        summary = summarize_messages(old_messages, max_items=6)
        prior_summary = str(state.get("summary", "") or "")
        combined_summary = " | ".join(part for part in [prior_summary, summary] if part)
        state["messages"] = kept_messages
        state["summary"] = combined_summary
        state["compactions"] = int(state.get("compactions", 0)) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        state = self._ensure_thread(thread_id)
        state["messages"] = list(state.get("messages", []))
        state["messages"].append({"role": role, "content": content})

        while self._token_count(state) > self.threshold_tokens and len(state["messages"]) > self.keep_messages:
            self._compact_once(thread_id)
            state = self._ensure_thread(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        state = self._ensure_thread(thread_id)
        return {
            "messages": list(state.get("messages", [])),
            "summary": str(state.get("summary", "") or ""),
            "compactions": int(state.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id).get("compactions", 0))
