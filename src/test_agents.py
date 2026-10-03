from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, CompactMemoryManager


def make_config(tmp_path: Path):
    """Build an isolated config with a low compaction threshold."""

    config = load_config(tmp_path)
    config.state_dir = tmp_path / "state"
    config.compact_threshold_tokens = 60
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify profile storage supports reading, writing, and editing."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "alice"

    profile_path = store.write_text(user_id, "# User profile\n\nname: Alice\n")
    assert store.read_text(user_id) == "# User profile\n\nname: Alice\n"
    assert profile_path.exists()
    assert store.file_size(user_id) == profile_path.stat().st_size
    assert store.facts(user_id)["name"] == "Alice"

    assert store.edit_text(user_id, "name: Alice", "name: Alice Nguyen")
    assert store.facts(user_id)["name"] == "Alice Nguyen"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify a long thread triggers compaction."""

    manager = CompactMemoryManager(threshold_tokens=20, keep_messages=2)
    for idx in range(10):
        manager.append("thread-1", "user", f"message {idx} " * 6)

    assert manager.compaction_count("thread-1") >= 1


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced recalls across sessions while baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    user_id = "alice"

    for turn in [
        "Mình tên là Alice.",
        "Mình thích cà phê sữa đá.",
        "Mình đang ở Hà Nội.",
    ]:
        advanced.reply(user_id, "thread-a", turn)
        baseline.reply(user_id, "thread-a", turn)

    answer = advanced.reply(user_id, "thread-b", "Mình tên gì và đồ uống yêu thích là gì?")
    baseline_answer = baseline.reply(user_id, "thread-b", "Mình tên gì và đồ uống yêu thích là gì?")

    assert "Alice" in answer["response"]
    assert "cà phê sữa đá" in answer["response"].lower()
    assert "Alice" not in baseline_answer["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify compaction reduces prompt load on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    long_thread = "long-demo"
    user_id = "long-user"
    for i in range(30):
        baseline.reply(user_id, long_thread, f"Cuộc hội thoại dài thứ {i}, mình đang luyện tập kỹ năng AI và memory.")
        advanced.reply(user_id, long_thread, f"Cuộc hội thoại dài thứ {i}, mình đang luyện tập kỹ năng AI và memory.")

    assert advanced.compaction_count(long_thread) > 0
    assert advanced.prompt_token_usage(long_thread) <= baseline.prompt_token_usage(long_thread)
