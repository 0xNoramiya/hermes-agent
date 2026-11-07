import pytest

from agent.message_metadata import (
    INFLIGHT_TASK_RESTATEMENT, TYPED_ASK, append_message, is_compaction_scaffolding, replay_metadata_fields,
    stamp_message_timestamp,
)


def test_stamp_preserves_source_timestamp(monkeypatch):
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: 999.0)
    message = {"role": "user", "content": "hello", "timestamp": 123.0}

    result = stamp_message_timestamp(message)

    assert result is message
    assert message["timestamp"] == 123.0


def test_append_stamps_same_mapping_at_append_time(monkeypatch):
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: 456.0)
    messages = []
    message = {"role": "tool", "content": "ok"}

    result = append_message(messages, message)

    assert result is message
    assert messages == [message]
    assert message["timestamp"] == 456.0


def test_replay_keeps_a_user_rows_typed_ask_beside_the_scaffolding_flag():
    """Gateway replay carries the stamp (previous_message reads it after a restart) without
    turning the row into compaction scaffolding, which would skip its timestamp."""
    stamped = {"role": "user", "content": "x", "display_metadata": {TYPED_ASK: "ask", "other": 1}}
    both = {"role": "user", "content": "x", "display_metadata": {TYPED_ASK: "ask", INFLIGHT_TASK_RESTATEMENT: True}}

    assert replay_metadata_fields(stamped) == {"display_metadata": {TYPED_ASK: "ask"}}
    assert not is_compaction_scaffolding(stamped)
    assert replay_metadata_fields(both) == {"display_metadata": {INFLIGHT_TASK_RESTATEMENT: True, TYPED_ASK: "ask"}}
    assert replay_metadata_fields({"role": "assistant", "content": "x"}) == {}


_NOTES = "PLUGIN NOTE"


def _stamped(ask, role="user"):
    return {"role": role, "content": f"{ask}\n[screenshot]\n\n{_NOTES}", "display_metadata": {TYPED_ASK: ask}}


def _previous(rows):
    from agent.turn_context import _previous_user_message_text
    return _previous_user_message_text([*rows, {"role": "assistant", "content": "x"}])


def _unstamped(rows):
    return [{**r, "display_metadata": {k: v for k, v in (r.get("display_metadata") or {}).items() if k != TYPED_ASK}}
            for r in rows]


def _carrier(role, row=None, force_user_leading=False):
    from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor
    row = row or {"role": role, "content": "ASSISTANT TEXT" if role == "assistant" else "earlier ask"}
    if role == "assistant":
        row["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor._summary_has_user_turn, compressor.quiet_mode = True, True
    compressor._merge_summary_into_tail_row(row, f"{SUMMARY_PREFIX}\nold work", role, force_user_leading)
    return compressor, row


def _repair_users(first, second):
    from agent.agent_runtime_helpers import _merge_consecutive_users
    return _merge_consecutive_users([first, second, {"role": "assistant", "content": "x"}])[0][:1]


def _repair_assistants(first, second):
    from agent.agent_runtime_helpers import _merge_consecutive_assistants
    return _merge_consecutive_assistants([{"role": "user", "content": "u"}, first, second])[0][-1:]


def _micro(first, second):
    from types import SimpleNamespace
    from agent.micro_compaction import MicroCompactionMixin
    return MicroCompactionMixin._merge_adjacent_user_turns(
        SimpleNamespace(_flush_scan_cursor_invalidated=False), [first, second, {"role": "assistant", "content": "x"}])[:1]


def _carrier_fold(role):
    def fold(_first, inflight):
        compressor, carrier = _carrier(role)
        compressor._reappend_inflight_user_task([{"role": "user", "content": "earlier"}, carrier], inflight)
        return [carrier]
    return fold


def _anchor_fold(anchor, _second):
    from agent.conversation_compression import _merge_anchor_into_user_message
    from tools.todo_tool import TODO_INJECTION_HEADER
    target = {"role": "user", "content": f"{TODO_INJECTION_HEADER}\n- [ ] t", "_todo_snapshot_synthetic": True}
    _merge_anchor_into_user_message(target, anchor)
    return [target]


_FOLDS = {
    "repair-stamped-then-text": (_repair_users, lambda: _stamped("IMAGE ASK"), lambda: {"role": "user", "content": "TEXT ASK"}),
    "repair-text-then-stamped": (_repair_users, lambda: {"role": "user", "content": "TEXT ASK"}, lambda: _stamped("IMAGE ASK")),
    "repair-both-stamped": (_repair_users, lambda: _stamped("FIRST ASK"), lambda: _stamped("SECOND ASK")),
    "repair-assistants": (_repair_assistants, lambda: {"role": "assistant", "content": "OTHER TEXT"}, lambda: _stamped("IMAGE ASK", "assistant")),
    "micro": (_micro, lambda: {"role": "user", "content": "TEXT ASK"}, lambda: _stamped("IMAGE ASK")),
    "carrier-user": (_carrier_fold("user"), lambda: None, lambda: _stamped("IMAGE ASK")),
    "carrier-assistant": (_carrier_fold("assistant"), lambda: None, lambda: _stamped("IMAGE ASK")),
    "anchor": (_anchor_fold, lambda: _stamped("IMAGE ASK"), lambda: None),
}


@pytest.mark.parametrize("case", list(_FOLDS))
def test_a_fold_drops_the_typed_ask_so_the_row_reads_as_without_stamps(case):
    """A folded row's content holds more than one turn's ask; a stamp that tried to follow it drifted from the
    content (an assistant's text, a todo snapshot, an ask the next restatement drops). Every host fold drops the
    stamp, so previous_message reads the row exactly as if no turn had been stamped."""
    fold, first, second = _FOLDS[case]
    stamped = fold(first(), second())
    plain = fold(*(_unstamped([row])[0] if row else None for row in (first(), second())))

    assert all(TYPED_ASK not in (row.get("display_metadata") or {}) for row in stamped)
    assert _previous(stamped) == _previous(plain)


@pytest.mark.parametrize("force_user_leading", [True, False], ids=["summary-first", "prior-context-first"])
def test_a_wrapped_turn_restated_by_the_next_compaction_keeps_its_typed_ask(force_user_leading):
    """Compaction wraps the in-flight stamped turn (no fold: the row is still that one turn) and the next
    compaction restates it standalone: the restatement keeps the turn's ask, never its notes."""
    compressor, wrapped = _carrier("user", _stamped("IMAGE ASK"), force_user_leading)
    assert wrapped["display_metadata"][TYPED_ASK] == "IMAGE ASK"
    restated = compressor._reappend_inflight_user_task([{**_summary(), "role": "assistant"}], wrapped)[-1]

    assert restated is not wrapped and restated["role"] == "user"
    assert _previous([restated]) == "IMAGE ASK"


def _summary():
    from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\nold work\n{_SUMMARY_END_MARKER}", "_compressed_summary": True}
