"""Safety guard, dispatcher and control loop (plan section 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gui_agent.actions import Action, ActionType
from gui_agent.config import HarnessConfig, SafetyConfig
from gui_agent.harness.dispatch import Dispatcher, NullBackend
from gui_agent.harness.loop import ControlLoop, SubtaskStatus, frame_hash, hamming
from gui_agent.harness.mcp_tool import TOOL_SCHEMA, LowLevelTaskExecutor
from gui_agent.harness.safety import ActionGuard, SafetyViolation, detect_sandbox
from gui_agent.harness.verifier import CallableVerifier

SCREEN = (1920, 1080)


@pytest.fixture
def guard(safety_config, codec):
    return ActionGuard(safety_config, codec, screen_size=SCREEN)


@pytest.fixture
def dispatcher(safety_config, codec, guard):
    return Dispatcher(NullBackend(SCREEN), guard, codec, safety_config)


class TestActionGuard:
    def test_ordinary_click_is_allowed(self, guard, codec):
        assert guard.check(Action(ActionType.MOVE_CLICK, row=2, col=2, offset=0), "chrome")

    def test_unknown_foreground_is_refused(self, guard):
        # The app blocklist cannot be enforced against a window we cannot see.
        verdict = guard.check(Action(ActionType.MOVE_CLICK, row=2, col=2, offset=0), None)
        assert not verdict and verdict.reason == "unknown_foreground"

    def test_blocked_app_is_refused(self, guard):
        verdict = guard.check(Action(ActionType.MOVE_CLICK, row=2, col=2, offset=0), "gnome-terminal")
        assert verdict.reason == "blocked_app"

    def test_blocked_key_is_refused(self, guard):
        assert guard.check(Action(ActionType.KEY, key="ALT_TAB"), "chrome").reason == "blocked_key"

    def test_destructive_text_is_refused(self, guard):
        verdict = guard.check(Action(ActionType.TYPE, text="sudo rm -rf /"), "chrome")
        assert verdict.reason == "blocked_text"

    def test_block_detail_never_contains_the_text(self, guard):
        # form_data may hold a password; the audit trail must not echo it.
        verdict = guard.check(Action(ActionType.TYPE, text="sudo rm -rf /secret"), "chrome")
        assert "secret" not in (verdict.detail or "")

    def test_field_fill_is_allowed_and_resolvable(self, guard):
        action = Action(ActionType.TYPE, field="email")
        assert guard.check(action, "chrome", {"email": "j@x.com"})

    def test_field_fill_without_the_value_is_refused(self, guard):
        action = Action(ActionType.TYPE, field="email")
        assert guard.check(action, "chrome", {}).reason == "unresolvable_text"

    def test_region_restriction(self, safety_config, codec):
        config = SafetyConfig(**{**safety_config.to_dict(), "allowed_region": (0, 0, 960, 540)})
        guard = ActionGuard(config, codec, screen_size=SCREEN)
        assert guard.check(Action(ActionType.MOVE_CLICK, row=0, col=0, offset=0), "chrome")
        assert guard.check(
            Action(ActionType.MOVE_CLICK, row=7, col=7, offset=0), "chrome"
        ).reason == "outside_region"

    def test_action_budget_is_enforced(self, safety_config, codec):
        config = SafetyConfig(**{**safety_config.to_dict(), "max_actions_per_subtask": 2})
        guard = ActionGuard(config, codec, screen_size=SCREEN)
        action = Action(ActionType.KEY, key="TAB")
        for _ in range(2):
            assert guard.check(action, "chrome")
            guard.note_dispatch(action)
        assert guard.check(action, "chrome").reason == "action_budget"

    def test_rate_limit_is_enforced(self, safety_config, codec):
        config = SafetyConfig(**{**safety_config.to_dict(), "max_actions_per_second": 2})
        guard = ActionGuard(config, codec, screen_size=SCREEN)
        action = Action(ActionType.KEY, key="TAB")
        for _ in range(2):
            guard.note_dispatch(action, now=100.0)
        assert guard.check(action, "chrome", now=100.1).reason == "rate_limit"

    def test_reset_clears_subtask_state(self, safety_config, codec):
        guard = ActionGuard(safety_config, codec, screen_size=SCREEN)
        guard.note_dispatch(Action(ActionType.KEY, key="TAB"))
        guard.reset_subtask()
        assert guard.stats.dispatched == 0

    def test_audit_log_records_blocks(self, tmp_path, codec):
        path = tmp_path / "audit.jsonl"
        config = SafetyConfig(audit_log=str(path), require_sandbox=False)
        guard = ActionGuard(config, codec, screen_size=SCREEN)
        guard.check(Action(ActionType.KEY, key="ALT_TAB"), "chrome")
        assert "blocked_key" in path.read_text()

    def test_preflight_refuses_outside_a_sandbox(self, codec, monkeypatch):
        monkeypatch.delenv("GUI_AGENT_ALLOW_UNSANDBOXED", raising=False)
        monkeypatch.delenv("GUI_AGENT_SANDBOX", raising=False)
        monkeypatch.setattr(
            "gui_agent.harness.safety.detect_sandbox",
            lambda: type("V", (), {"allowed": False, "reason": "not sandboxed"})(),
        )
        guard = ActionGuard(SafetyConfig(audit_log=None, require_sandbox=True), codec, SCREEN)
        with pytest.raises(SafetyViolation, match="VM"):
            guard.preflight()

    def test_explicit_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("GUI_AGENT_ALLOW_UNSANDBOXED", "1")
        assert detect_sandbox().allowed


class TestDispatcher:
    def test_every_action_type_reaches_the_backend(self, dispatcher):
        backend = dispatcher.backend
        actions = [
            Action(ActionType.MOVE_CLICK, row=2, col=2, offset=0),
            Action(ActionType.DOUBLE_CLICK, row=2, col=2, offset=0),
            Action(ActionType.RIGHT_CLICK, row=2, col=2, offset=0),
            Action(ActionType.MOVE_REL, dx=32, dy=-16),
            Action(ActionType.SCROLL, scroll_dx=0, scroll_dy=-2),
            Action(ActionType.KEY, key="ENTER"),
            Action(ActionType.TYPE, field="email"),
        ]
        for action in actions:
            assert dispatcher.send(action, "chrome", {"email": "j@x.com"}).dispatched
        assert [c[0] for c in backend.calls] == [
            "click", "click", "click", "move_relative", "scroll", "key", "type_text",
        ]

    def test_field_fill_types_the_exact_supplied_value(self, dispatcher):
        dispatcher.send(Action(ActionType.TYPE, field="email"), "chrome", {"email": "j@x.com"})
        assert dispatcher.backend.calls[-1] == ("type_text", ("j@x.com",), {})

    def test_key_chords_expand(self, dispatcher):
        dispatcher.send(Action(ActionType.KEY, key="CTRL_C"), "chrome")
        assert dispatcher.backend.calls[-1] == ("key", (("ctrl", "c"),), {})

    def test_blocked_actions_never_reach_the_backend(self, dispatcher):
        result = dispatcher.send(Action(ActionType.KEY, key="ALT_TAB"), "chrome")
        assert not result.dispatched and result.blocked
        assert dispatcher.backend.calls == []

    def test_terminal_actions_are_not_dispatched(self, dispatcher):
        assert not dispatcher.send(Action(ActionType.DONE), "chrome").dispatched
        assert dispatcher.backend.calls == []

    def test_open_drags_are_released(self, dispatcher):
        dispatcher.send(Action(ActionType.DRAG_START, row=1, col=1, offset=0), "chrome")
        assert dispatcher.drag is not None
        dispatcher.reset_subtask()
        assert dispatcher.drag is None
        assert dispatcher.backend.calls[-1][0] == "mouse_up"


class TestFrameHash:
    def test_identical_frames_hash_equal(self, frame):
        assert hamming(frame_hash(frame), frame_hash(frame)) == 0

    def test_a_single_pixel_does_not_read_as_a_change(self, frame):
        # A blinking caret must not defeat the stuck detector.
        other = frame.copy()
        other[0, 0] = 255 - other[0, 0]
        assert hamming(frame_hash(frame), frame_hash(other)) <= 2

    def test_different_frames_hash_far_apart(self, frame):
        other = np.random.RandomState(1).randint(0, 255, frame.shape, dtype=np.uint8)
        assert hamming(frame_hash(frame), frame_hash(other)) > 20


@dataclass
class _Result:
    action: Action
    atoms: list
    error: str | None = None

    @property
    def ok(self):
        return self.action is not None


class _ScriptedPolicy:
    def __init__(self, actions, codec):
        self.actions = list(actions)
        self.codec = codec
        self.index = 0

    def make_session(self, *args, **kwargs):
        return None

    def act(self, frame, instruction, form_data, history, session=None):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return _Result(action, self.codec.encode(action, list((form_data or {}).keys())))


class TestControlLoop:
    def _loop(self, actions, frames, codec, safety_config, verifier=None, app="chrome",
              config=None):
        dispatcher = Dispatcher(
            NullBackend(SCREEN), ActionGuard(safety_config, codec, SCREEN), codec, safety_config
        )
        iterator, last = iter(frames), [None]

        def capture():
            try:
                last[0] = next(iterator)
            except StopIteration:
                pass
            return last[0]

        return ControlLoop(
            _ScriptedPolicy(actions, codec), dispatcher, capture,
            config or HarnessConfig(target_hz=0, stuck_frames=3),
            verifier=verifier, foreground=lambda: app,
        )

    @staticmethod
    def _frames(n, seed=0):
        return [
            np.random.RandomState(seed + i).randint(0, 255, (60, 80, 3), dtype=np.uint8)
            for i in range(n)
        ]

    def test_done_is_verified_before_being_reported(self, codec, safety_config):
        actions = [Action(ActionType.MOVE_CLICK, row=1, col=1, offset=0), Action(ActionType.DONE)]
        loop = self._loop(actions, self._frames(2), codec, safety_config,
                          CallableVerifier(lambda c, s, i: True))
        result = loop.run("click submit", success_criteria="the form is submitted")
        assert result.status is SubtaskStatus.DONE
        assert result.verification.met and result.verification.checked

    def test_done_with_unmet_criteria_escalates(self, codec, safety_config):
        # A drifted policy emits DONE confidently at the wrong moment.
        actions = [Action(ActionType.DONE)]
        loop = self._loop(actions, self._frames(1), codec, safety_config,
                          CallableVerifier(lambda c, s, i: False))
        result = loop.run("click submit", success_criteria="the form is submitted")
        assert result.status is SubtaskStatus.ESCALATED

    def test_escalate_token_stops_the_loop(self, codec, safety_config):
        loop = self._loop([Action(ActionType.ESCALATE)], self._frames(1), codec, safety_config)
        assert loop.run("x").status is SubtaskStatus.ESCALATED

    def test_unchanging_screen_escalates(self, codec, safety_config):
        frames = [self._frames(1)[0]] * 30
        actions = [Action(ActionType.MOVE_CLICK, row=1, col=c % 8, offset=0) for c in range(30)]
        result = self._loop(actions, frames, codec, safety_config).run("x")
        assert result.status is SubtaskStatus.ESCALATED
        assert "no visual change" in result.stop_reason

    def test_wait_does_not_count_toward_stuck(self, codec, safety_config):
        # Counting WAIT as stuck would escalate out of every page load.
        frames = [self._frames(1)[0]] * 30
        actions = [Action(ActionType.WAIT, wait_ms=100)] * 6 + [Action(ActionType.DONE)]
        result = self._loop(actions, frames, codec, safety_config).run("x")
        assert result.status is SubtaskStatus.DONE

    def test_repeated_action_escalates_even_when_the_screen_animates(self, codec, safety_config):
        actions = [Action(ActionType.MOVE_CLICK, row=1, col=1, offset=0)] * 30
        result = self._loop(actions, self._frames(30), codec, safety_config).run("x")
        assert result.status is SubtaskStatus.ESCALATED
        assert "repeated" in result.stop_reason

    def test_timeout(self, codec, safety_config):
        actions = [Action(ActionType.MOVE_CLICK, row=1, col=1, offset=0)] * 50
        result = self._loop(actions, self._frames(50), codec, safety_config).run("x", timeout_s=0.0)
        assert result.status is SubtaskStatus.TIMEOUT

    def test_repeated_safety_blocks_escalate_with_the_reason(self, codec, safety_config):
        actions = [Action(ActionType.KEY, key="ALT_TAB")] * 20
        result = self._loop(actions, self._frames(20), codec, safety_config).run("x")
        assert result.status is SubtaskStatus.ESCALATED
        assert "blocked_key" in result.stop_reason
        # The trace must carry why, or stage 3 has nothing to learn from.
        assert all(t.blocked_reason == "blocked_key" for t in result.trace.ticks)

    def test_trace_records_latency(self, codec, safety_config):
        actions = [Action(ActionType.MOVE_CLICK, row=1, col=1, offset=0), Action(ActionType.DONE)]
        result = self._loop(actions, self._frames(2), codec, safety_config).run("x")
        assert result.trace.latency_percentiles()["n"] == 2


class TestMcpTool:
    def test_schema_matches_the_documented_interface(self):
        assert set(TOOL_SCHEMA["properties"]) == {
            "instruction", "form_data", "timeout_s", "success_criteria"
        }
        assert TOOL_SCHEMA["required"] == ["instruction"]

    def test_empty_instruction_is_rejected_without_running(self, codec, safety_config, tmp_path):
        loop = TestControlLoop()._loop([Action(ActionType.DONE)], [], codec, safety_config)
        executor = LowLevelTaskExecutor(loop, trace_dir=tmp_path)
        assert executor.execute("  ")["status"] == "error"
        assert executor.stats.calls == 0

    def test_payload_omits_the_per_frame_trace(self, codec, safety_config, tmp_path):
        # Returning the trace would undo the context saving this tool exists for.
        frames = TestControlLoop._frames(2)
        loop = TestControlLoop()._loop(
            [Action(ActionType.MOVE_CLICK, row=1, col=1, offset=0), Action(ActionType.DONE)],
            frames, codec, safety_config,
        )
        executor = LowLevelTaskExecutor(loop, trace_dir=tmp_path, return_screenshot=False)
        payload = executor.execute("click submit")
        assert "ticks" not in payload and "trace" not in payload
        assert payload["status"] == "done"
        # ...but it is on disk for stage 3.
        assert (tmp_path / f"{payload['rollout_id']}.json").exists()
