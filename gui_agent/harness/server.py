"""MCP server exposing ``execute_low_level_task``.

    python -m gui_agent.harness.server --policy runs/stage2/best

Wires a trained policy to the live screen and serves the single tool over
stdio.  Safety preflight runs before the server accepts anything: the policy
dispatches un-reviewed input, so a misconfigured sandbox has to fail at startup
rather than on the first tool call.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..config import HarnessConfig
from .dispatch import Dispatcher, build_backend
from .loop import ControlLoop
from .mcp_tool import TOOL_DESCRIPTION, TOOL_NAME, TOOL_SCHEMA, LowLevelTaskExecutor
from .safety import ActionGuard
from .verifier import AlwaysTrueVerifier, LLMVerifier

log = logging.getLogger(__name__)


def build_executor(
    policy_path: str | None,
    harness: HarnessConfig | None = None,
    backend_name: str = "auto",
    use_llm_verifier: bool = True,
    trace_dir: str | None = "runs/rollouts",
) -> LowLevelTaskExecutor:
    harness = harness or HarnessConfig()

    backend = build_backend(backend_name)
    guard = ActionGuard(harness.safety, screen_size=backend.screen_size())
    guard.preflight()
    dispatcher = Dispatcher(backend, guard, config=harness.safety)

    if policy_path:
        from ..model.policy import GuiPolicy

        policy = GuiPolicy.load(policy_path)
        policy.eval()
    else:
        raise SystemExit(
            "no policy given. Pass --policy <checkpoint dir>; there is no useful "
            "default, and an untrained policy driving the screen is not a safe fallback."
        )

    from ..capture.screen import ScreenGrabber

    grabber = ScreenGrabber().open()

    def capture():
        frame = grabber.grab()
        import numpy as np

        array = np.frombuffer(frame.data, dtype=np.uint8).reshape(
            frame.height, frame.width, 4
        )
        return array[:, :, :3][:, :, ::-1]  # BGRA -> RGB

    verifier = LLMVerifier() if use_llm_verifier else AlwaysTrueVerifier()
    loop = ControlLoop(policy, dispatcher, capture, harness, verifier=verifier)
    return LowLevelTaskExecutor(loop, harness, trace_dir=trace_dir)


def serve(executor: LowLevelTaskExecutor) -> None:
    """Serve the tool over MCP stdio."""
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError:
        raise SystemExit(
            "the mcp package is not installed (pip install 'gui-agent[mcp]')"
        ) from None

    server = FastMCP("gui-agent")

    @server.tool(name=TOOL_NAME, description=TOOL_DESCRIPTION)
    def execute_low_level_task(
        instruction: str,
        form_data: dict | None = None,
        timeout_s: float | None = None,
        success_criteria: str = "",
    ) -> dict:
        return executor.execute(instruction, form_data, timeout_s, success_criteria)

    log.info("serving %s over stdio", TOOL_NAME)
    server.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gui-agent-mcp", description=__doc__)
    parser.add_argument("--policy", required=False, help="trained policy checkpoint directory")
    parser.add_argument("--harness-config", default=None, help="HarnessConfig JSON")
    parser.add_argument("--backend", default="auto", choices=["auto", "pyautogui", "xdotool", "null"])
    parser.add_argument("--no-verifier", action="store_true",
                        help="skip the independent success check (DONE is trusted)")
    parser.add_argument("--trace-dir", default="runs/rollouts")
    parser.add_argument("--print-schema", action="store_true",
                        help="print the tool definition and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stdout is the MCP transport
    )

    if args.print_schema:
        import json

        print(json.dumps(
            {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": TOOL_SCHEMA},
            indent=2,
        ))
        return 0

    harness = HarnessConfig.load(args.harness_config) if args.harness_config else HarnessConfig()
    executor = build_executor(
        args.policy, harness, args.backend,
        use_llm_verifier=not args.no_verifier, trace_dir=args.trace_dir,
    )
    serve(executor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
