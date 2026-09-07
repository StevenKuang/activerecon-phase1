"""Subprocess RPC boundary: run a method agent in its own conda env.

The proxy speaks newline-delimited JSON over the worker's stdin/stdout; pixel
arrays travel as file paths (the runner already persists every capture).
Method code is free to print or spawn progress bars: the worker re-points
Python-level stdout to stderr at startup and keeps a private duplicate of the
real stdout for protocol frames.

Requests:  {"op": "info"} | {"op": "reset", "seed": .., "task": ..}
           | {"op": "act", "observation": {...}} | {"op": "close"}
Responses: {"ok": true, "result": ...} | {"ok": false, "error": "..."}
"""

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, IO, Optional

from activebench.api import AgentAction, MethodInfo, Observation


class AgentProcessProxy:
    """ActiveAgent facade that forwards calls to an agent in a subprocess."""

    def __init__(
        self,
        agent_name: str,
        options: Dict[str, Any],
        python_exe: str,
        repo_src: Optional[str] = None,
        stderr_log: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        repo_src = repo_src or str(Path(__file__).resolve().parent.parent)
        worker_env = dict(os.environ)
        worker_env["PYTHONPATH"] = repo_src + os.pathsep + worker_env.get("PYTHONPATH", "")
        worker_env["PYTHONFAULTHANDLER"] = "1"
        worker_env.update(env or {})

        self._stderr_log = Path(stderr_log) if stderr_log else None
        self._stderr_file: Optional[IO[bytes]] = None
        if self._stderr_log is not None:
            self._stderr_log.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = open(self._stderr_log, "wb")

        self._proc = subprocess.Popen(
            [python_exe, "-m", "activebench.rpc", agent_name, json.dumps(options)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file if self._stderr_file is not None else None,
            env=worker_env,
            text=True,
        )

    def _call(self, request: Dict[str, Any]) -> Any:
        proc = self._proc
        if proc.poll() is not None:
            raise RuntimeError(self._death_notice("worker already exited"))
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(self._death_notice("worker closed its pipe"))
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError("agent worker error: %s" % response.get("error"))
        return response.get("result")

    def _death_notice(self, reason: str) -> str:
        detail = ""
        if self._stderr_log is not None and self._stderr_log.exists():
            tail = self._stderr_log.read_text(errors="replace").splitlines()[-15:]
            detail = "\nworker stderr tail:\n" + "\n".join(tail)
        code = self._proc.poll()
        return "%s (exit code %s)%s" % (reason, code, detail)

    # -- ActiveAgent protocol -------------------------------------------------

    def info(self) -> MethodInfo:
        result = self._call({"op": "info"})
        return MethodInfo(**result)

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        self._call({"op": "reset", "seed": int(seed), "task": task})

    def act(self, observation: Observation) -> AgentAction:
        if observation.rgb_path is None:
            raise ValueError("RPC transport needs observation.rgb_path to be set")
        result = self._call({"op": "act", "observation": observation.to_dict()})
        return AgentAction.from_dict(result)

    def policy_state(self, observation: Observation, *, remaining_fraction: float) -> Dict[str, Any]:
        """Request a bounded policy feature state from a specialized worker."""

        if observation.rgb_path is None:
            raise ValueError("RPC transport needs observation.rgb_path to be set")
        result = self._call(
            {
                "op": "policy_state",
                "observation": observation.to_dict(),
                "remaining_fraction": float(remaining_fraction),
            }
        )
        if not isinstance(result, dict):
            raise RuntimeError("policy_state worker result must be an object")
        return result

    def record_macro_action(self, slot: int) -> None:
        """Advance a specialized worker's bounded action-history feature."""

        self._call({"op": "record_macro_action", "slot": int(slot)})

    def decision_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Fetch optional post-decision diagnostics without exposing them to policy input."""

        result = self._call({"op": "decision_diagnostics"})
        if result is None:
            return None
        if not isinstance(result, dict):
            raise RuntimeError("decision diagnostics worker result must be an object")
        return result

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._call({"op": "close"})
            except RuntimeError:
                pass
            self._proc.wait(timeout=30)
        if self._stderr_file is not None:
            self._stderr_file.close()

    def __del__(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass


def _worker_main() -> int:
    # Keep a private handle on the real stdout for protocol frames, then send
    # everything the method prints (Python and C level) to stderr.
    proto_out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    from activebench.registry import build_agent

    agent_name = sys.argv[1]
    options = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    agent = build_agent(agent_name, options)

    def respond(payload: Dict[str, Any]) -> None:
        proto_out.write(json.dumps(payload) + "\n")
        proto_out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "info":
                respond({"ok": True, "result": agent.info().__dict__})
            elif op == "reset":
                agent.reset(seed=request["seed"], task=request.get("task"))
                respond({"ok": True})
            elif op == "act":
                observation = Observation.from_dict(request["observation"])
                action = agent.act(observation)
                respond({"ok": True, "result": action.to_dict()})
            elif op == "policy_state":
                if not hasattr(agent, "policy_state"):
                    raise ValueError("agent does not implement policy_state")
                observation = Observation.from_dict(request["observation"])
                result = agent.policy_state(
                    observation, remaining_fraction=float(request["remaining_fraction"])
                )
                respond({"ok": True, "result": result})
            elif op == "record_macro_action":
                if not hasattr(agent, "record_macro_action"):
                    raise ValueError("agent does not implement record_macro_action")
                agent.record_macro_action(int(request["slot"]))
                respond({"ok": True})
            elif op == "decision_diagnostics":
                callback = getattr(agent, "decision_diagnostics", None)
                respond({"ok": True, "result": callback() if callable(callback) else None})
            elif op == "close":
                respond({"ok": True})
                return 0
            else:
                respond({"ok": False, "error": "unknown op %r" % op})
        except Exception as exc:  # keep serving; the proxy decides what's fatal
            traceback.print_exc()
            respond({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main())
