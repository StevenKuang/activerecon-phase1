"""Minimal executable ActiveAgent: rotate in place using camera-frame actions.

This exercises the integration contract; it is not a competitive exploration
method. Load with --agent examples/methods/spin_agent.py:build_agent.
"""

import math

from activebench.api import AgentAction, MethodInfo


class SpinAgent:
    def __init__(self, turn_degrees=30.0):
        if not 0.0 < float(turn_degrees) <= 180.0:
            raise ValueError("turn_degrees must be in (0, 180]")
        self.turn = math.radians(float(turn_degrees))

    def info(self):
        return MethodInfo(name="example-spin", needs_depth=False, pose_access="none")

    def reset(self, seed, task=None):
        # A stochastic policy should initialize its RNG here, from seed.
        self.decisions = 0

    def act(self, observation):
        self.decisions += 1
        return AgentAction.move_rel([0.0, 0.0, 0.0], dyaw=self.turn)


def build_agent(options):
    # The launcher also supplies seed, camera intrinsics, scene bounds and
    # budget. Read only what your method needs; document privileged inputs.
    return SpinAgent(turn_degrees=options.get("turn_degrees", 30.0))
