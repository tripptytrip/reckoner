"""reckoner — math in base-625, trained by search against a checker.

Experiment two, Stage A: a single specialist that learns multi-step mathematics
by search over discrete rewrite actions, with an external verifier as the only
source of truth. See ``experiment2_math_base625_spec.md`` for the spec and
``experiment2_agent_plan.md`` for the chunk plan.
"""

from __future__ import annotations

__version__ = "0.1.0"
