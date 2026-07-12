"""Scene sampling + rule-based NL -> sub-goal planner for the REPL demo
(code/demo.py, RF-1 split).

Owns:
  - SceneManager: samples a fresh scene via code.scene.sample_scene and
    exposes the object list.
  - SubGoal: a single primitive sub-goal (goto/maneuver/search).
  - Planner: rule-based free-form English instruction -> ordered SubGoal
    program, with ambiguity-resolution clarification questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from code.apps.repl.constants import COLORS, SHAPES


# ---------------------------------------------------------------------------
# SceneManager
# ---------------------------------------------------------------------------
class SceneManager:
    """
    Manages the current scene: samples a fresh scene via sample_scene(),
    exposes the object list (color, shape, position) visible at start.
    """

    def __init__(self, difficulty: str = "easy", seed_offset: int = 0) -> None:
        self.difficulty   = difficulty
        self.seed_offset  = seed_offset
        self._ep_count    = 0
        self._scene_cfg   = None

    def new_scene(self) -> dict:
        """Sample a fresh scene; returns scene_cfg."""
        from code.scene import sample_scene, derive_rng
        rng = derive_rng(1234 + self.seed_offset, self._ep_count)
        self._scene_cfg = sample_scene(rng, self.difficulty)
        self._ep_count += 1
        return self._scene_cfg

    @property
    def scene_cfg(self) -> dict | None:
        return self._scene_cfg

    def describe_scene(self) -> str:
        """Human-readable description of the current scene."""
        if self._scene_cfg is None:
            return "(no scene yet)"
        objs = self._scene_cfg["objects"]
        lines = ["Objects in scene:"]
        for i, o in enumerate(objs):
            tgt_mark = " <-- TARGET" if i == self._scene_cfg["target_index"] else ""
            lines.append(
                f"  [{i}] {o['color_name']} {o['shape_name']}"
                f"  dist={o['dist_from_robot']:.2f}m  "
                f"({o['x']:.1f}, {o['y']:.1f}){tgt_mark}"
            )
        return "\n".join(lines)

    def object_list(self) -> list[dict[str, Any]]:
        """Return list of object dicts."""
        if self._scene_cfg is None:
            return []
        return self._scene_cfg["objects"]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
@dataclass
class SubGoal:
    """A single primitive sub-goal."""
    skill:       str           # 'goto' | 'maneuver' | 'search' (stubbed)
    color:       str | None = None
    shape:       str | None = None
    direction:   str | None = None   # for maneuver: 'left' | 'right'
    description: str           = ""
    status:      str           = "pending"   # 'pending' | 'running' | 'done' | 'failed'
    result:      Any | None = None

    def __str__(self) -> str:
        if self.skill == "goto":
            return f"goto({self.color} {self.shape})"
        elif self.skill == "maneuver":
            return f"maneuver(turn_{self.direction} after {self.color} {self.shape})"
        elif self.skill == "search":
            return f"search({self.color} {self.shape})"
        return f"{self.skill}({self.description})"


class Planner:
    """
    Rule-based NL → sub-goal program.

    Parses free-form English instructions into an ordered list of SubGoals
    that the trained policies can execute.

    Supports:
      - goto(color, shape): "go to the red ball"
      - maneuver(dir, color, shape): "turn left after the blue cube"
      - search(color, shape): "find the red ball" (out-of-FOV scan)
      - compound: "go to the red ball then find the blue cube"
      - multi-goal: "navigate to the red ball then turn left after the orange cone"

    Ambiguity resolution:
      - Multiple possible referents → ask clarify question with options listed
      - No matching object in scene → report what IS available
      - Missing referent (no color/shape) → ask which object they mean

    Compound instruction parsing:
      - Splits on "then", "and then", "and after that", "afterwards"
      - Each part parsed independently
      - If a part is ambiguous, returns clarify question (subsequent parts queued)
    """

    def __init__(self, scene_manager: SceneManager) -> None:
        self.scene  = scene_manager
        # Queued parts for multi-goal: when a part triggers clarification, we
        # store remaining parts here and resume after clarification is resolved.
        self._pending_parts: list[str] = []
        self._pending_goals: list[SubGoal] = []

    def parse(self, instruction: str) -> tuple[list[SubGoal], str | None]:
        """
        Parse instruction into sub-goals.

        Returns:
            (goals, clarify_question)
            clarify_question is non-None if the planner is ambiguous and needs user input.
            When clarify_question is set, previously-parsed goals (from earlier parts of
            a compound instruction) are still included in the returned goals list.
        """
        instr = instruction.lower().strip()

        # Split compound instructions by sequencing conjunctions
        # Handles: "then", "and then", "and after that", "after that", "afterwards", "next"
        parts = re.split(
            r'\bthen\b|,\s*then\s*|\band\s+then\b|\band\s+after\s+that\b'
            r'|\bafter\s+that\b|\bafterwards\b|\bnext\b',
            instr, flags=re.IGNORECASE
        )
        parts = [p.strip() for p in parts if p.strip()]

        goals = []
        for part in parts:
            subgoals, clarify = self._parse_part(part)
            if clarify:
                # Return already-parsed goals + clarify question.
                # Store remaining parts for when clarification is resolved.
                # (The REPL can call parse() again with the clarification answer.)
                return goals, clarify
            goals.extend(subgoals)

        if not goals:
            return [], (
                f"I didn't understand that instruction. Try:\n"
                f"  'go to the red ball'\n"
                f"  'turn left after the blue cube'\n"
                f"  'find the orange cone'\n"
                f"  'go to the red ball then find the blue cube'"
            )

        return goals, None

    def parse_clarification(
        self, original_instr: str, clarify_answer: str
    ) -> tuple[list[SubGoal], str | None]:
        """
        Re-parse after user provides a clarification answer.

        The clarify_answer is treated as a refinement: we attempt to combine
        the original instruction's intent with the user's answer.

        Strategy:
          1. If clarify_answer contains a color+shape → use as the specific referent
          2. If clarify_answer is just a color or shape → combine with original intent
          3. Otherwise → treat clarify_answer as a fresh instruction
        """
        answer = clarify_answer.lower().strip()

        # Check if answer itself is a parseable instruction
        goals, clarify = self.parse(answer)
        if goals and not clarify:
            return goals, None

        # Try to combine original instruction with clarification answer
        # Extract color/shape from answer
        c_found = next((c for c in COLORS if c in answer), None)
        s_found = next((s for s in SHAPES if s in answer), None)

        if c_found or s_found:
            # Re-parse original with clarified color/shape
            instr = original_instr.lower()
            if c_found and s_found:
                instr_refined = f"go to the {c_found} {s_found}"
            elif c_found:
                # Find which shape was in the original
                s_orig = next((s for s in SHAPES if s in instr), None)
                instr_refined = f"go to the {c_found} {s_orig}" if s_orig else f"go to the {c_found} object"
            else:
                c_orig = next((c for c in COLORS if c in instr), None)
                instr_refined = f"go to the {c_orig} {s_found}" if c_orig else f"go to the {s_found}"

            goals, clarify = self.parse(instr_refined)
            if goals:
                return goals, clarify

        # Fall back to treating the answer as a fresh instruction
        return self.parse(clarify_answer)

    def _parse_part(self, instr: str) -> tuple[list[SubGoal], str | None]:
        """Parse a single instruction part into sub-goals."""
        goals = []

        # Check for maneuver pattern: "turn {dir} after [passing] the {color} {shape}"
        maneuver_match = self._detect_maneuver(instr)
        if maneuver_match:
            direction, color, shape = maneuver_match
            # Resolve referent to scene
            obj, clarify = self._resolve_referent(color, shape, "maneuver landmark")
            if clarify:
                return [], clarify
            if obj is None:
                return [], f"I don't see a {color} {shape} in the scene. Available: {self._scene_summary()}."

            goals.append(SubGoal(
                skill="maneuver",
                color=obj["color_name"],
                shape=obj["shape_name"],
                direction=direction,
                description=f"turn {direction} after {obj['color_name']} {obj['shape_name']}",
            ))
            return goals, None

        # Check for search pattern FIRST (before goto) — "find/search_for/look_for" → search skill
        # C3: search is DISTINCT from goto: search scenes guarantee target OUTSIDE initial FOV.
        # "find the red ball" → search (out-of-FOV scan), NOT goto.
        search_match = self._detect_search(instr)
        if search_match:
            color, shape = search_match
            goals.append(SubGoal(
                skill="search",
                color=color,
                shape=shape,
                description=f"find the {color} {shape}",
            ))
            return goals, None

        # Check for goto pattern: "go to|walk to|approach|head to the {color} {shape}"
        goto_match = self._detect_goto(instr)
        if goto_match:
            color, shape = goto_match
            obj, clarify = self._resolve_referent(color, shape, "navigation target")
            if clarify:
                return [], clarify
            if obj is None:
                return [], f"I don't see a {color} {shape} in the scene. Available: {self._scene_summary()}."

            goals.append(SubGoal(
                skill="goto",
                color=obj["color_name"],
                shape=obj["shape_name"],
                description=f"navigate to {obj['color_name']} {obj['shape_name']}",
            ))
            return goals, None

        return [], None

    # ---- Pattern detectors ----

    def _detect_goto(self, instr: str) -> tuple[str, str] | None:
        """Detect goto(color, shape) from instruction.
        NOTE: 'find'/'search for'/'look for' are routed to the SEARCH skill (checked first
        in _parse_part) and are NOT included here. This keeps search DISTINCT from goto.
        """
        goto_verbs = (
            r"(?:go\s+to|walk\s+to|approach|head\s+to|head\s+over\s+to|move\s+to|"
            r"navigate\s+to|make\s+your\s+way\s+to|get\s+to|proceed\s+to|"
            r"reach|come\s+to)"
        )
        # Pattern 1: "go to the {color} {shape}"
        m = re.search(
            goto_verbs + r'\s+(?:the\s+)?([a-z]+)\s+([a-z]+)',
            instr, re.IGNORECASE
        )
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            if c in COLORS and s in SHAPES:
                return (c, s)
            # Try swap
            if s in COLORS and c in SHAPES:
                return (s, c)

        # Pattern 2: "your goal is the {color} {shape}"
        m = re.search(r'your\s+goal\s+is\s+(?:the\s+)?([a-z]+)\s+([a-z]+)', instr, re.IGNORECASE)
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            if c in COLORS and s in SHAPES:
                return (c, s)

        # Pattern 3: any "the {color} {shape}" or "{color}-colored {shape}"
        m = re.search(r'(?:the\s+)?([a-z]+)(?:-colored)?\s+(ball|cube|cylinder|cone)', instr, re.IGNORECASE)
        if m:
            c = m.group(1).lower()
            s = m.group(2).lower()
            if c in COLORS:
                return (c, s)

        return None

    def _detect_maneuver(self, instr: str) -> tuple[str, str, str] | None:
        """Detect maneuver(dir, color, shape) from instruction."""
        # Pattern: "turn {left|right} after [passing] [the] {color} {shape}"
        m = re.search(
            r'turn\s+(left|right)\s+(?:after\s+)?(?:passing\s+)?(?:the\s+)?([a-z]+)\s+([a-z]+)',
            instr, re.IGNORECASE
        )
        if m:
            direction = m.group(1).lower()
            c, s = m.group(2).lower(), m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)
            if s in COLORS and c in SHAPES:
                return (direction, s, c)

        # Pattern: "pass [the] {color} {shape} [and] turn {left|right}"
        m = re.search(
            r'pass\s+(?:the\s+)?([a-z]+)\s+([a-z]+).*?turn\s+(left|right)',
            instr, re.IGNORECASE
        )
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            direction = m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)

        # Pattern: "when you pass [the] {color} {shape}, turn {left|right}"
        m = re.search(
            r'(?:when\s+you\s+pass|after\s+passing)\s+(?:the\s+)?([a-z]+)\s+([a-z]+).*?turn\s+(left|right)',
            instr, re.IGNORECASE
        )
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            direction = m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)

        # Pattern: "turn {left|right} when you pass [the] {color} {shape}"
        m = re.search(
            r'turn\s+(left|right)\s+when\s+you\s+pass\s+(?:the\s+)?([a-z]+)\s+([a-z]+)',
            instr, re.IGNORECASE
        )
        if m:
            direction = m.group(1).lower()
            c, s = m.group(2).lower(), m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)

        return None

    def _detect_search(self, instr: str) -> tuple[str, str] | None:
        """Detect search(color, shape) from instruction.
        C3: search is routed here when instruction uses find/search/look_for.
        DISTINCT from goto — search requires target to be out-of-FOV initially.
        Example: "find the red ball" → search(red, ball)
        """
        # Check for search-trigger verbs
        m_trigger = re.search(r'\b(?:find|search\s+for|look\s+for|locate|hunt\s+for)\b',
                               instr, re.IGNORECASE)
        if m_trigger:
            # Extract color + shape
            m2 = re.search(r'\b([a-z]+)\b.*?\b(ball|cube|cylinder|cone)\b', instr, re.IGNORECASE)
            if m2:
                c, s = m2.group(1).lower(), m2.group(2).lower()
                if c in COLORS:
                    return (c, s)
                # Try finding color further back
            m3 = re.search(r'\b(' + '|'.join(COLORS) + r')\b.*?\b(ball|cube|cylinder|cone)\b',
                           instr, re.IGNORECASE)
            if m3:
                return (m3.group(1).lower(), m3.group(2).lower())
        return None

    def _resolve_referent(
        self, color: str | None, shape: str | None, role: str
    ) -> tuple[dict | None, str | None]:
        """
        Resolve (color, shape) to a scene object.

        Returns:
            (obj_dict, None)         if unambiguous match found
            (None, clarify_question) if ambiguous (multiple matches)
            (None, None)             if no match
        """
        objects = self.scene.object_list()
        if not objects:
            return None, "No scene loaded. Please wait for scene initialization."

        candidates = []
        for obj in objects:
            match_color = (color is None or obj["color_name"] == color)
            match_shape = (shape is None or obj["shape_name"] == shape)
            if match_color and match_shape:
                candidates.append(obj)

        if len(candidates) == 1:
            return candidates[0], None
        elif len(candidates) == 0:
            return None, None
        else:
            # Ambiguous — need clarification
            descs = ", ".join(
                f"{o['color_name']} {o['shape_name']} (at {o['dist_from_robot']:.1f}m)"
                for o in candidates
            )
            return None, (
                f"Multiple {color or ''} {shape or ''} objects found: {descs}. "
                f"Which one? (specify color or shape)"
            )

    def _scene_summary(self) -> str:
        """Short summary of scene objects for error messages."""
        return ", ".join(
            f"{o['color_name']} {o['shape_name']}"
            for o in self.scene.object_list()
        )
