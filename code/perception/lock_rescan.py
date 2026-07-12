"""
code/perception/lock_rescan.py — bounded reacquisition rescan (RF-1 split of
code/lock_mgmt.py; docs/refactor_plan.md).

`ReacquisitionScan` wraps NX-1's `BidirectionalScanSchedule`
(code/scan_sched.py) for a rescan triggered mid-episode by M4/M5/M7 — see
code/perception/lock_mgmt.py's module docstring for the full rationale.
"""
from __future__ import annotations

from code.scan_sched import (BidirectionalScanSchedule, SCAN_DWELL_STEPS, SCAN_LEG_DEG,
                             SCAN_TIMEOUT as _RESCAN_TIMEOUT_STEPS)


class ReacquisitionScan:
    """
    Thin wrapper around NX-1's `BidirectionalScanSchedule` for a rescan
    triggered mid-episode by M4/M5. Deliberately NOT the same schedule
    instance/mechanism the caller uses for its OWN initial scan:
      - inferencer.py's initial "H3" scan gates its SCAN_TIMEOUT off the
        episode's ABSOLUTE step count (`if step >= SCAN_TIMEOUT`) -- re-arming
        it mid-episode (step already >> SCAN_TIMEOUT=200) would immediately
        "time out" without ever actually rotating.
      - eval_search.py's initial scan uses `BidirectionalScanSchedule`
        already, but its OWN outer timeout check is likewise gated off the
        absolute episode step, with the same bug for a mid-episode re-arm.
    `ReacquisitionScan` tracks its own LOCAL step counter starting from 0 at
    construction time, so it is always safe to instantiate fresh whenever
    M4/M5 drops a lock, regardless of how far into the episode that happens.
    Still fully bounded/never-unbounded-rotation, per the design brief's
    mandatory carve-out: it's the exact same `BidirectionalScanSchedule`
    class and constants (`SCAN_LEG_DEG`, `SCAN_DWELL_STEPS`) NX-1 validated.
    """

    def __init__(self, scan_rate: float = 0.6) -> None:
        """Start a fresh bounded rescan with its own local step counter.

        Args:
            scan_rate: Commanded yaw rate magnitude (rad/s) while rotating.
        """
        self._sched = BidirectionalScanSchedule(
            scan_rate=scan_rate, leg_deg=SCAN_LEG_DEG, dwell_steps=SCAN_DWELL_STEPS)
        self._local_step = 0

    def step(self, current_yaw_rad: float) -> float | None:
        """Advance by one control step.

        Args:
            current_yaw_rad: Robot's current world yaw (radians).

        Returns:
            Commanded wz (rad/s), or None if the bounded rescan has timed
            out (caller should exit rescan mode and fall back to the
            default/cached goal, same as the existing scan-timeout
            fallback).
        """
        if self._local_step >= _RESCAN_TIMEOUT_STEPS:
            return None
        self._local_step += 1
        return self._sched.step(current_yaw_rad)
