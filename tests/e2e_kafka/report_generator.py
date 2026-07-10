"""Kafka E2E integration report generator.

Produces a structured markdown report comparing Mode 1 (Manual Handler)
and Mode 2 (Full Consumer) verification results, including per-step
latency, offset advancement, event IDs, and pass/fail status.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class StepResult:
    """Result of a single verification step."""

    step_number: int
    step_name: str
    passed: bool
    latency_ms: float = 0.0
    event_id: Optional[str] = None
    details: str = ""
    error: Optional[str] = None
    payload_summary: Optional[Dict[str, Any]] = None


@dataclass
class OffsetSnapshot:
    """Offset snapshot for a topic-partition."""

    topic: str
    partition: int
    start_offset: int
    end_offset: int

    @property
    def delta(self) -> int:
        return self.end_offset - self.start_offset


@dataclass
class ModeResult:
    """Aggregated results for a single verification mode."""

    mode_name: str
    steps: List[StepResult] = field(default_factory=list)
    offsets: List[OffsetSnapshot] = field(default_factory=list)
    total_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def passed_count(self) -> int:
        return sum(1 for s in self.steps if s.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if not s.passed)

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.steps)


def generate_report(
    mode1: ModeResult,
    mode2: ModeResult,
    broker: str = "localhost:9092",
    output_path: Optional[str] = None,
) -> str:
    """Generate the full E2E integration report as markdown.

    Args:
        mode1: Results from Mode 1 (Manual Handler).
        mode2: Results from Mode 2 (Full Consumer).
        broker: Kafka broker address.
        output_path: If provided, writes the report to this file.

    Returns:
        The markdown report content.
    """
    now = datetime.now(timezone.utc).isoformat()

    lines = [
        "# Kafka End-to-End Integration Report",
        "",
        "## Test Run Metadata",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| **Generated** | {now} |",
        f"| **Broker** | `{broker}` |",
        f"| **Topics** | `sensor.reading.anomaly`, `compound.risk.detected`, `hazard.propagated` |",
        f"| **Mode 1 Status** | {'✅ ALL PASSED' if mode1.all_passed else '❌ FAILURES DETECTED'} ({mode1.passed_count}/{len(mode1.steps)}) |",
        f"| **Mode 2 Status** | {'✅ ALL PASSED' if mode2.all_passed else '❌ FAILURES DETECTED'} ({mode2.passed_count}/{len(mode2.steps)}) |",
        "",
        "---",
        "",
    ]

    # Mode 1 results
    lines.extend(_render_mode_section(mode1))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Mode 2 results
    lines.extend(_render_mode_section(mode2))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Comparison section
    lines.extend(_render_comparison(mode1, mode2))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Offset verification
    lines.extend(_render_offset_table(mode1, mode2))

    content = "\n".join(lines) + "\n"

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

    return content


def _render_mode_section(mode: ModeResult) -> List[str]:
    """Render a single mode's results section."""
    status = "✅" if mode.all_passed else "❌"
    lines = [
        f"## {status} {mode.mode_name}",
        "",
        f"| Started | Completed | E2E Latency | Steps Passed |",
        f"|---------|-----------|-------------|--------------|",
        f"| {mode.started_at or 'N/A'} | {mode.completed_at or 'N/A'} | {mode.e2e_latency_ms:.1f}ms | {mode.passed_count}/{len(mode.steps)} |",
        "",
        "### Step Results",
        "",
        "| Step | Name | Status | Latency | Event ID | Details |",
        "|------|------|--------|---------|----------|---------|",
    ]

    for step in mode.steps:
        status_icon = "✅" if step.passed else "❌"
        eid = f"`{step.event_id[:12]}…`" if step.event_id and len(step.event_id) > 12 else (f"`{step.event_id}`" if step.event_id else "—")
        detail = step.details[:80] if step.details else "—"
        if step.error:
            detail = f"⚠️ {step.error[:60]}"
        lines.append(
            f"| {step.step_number} | {step.step_name} | {status_icon} | {step.latency_ms:.1f}ms | {eid} | {detail} |"
        )

    return lines


def _render_comparison(mode1: ModeResult, mode2: ModeResult) -> List[str]:
    """Render the comparison section between both modes."""
    lines = [
        "## ⚖️ Mode Comparison",
        "",
        "| Metric | Mode 1 (Manual Handler) | Mode 2 (Full Consumer) |",
        "|--------|------------------------|------------------------|",
        f"| **Steps Passed** | {mode1.passed_count}/{len(mode1.steps)} | {mode2.passed_count}/{len(mode2.steps)} |",
        f"| **Steps Failed** | {mode1.failed_count} | {mode2.failed_count} |",
        f"| **E2E Latency** | {mode1.e2e_latency_ms:.1f}ms | {mode2.e2e_latency_ms:.1f}ms |",
        f"| **Total Step Latency** | {mode1.total_latency_ms:.1f}ms | {mode2.total_latency_ms:.1f}ms |",
        f"| **Result** | {'✅ PASS' if mode1.all_passed else '❌ FAIL'} | {'✅ PASS' if mode2.all_passed else '❌ FAIL'} |",
        "",
    ]

    # Per-step latency comparison
    lines.extend([
        "### Per-Step Latency Comparison",
        "",
        "| Step | Mode 1 (ms) | Mode 2 (ms) | Delta |",
        "|------|-------------|-------------|-------|",
    ])

    max_steps = max(len(mode1.steps), len(mode2.steps))
    for i in range(max_steps):
        m1 = mode1.steps[i] if i < len(mode1.steps) else None
        m2 = mode2.steps[i] if i < len(mode2.steps) else None
        name = (m1 or m2).step_name if (m1 or m2) else "—"
        m1_lat = f"{m1.latency_ms:.1f}" if m1 else "—"
        m2_lat = f"{m2.latency_ms:.1f}" if m2 else "—"
        if m1 and m2:
            delta = m2.latency_ms - m1.latency_ms
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.1f}"
        else:
            delta_str = "—"
        lines.append(f"| {i + 1} | {m1_lat} | {m2_lat} | {delta_str} |")

    return lines


def _render_offset_table(mode1: ModeResult, mode2: ModeResult) -> List[str]:
    """Render offset verification tables for both modes."""
    lines = [
        "## 📊 Kafka Offset Verification",
        "",
    ]

    for mode in [mode1, mode2]:
        if not mode.offsets:
            continue
        lines.extend([
            f"### {mode.mode_name}",
            "",
            "| Topic | Partition | Start Offset | End Offset | Delta |",
            "|-------|-----------|-------------|-----------|-------|",
        ])
        for o in mode.offsets:
            delta_icon = "✅" if o.delta > 0 else "⚠️"
            lines.append(
                f"| `{o.topic}` | {o.partition} | {o.start_offset} | {o.end_offset} | {delta_icon} +{o.delta} |"
            )
        lines.append("")

    return lines
