"""Report generation test — runs LAST to produce the comparison report.

Collects results from Mode 1 and Mode 2 and generates the combined
markdown integration report at tests/e2e_kafka/kafka_e2e_report.md.
"""

from __future__ import annotations

import os

import pytest

from tests.e2e_kafka.report_generator import ModeResult, generate_report
from tests.e2e_kafka.test_mode1_manual_handler import get_mode1_result
from tests.e2e_kafka.test_mode2_full_consumer import get_mode2_result


REPORT_PATH = os.path.join(
    os.path.dirname(__file__), "kafka_e2e_report.md",
)


@pytest.mark.usefixtures("kafka_available")
class TestGenerateReport:
    """Generate the combined E2E integration report."""

    def test_generate_comparison_report(self, kafka_bootstrap):
        """Produce the final markdown report comparing both modes."""
        mode1 = get_mode1_result()
        mode2 = get_mode2_result()

        report = generate_report(
            mode1=mode1,
            mode2=mode2,
            broker=kafka_bootstrap,
            output_path=REPORT_PATH,
        )

        assert os.path.exists(REPORT_PATH), "Report file must be created"
        assert len(report) > 100, "Report must have meaningful content"
        assert "Mode 1" in report
        assert "Mode 2" in report
        assert "Comparison" in report

        print(f"\n{'='*60}")
        print(f"Integration report generated: {REPORT_PATH}")
        print(f"{'='*60}")
        print(f"Mode 1: {mode1.passed_count}/{len(mode1.steps)} steps passed")
        print(f"Mode 2: {mode2.passed_count}/{len(mode2.steps)} steps passed")
        print(f"{'='*60}\n")
