"""Unit tests for the scheduler watchdog."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_components.invariant_checker import (
    create_scheduler_watchdog,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerWatchdog(CustomTestCase):
    @patch(
        "sglang.srt.managers.scheduler_components.invariant_checker.WatchdogRaw"
    )
    def test_uses_debug_batch_field(self, watchdog_cls):
        scheduler = SimpleNamespace(
            is_initializing=False,
            forward_ct=1,
            cur_batch_for_debug=SimpleNamespace(
                batch_size=lambda: 1, reqs=["request"]
            ),
            invariant_checker=SimpleNamespace(
                _check_all_pools=lambda _: (True, [])
            ),
            pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: None),
        )

        create_scheduler_watchdog(scheduler, watchdog_timeout=10)

        kwargs = watchdog_cls.call_args.kwargs
        self.assertTrue(kwargs["is_active"]())
        self.assertIn("batch.batch_size()=1", kwargs["dump_info"]())


if __name__ == "__main__":
    unittest.main()
