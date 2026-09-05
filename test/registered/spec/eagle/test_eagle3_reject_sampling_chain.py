"""EAGLE3 under the classic (Leviathan) chain rejection sampler.

`--speculative-draft-temperature > 0` with `--speculative-eagle-topk 1` routes
EAGLE3 to `chain_speculative_sampling_triton`, which verifies with
`min(1, p/q)`. Without the temperature the server keeps EAGLE3's target-only
tree kernel, which never reads `q` -- so this file is the only e2e coverage
where the draft temperature is observable at all.

The draft head vocabulary is reduced (`d2t`), so every step also exercises the
scatter of `q` into the target vocabulary. That scatter fails *silently* in the
optimistic direction: a wrong column leaves `q == 0` on the proposed token,
`coin * q < p` becomes vacuously true, and accept length pins to
`speculative_num_steps` while the text degrades. `SGLANG_ENABLE_ASYNC_ASSERT`
is on so the `q > 0` probe in `_build_verify_draft_probs` turns that into a
crash instead of a good-looking number.

gsm8k is scored at the *unmodified* EAGLE3 threshold: the eval is greedy, so
`p` is one-hot and the chain sampler is exactly lossless. Only the accept
length is allowed to drop, because sampling the draft at temperature 1 instead
of taking its argmax proposes worse tokens.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import SpecAccuracyKit, SpecCorrectnessKit
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")


class TestEagle3RejectSamplingChain(Eagle3Base, SpecCorrectnessKit, SpecAccuracyKit):
    extra_args = (
        "--speculative-use-rejection-sampling",
        "--speculative-draft-temperature",
        "1.0",
    )
    env_overrides = ((envs.SGLANG_ENABLE_ASYNC_ASSERT, 1),)

    # Greedy gsm8k is lossless under min(1, p/q), so the score threshold is the
    # inherited one. Accept length is lower than argmax drafting by design.
    acc_length_thres = 1.2
    batch_accept_len_thres = 1.1
    gsm8k_accept_len_thres = 1.1


if __name__ == "__main__":
    unittest.main()
