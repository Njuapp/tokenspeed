from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.moe import expert_file_placement as placement_module
from tokenspeed.runtime.moe.expert_file_placement import (
    LOAD_MODES,
    LOAD_PROFILE_LIVE_ROWS_ENV,
    ExpertLoadProfile,
)

DECODE = LOAD_MODES.index("decode")


@pytest.fixture(autouse=True)
def _no_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _counts(ids: torch.Tensor, experts: int) -> torch.Tensor:
    return torch.bincount(ids.reshape(-1).long(), minlength=experts)


def _profile(tmp_path, monkeypatch: pytest.MonkeyPatch, live_rows: str | None = None):
    if live_rows is None:
        monkeypatch.delenv(LOAD_PROFILE_LIVE_ROWS_ENV, raising=False)
    else:
        monkeypatch.setenv(LOAD_PROFILE_LIVE_ROWS_ENV, live_rows)
    return ExpertLoadProfile(str(tmp_path), interval=10**9)


def test_load_profile_counts_real_rows_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = torch.Generator().manual_seed(1)
    profile = _profile(tmp_path, monkeypatch)
    assert profile.live_rows_only
    target = profile.counter(
        layer_index=0, num_experts=384, top_k=6, ep_rank=3, ep_size=16
    )
    draft = profile.counter(
        layer_index=0, num_experts=128, top_k=3, ep_rank=3, ep_size=16
    )
    cpu = torch.device("cpu")
    target.allocate(cpu, 24 * 6)
    draft.allocate(cpu, 24)
    expect_target = torch.zeros(384, dtype=torch.long)
    expect_draft = torch.zeros(128, dtype=torch.long)

    # Warm-up and capture forwards carry dummy rows: nothing counts.
    assert profile.mark_live_requests(0, 1) == (1, 1)
    target.record(torch.randint(0, 384, (48, 6), generator=generator), DECODE)
    assert profile.mark_live_requests(1, 1) == (0, 1)
    assert int(target.routes.sum()) == 0

    # An eager forward is never padded: every row counts.
    ids = torch.randint(0, 384, (18, 6), generator=generator)
    target.record(ids, DECODE)
    expect_target += _counts(ids, 384)

    # Padded replays: `live` of `replayed` requests are real, the real ones
    # first; 6 target rows and one draft row per request and draft step.
    for live, replayed in ((2, 4), (0, 8), (7, 8), (12, 12), (1, 24), (23, 24)):
        previous = profile.mark_live_requests(live, replayed)
        assert previous == (1, 1)
        ids = torch.randint(0, 384, (replayed * 6, 6), generator=generator)
        target.record(ids, DECODE)
        expect_target += _counts(ids[: live * 6], 384)
        for _ in range(5):
            ids = torch.randint(
                0, 128, (replayed, 3), generator=generator, dtype=torch.int32
            )
            draft.record(ids, DECODE)
            expect_draft += _counts(ids[:live], 128)
        profile.mark_live_requests(*previous)

    assert torch.equal(target.routes[DECODE], expect_target)
    assert torch.equal(draft.routes[DECODE], expect_draft)
    snapshot = profile.snapshot()
    layer = snapshot["groups"]["e384k6"]["layers"][0]
    assert layer["modes"]["decode"]["rows"] == int(expect_target.sum()) // 6
    assert snapshot["coverage"]["live_rows_only"] is True


def test_load_profile_marks_set_before_the_buffers_exist(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = torch.Generator().manual_seed(2)
    profile = _profile(tmp_path, monkeypatch)
    counter = profile.counter(
        layer_index=0, num_experts=384, top_k=6, ep_rank=0, ep_size=8
    )
    profile.mark_live_requests(0, 1)
    counter.allocate(torch.device("cpu"), 12)
    counter.record(torch.randint(0, 384, (12, 6), generator=generator), DECODE)
    assert int(counter.routes.sum()) == 0

    # A forward larger than the reserved row index grows it outside capture.
    profile.mark_live_requests(1, 1)
    counter.record(torch.randint(0, 384, (40000, 6), generator=generator), DECODE)
    assert int(counter.routes.sum()) == 40000 * 6


def test_load_profile_can_count_every_row(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = torch.Generator().manual_seed(3)
    profile = _profile(tmp_path, monkeypatch, live_rows="0")
    assert not profile.live_rows_only
    counter = profile.counter(
        layer_index=0, num_experts=384, top_k=6, ep_rank=0, ep_size=8
    )
    counter.allocate(torch.device("cpu"), 48)
    profile.mark_live_requests(2, 4)
    ids = torch.randint(0, 384, (24, 6), generator=generator)
    counter.record(ids, DECODE)
    assert torch.equal(counter.routes[DECODE], _counts(ids, 384))
    assert placement_module.LOAD_PROFILE_LIVE_ROWS_ENV.endswith("LIVE_ROWS")
