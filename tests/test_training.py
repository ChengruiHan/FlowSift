from types import SimpleNamespace

import pytest

from flowvla_rl.training import episode_balanced_chunk_weights


def test_episode_balanced_chunk_weights_do_not_favor_long_episodes() -> None:
    chunks = [
        SimpleNamespace(episode_id=0),
        SimpleNamespace(episode_id=0),
        SimpleNamespace(episode_id=0),
        SimpleNamespace(episode_id=1),
    ]
    weights = episode_balanced_chunk_weights(chunks)
    assert weights[:3] == pytest.approx([1 / 6, 1 / 6, 1 / 6])
    assert weights[3] == pytest.approx(1 / 2)
    assert sum(weights) == pytest.approx(1.0)
