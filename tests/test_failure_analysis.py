import torch

from src.analysis.failure_cases import select_top_failures


def test_select_top_failures_returns_high_confidence_mistakes_first():
    logits = torch.tensor(
        [
            [0.1, 2.0, 0.2],
            [3.0, 0.2, 0.1],
            [0.2, 0.3, 4.0],
        ]
    )
    targets = torch.tensor([1, 2, 0])

    failures = select_top_failures(logits, targets, top_k=2)

    assert len(failures) == 2
    assert failures[0].index == 2
    assert failures[0].predicted == 2
    assert failures[0].target == 0
    assert failures[0].confidence > failures[1].confidence

