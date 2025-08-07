import numpy as np
import numpy.typing as npt
import torch

from src.metrics.base_metric import BaseMetric

# Taken from calculate_eer.py


def compute_det_curve(
    target_scores: npt.NDArray[np.float32], nontarget_scores: npt.NDArray[np.float32]
):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(
    bonafide_scores: npt.NDArray[np.float32], other_scores: npt.NDArray[np.float32]
):
    """
    Returns equal error rate (EER) and the corresponding threshold.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


class EER(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonafide_scores: list[torch.Tensor] = []
        self.spoof_scores: list[torch.Tensor] = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        predictions = torch.nn.Softmax(dim=1)(logits)
        scores: torch.Tensor = predictions[:, 1]
        bonafide_scores_batch: torch.Tensor = scores.masked_select(labels == 1)
        spoof_scores_batch: torch.Tensor = scores.masked_select(labels == 0)
        self.bonafide_scores.append(bonafide_scores_batch)
        self.spoof_scores.append(spoof_scores_batch)
        return 0

    def report_epoch(self) -> float:
        bonafide_total = torch.cat(self.bonafide_scores).cpu().numpy()
        spoof_total = torch.cat(self.spoof_scores).cpu().numpy()
        eer, _ = compute_eer(bonafide_total, spoof_total)
        self.bonafide_scores.clear()
        self.spoof_scores.clear()
        return eer
