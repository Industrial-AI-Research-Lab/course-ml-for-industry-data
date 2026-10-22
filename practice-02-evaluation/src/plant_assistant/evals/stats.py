"""Statistics for small evaluation sets. Standard library only, every formula is short and readable.

- wilson:          95% interval for a share (works on 50 cases, unlike the textbook formula)
- paired_counts:   both right / only A / only B / both wrong, for two versions on the same cases
- mcnemar_exact:   is "fixed vs broken" a real difference or chance?
- compare:         the decision "better / worse / no detectable difference"
- sample sizes:    how many cases you need to see a difference (the lecture chart)
- cohen_kappa:     agreement corrected for chance
- ppi_mean:        prediction-powered inference: many judge labels + few human labels
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

Z95 = 1.959964
Z80 = 0.841621


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Paired:
    both: int  # right in A and in B
    only_a: int  # right in A, wrong in B: B broke it
    only_b: int  # wrong in A, right in B: B fixed it
    neither: int

    @property
    def n(self) -> int:
        return self.both + self.only_a + self.only_b + self.neither


def paired_counts(a: Sequence[bool], b: Sequence[bool]) -> Paired:
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same cases in the same order")
    both = sum(x and y for x, y in zip(a, b))
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    return Paired(both, only_a, only_b, len(a) - both - only_a - only_b)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test: binomial test on the cases where the versions disagree."""
    m = only_a + only_b
    if m == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def compare(a: Sequence[bool], b: Sequence[bool], alpha: float = 0.05) -> dict:
    """Is B better than A on the same cases? Returns the counts, the p-value and the decision."""
    pc = paired_counts(a, b)
    p = mcnemar_exact(pc.only_a, pc.only_b)
    if p < alpha:
        decision = "B is better" if pc.only_b > pc.only_a else "B is worse"
    else:
        decision = "no detectable difference"
    return {"n": pc.n, "A right": pc.both + pc.only_a, "B right": pc.both + pc.only_b,
            "B fixed": pc.only_b, "B broke": pc.only_a, "p_value": round(p, 4), "decision": decision}


def n_independent(p1: float, p2: float, z_a: float = Z95, z_b: float = Z80) -> int:
    """Cases per version to detect p1 -> p2 with two separate samples (95% confidence, 80% power)."""
    return math.ceil((z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)


def n_paired(diff: float, discordant: float, z_a: float = Z95, z_b: float = Z80) -> int:
    """Cases to detect a difference `diff` when a share `discordant` of the cases changes (McNemar)."""
    if discordant <= diff:
        raise ValueError("the share of changed cases must be larger than the difference")
    return math.ceil((z_a * math.sqrt(discordant) + z_b * math.sqrt(discordant - diff ** 2)) ** 2 / diff ** 2)


def confusion(truth: Sequence[str], pred: Sequence[str], positive: str = "FAIL") -> dict[str, int]:
    """Confusion counts with FAIL as the class we want to catch."""
    tp = sum(t == positive and p == positive for t, p in zip(truth, pred))
    fn = sum(t == positive and p != positive for t, p in zip(truth, pred))
    fp = sum(t != positive and p == positive for t, p in zip(truth, pred))
    tn = sum(t != positive and p != positive for t, p in zip(truth, pred))
    return {"caught": tp, "missed": fn, "false alarm": fp, "correct pass": tn}


def cohen_kappa(truth: Sequence[str], pred: Sequence[str]) -> float:
    n = len(truth)
    labels = sorted(set(truth) | set(pred))
    po = sum(t == p for t, p in zip(truth, pred)) / n
    pe = sum((truth.count(lab) / n) * (list(pred).count(lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement_report(truth: Sequence[str], pred: Sequence[str]) -> dict:
    c = confusion(truth, pred)
    fails, passes = c["caught"] + c["missed"], c["correct pass"] + c["false alarm"]
    return {
        **c,
        "agreement, %": round(100 * sum(t == p for t, p in zip(truth, pred)) / len(truth), 1),
        "kappa": round(cohen_kappa(list(truth), list(pred)), 2),
        "catches failures, %": round(100 * c["caught"] / fails, 1) if fails else None,
        "passes good answers, %": round(100 * c["correct pass"] / passes, 1) if passes else None,
    }


def _var(xs: Sequence[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def ppi_mean(judge_all: Sequence[float], judge_labeled: Sequence[float], human_labeled: Sequence[float],
             z: float = Z95) -> dict:
    """Prediction-powered estimate of a mean (Angelopoulos et al., 2023), the simple version.

    estimate = mean(judge on all cases) + mean(human − judge on the labelled cases)
    The interval adds the uncertainty of both parts. Use judge_all = cases WITHOUT human labels.
    """
    rectifier = [h - j for h, j in zip(human_labeled, judge_labeled)]
    est = sum(judge_all) / len(judge_all) + sum(rectifier) / len(rectifier)
    se = math.sqrt(_var(judge_all) / len(judge_all) + _var(rectifier) / len(rectifier))
    return {"estimate": est, "low": est - z * se, "high": est + z * se,
            "judge_only": sum(judge_all) / len(judge_all), "correction": sum(rectifier) / len(rectifier)}


def mean_ci(xs: Sequence[float], z: float = Z95) -> tuple[float, float, float]:
    """Mean with a normal 95% interval (for the human-only estimate)."""
    m = sum(xs) / len(xs)
    se = math.sqrt(_var(xs) / len(xs))
    return m, m - z * se, m + z * se
