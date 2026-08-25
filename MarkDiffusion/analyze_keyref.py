"""Merge and analyse the key-derived-reference counterfactual shards.

Produces the numbers that would replace Table 7: for each baseline scheme, the
TPR of LatentSieve's decoder on that scheme's own codeword under SHIFT
lambda=0.7, under two reference sources and two decoder variants.

Because both references are evaluated on the same images, the comparison
between them is paired, so McNemar is the right test rather than a two-sample
proportion test.

  python qi_analyze_keyref.py
"""
import json
import os
import sys
from math import sqrt

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qi_eval_n500_sdp")

# Native-detector TPR at SHIFT lambda=0.7, unified tau=0.67, from Table 6.
NATIVE = {"PRC": 0.0, "T2SMark": 8.8, "GS": 19.2}


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_bootstrap(rows, field, n_boot=5000, seed=0):
    """Percentile CI resampling whole prompts, not individual images.

    The evaluation is 100 prompts x 5 seeds, so images within a prompt are
    correlated and a plain binomial interval understates the uncertainty.
    Resample prompts with replacement and recompute the rate.
    """
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_idx"], []).append(bool(r[field]))
    keys = list(by_prompt)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        hits = tot = 0
        for j in pick:
            v = by_prompt[keys[j]]
            hits += sum(v)
            tot += len(v)
        vals.append(hits / tot)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def mcnemar(a, b):
    """Paired test on two boolean arrays. Returns (b01, b10, chi2_cc, note)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int((a & ~b).sum())   # a detects, b does not
    n10 = int((~a & b).sum())
    if n01 + n10 == 0:
        return n01, n10, 0.0, "identical"
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, chi2, ""


def load(scheme):
    rows, seen = [], set()
    for half in ("_lo", "_hi", ""):
        p = os.path.join(BASE, f"{scheme.lower()}_keyref_counterfactual{half}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)).get("results", []):
            if r["idx"] not in seen:
                seen.add(r["idx"])
                rows.append(r)
    return sorted(rows, key=lambda r: r["idx"])


def main():
    schemes = sys.argv[1:] or ["GS", "PRC", "T2SMark"]
    print(f"{'scheme':<9} {'n':>4}  {'reference':<16} {'decoder':<11} "
          f"{'TPR':>7}  {'Wilson':<15} {'cluster boot':<15} {'BA':>7}  {'lift':>8}")
    print("-" * 108)
    for scheme in schemes:
        rows = load(scheme)
        if not rows:
            print(f"{scheme:<9} (no data yet)")
            continue
        n = len(rows)
        for tag, label in (("inv", "inverted-clean"), ("key", "key-derived")):
            for dec, dlabel in (("m1", "no erasure"), ("m2", "erase 32")):
                det = [r[f"det_{tag}_{dec}"] for r in rows]
                ba = np.mean([r[f"ba_{tag}_{dec}"] for r in rows])
                k = sum(det)
                lo, hi = wilson(k, n)
                clo, chi = cluster_bootstrap(rows, f"det_{tag}_{dec}")
                lift = 100 * k / n - NATIVE.get(scheme, float("nan"))
                print(f"{scheme:<9} {n:>4}  {label:<16} {dlabel:<11} "
                      f"{100*k/n:6.1f}%  [{100*lo:4.1f}, {100*hi:5.1f}]   "
                      f"[{100*clo:4.1f}, {100*chi:5.1f}]   "
                      f"{ba:7.4f}  {lift:+7.1f}pp")

        # Paired comparisons on the default decoder (M1+M2).
        inv = [r["det_inv_m2"] for r in rows]
        key = [r["det_key_m2"] for r in rows]
        n01, n10, chi2, note = mcnemar(inv, key)
        print(f"{'':<9} {'':<4}  McNemar, inverted-clean vs key-derived (erase 32): "
              f"inv-only={n01} key-only={n10} chi2_cc={chi2:.1f} {note}")

        # How much of the lift is erasure rather than plain block aggregation?
        for tag, label in (("inv", "inverted-clean"), ("key", "key-derived")):
            m1 = 100 * np.mean([r[f"det_{tag}_m1"] for r in rows])
            m2 = 100 * np.mean([r[f"det_{tag}_m2"] for r in rows])
            print(f"{'':<9} {'':<4}  {label:<16} erasure contributes "
                  f"{m2-m1:+.1f} pp on top of block aggregation "
                  f"({m1:.1f}% -> {m2:.1f}%)")
        print()


if __name__ == "__main__":
    main()
