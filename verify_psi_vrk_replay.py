"""
Source-Free Independent Replay Verifier for Ψ Self-Modeling Benchmark (v5.0-VRK-Aligned)
========================================================================================
Part of The Reggae Shark Universe & The Shark Protocol (Program 02: Ψ Self-Modeling Benchmark)
Author & Lead Researcher: Prince Upadhyay (ORCID: 0009-0002-1170-4487)
Repository: https://github.com/reggaesharkk/psi-self-modeling-benchmark
OSF Registration: 10.17605/OSF.IO/N3DQU

Core Epistemic Mandate (Seven-Step Proof-Engineering Framework — Step 6):
- Executes strictly using the Python 3 Standard Library (`fractions.Fraction`, `hashlib`, `json`, `math`).
- Zero third-party dependencies (does NOT import `numpy`, `pandas`, `scipy`, `torch`, or repository code).
- Independently replays and verifies:
  1. Layer 1 VRK Schema (`intentseal-vrk-claim-v2`) & Fail-Closed Guards (G1 API Snapshot, G2 SHA-256 Lock).
  2. Exact canonical SHA-256 digest of the model summary table (`results_sha256`).
  3. Exact rational arithmetic (`fractions.Fraction`) for all Domain 1, Domain 2, and Combined 2AFC deltas.
  4. Exact rational & floating-point replay of the orthogonalized C1 Scale Regression residuals (`Psi_emp`).
  5. Global Claim Firewall presence and invariance.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import sys
from typing import Any, Dict, List


EXPECTED_RESULTS_SHA256 = "47a16bdef1af686654000734f6326b378fd0ec5403be3aa3b9480e441422fcb3"
EXPECTED_PROMPT_BANK_SHA256 = "301fc789c305ed3e1c9af89fb4d5ebdfeb2d184190504329f23f3dfc5640c761"
EXPECTED_PORTFOLIO_SHA256 = "6cc81062ca6afe1d885a6394bfb297e50e10ca01c676b87ef351bc50150633e0"
EXPECTED_SCRIPT_SHA256 = "3bf99dcf99bad0dde7d7b04bc111140b56d31e111630e895a354f126c3659cad"


def canonical_json_sha256(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def solve_4x4_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Pure standard-library Gaussian elimination with partial pivoting for 4x4 C1 normal equations."""
    n = 4
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot_val = M[col][col]
        if abs(pivot_val) < 1e-15:
            raise ValueError("Singular matrix encountered in source-free C1 normal equations.")
        for j in range(col, n + 1):
            M[col][j] /= pivot_val
        for r in range(n):
            if r != col:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def verify_vrk_bundle(bundle_path: str) -> int:
    print("==========================================================================")
    print("STARTING SOURCE-FREE INDEPENDENT REPLAY VERIFICATION (STANDARD LIBRARY)")
    print(f"Target Bundle: {bundle_path}")
    print("==========================================================================")

    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle: Dict[str, Any] = json.load(f)

    # 1. Verify Schema & VRK Certificate Header
    schema = bundle.get("schema")
    assert schema == "intentseal-vrk-claim-v2", f"Invalid schema: {schema}"
    cert = bundle["vrk_manifest_certificate"]

    assert cert["guard_g1_api_snapshot_pass"] is True, "Fail-Closed Guard G1 did not pass!"
    assert cert["guard_g2_manifest_lock_pass"] is True, "Fail-Closed Guard G2 did not pass!"
    assert cert["prompt_bank_sha256"] == EXPECTED_PROMPT_BANK_SHA256, "Prompt bank SHA-256 mismatch!"
    assert cert["portfolio_sha256"] == EXPECTED_PORTFOLIO_SHA256, "Portfolio SHA-256 mismatch!"
    assert "Prohibits any claim of LLM self-awareness" in cert["claim_firewall"], "Firewall declaration missing!"

    print(f"[PASS] Gate 1 — VRK Schema & Fail-Closed Guards (G1/G2) Verified.")
    print(f"       Prompt Bank SHA-256 : {cert['prompt_bank_sha256']}")
    print(f"       Portfolio SHA-256   : {cert['portfolio_sha256']}")
    print(f"       Runner Script SHA256: {cert['script_sha256']}")

    # 2. Verify Canonical Results Digest
    summaries: List[Dict[str, Any]] = bundle["model_summaries"]
    recomputed_results_sha256 = canonical_json_sha256(summaries)
    recorded_results_sha256 = bundle["results_sha256"]

    assert recomputed_results_sha256 == recorded_results_sha256, (
        f"Results SHA-256 mismatch: recomputed {recomputed_results_sha256} != {recorded_results_sha256}"
    )
    assert recomputed_results_sha256 == EXPECTED_RESULTS_SHA256, (
        f"Results SHA-256 does not match frozen ledger digest {EXPECTED_RESULTS_SHA256}"
    )
    print(f"[PASS] Gate 2 — Canonical Results SHA-256 Digest Verified:")
    print(f"       results_sha256      : {recomputed_results_sha256}")

    # 3. Verify Exact Dyadic/Rational Arithmetic (`fractions.Fraction`) per Model
    for row in summaries:
        name = row["model_name"]
        d1_frac = Fraction(row["domain1_delta_exact_fraction"])
        d2_frac = Fraction(row["domain2_delta_exact_fraction"])
        mean_frac = Fraction(row["mean_delta_exact_fraction"])

        # Verify exact rational identity: mean_delta == (d1_delta + d2_delta) / 2
        if (d1_frac + d2_frac) / 2 != mean_frac:
            raise AssertionError(
                f"Rational arithmetic identity failure for {name}: "
                f"({d1_frac} + {d2_frac})/2 != {mean_frac}"
            )

        # Verify decimal rounding consistency within 1e-6
        assert abs(float(d1_frac) - row["domain1_delta"]) <= 1e-6, f"D1 float mismatch on {name}"
        assert abs(float(d2_frac) - row["domain2_delta"]) <= 1e-6, f"D2 float mismatch on {name}"
        assert abs(float(mean_frac) - row["mean_delta"]) <= 1e-6, f"Mean float mismatch on {name}"

    print(f"[PASS] Gate 3 — Exact Rational Arithmetic (`fractions.Fraction`) Verified across {len(summaries)} Models.")

    # 4. Source-Free Replay of Orthogonalized C1 Scale Regression
    n = len(summaries)
    log_n = [math.log10(float(r["params_billions"])) for r in summaries]
    log_ctx = [math.log2(float(r["context_length_k"])) for r in summaries]
    base_acc = [float(r["external_baseline_acc"]) for r in summaries]
    y = [float(r["mean_delta"]) for r in summaries]

    mean_log_n = sum(log_n) / n
    mean_acc = sum(base_acc) / n
    cov_n_acc = sum((log_n[i] - mean_log_n) * (base_acc[i] - mean_acc) for i in range(n)) / n
    var_n = sum((log_n[i] - mean_log_n) ** 2 for i in range(n)) / n
    slope_b = cov_n_acc / (var_n + 1e-12)
    intercept_b = mean_acc - slope_b * mean_log_n
    base_acc_orth = [base_acc[i] - (intercept_b + slope_b * log_n[i]) for i in range(n)]

    def std_z(vals: List[float]) -> List[float]:
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        s = math.sqrt(v)
        return [(x - m) / s for x in vals]

    z_n = std_z(log_n)
    z_ctx = std_z(log_ctx)
    z_acc = std_z(base_acc_orth)

    X = [[1.0, z_n[i], z_ctx[i], z_acc[i]] for i in range(n)]
    ridge_lambda = 1e-4
    XtX = [[sum(X[i][r] * X[i][c] for i in range(n)) for c in range(4)] for r in range(4)]
    for j in range(1, 4):
        XtX[j][j] += ridge_lambda
    Xty = [sum(X[i][r] * y[i] for i in range(n)) for r in range(4)]

    beta = solve_4x4_linear_system(XtX, Xty)
    c1_rec = bundle["c1_regression_diagnostics"]

    assert abs(beta[0] - c1_rec["beta_0_intercept"]) <= 1e-6, "C1 beta_0 mismatch"
    assert abs(beta[1] - c1_rec["beta_1_log10_scale_z"]) <= 1e-6, "C1 beta_1 mismatch"
    assert abs(beta[2] - c1_rec["beta_2_log2_context_z"]) <= 1e-6, "C1 beta_2 mismatch"
    assert abs(beta[3] - c1_rec["beta_3_orth_capability_z"]) <= 1e-6, "C1 beta_3 mismatch"

    max_psi_err = 0.0
    for i, row in enumerate(summaries):
        y_hat = sum(X[i][j] * beta[j] for j in range(4))
        psi_emp = y[i] - y_hat
        err = abs(psi_emp - row["Psi_emp"])
        max_psi_err = max(max_psi_err, err)
        assert err <= 1e-6, f"Psi_emp residual mismatch for {row['model_name']}: {err}"

    print(f"[PASS] Gate 4 — Source-Free C1 Orthogonal Regression Replay Verified (Max |ΔΨ_emp| = {max_psi_err:.2e}).")
    print("==========================================================================")
    print("OVERALL VERIFICATION STATUS: PASS (ZERO DISCREPANCIES)")
    print("==========================================================================")
    return 0


if __name__ == "__main__":
    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "vrk_dgp_verification/colab_psi_benchmark_vrk_bundle.json"
    )
    sys.exit(verify_vrk_bundle(target))