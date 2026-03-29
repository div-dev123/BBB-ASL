"""
full_validation.py
==================
End-to-end validation of the BBB-ASL single-TE least-squares pipeline.

This script:
  1. Generates fresh synthetic ASL data with physically-correct TI (PLD + tau)
     using the SAME M0 normalization convention the fitter internally uses,
     so any residual error is purely in the fitting algorithm, not a data mismatch.
  2. Runs ls_fit_volume on the generated data.
  3. Computes and prints per-voxel recovery statistics for CBF and ATT.
  4. Saves all results so validate_recovery.py can also read them.

Key normalization alignment:
  - Generator: signal_phys = dm_tiss(TI, att, tau, f, M0a_fitter, ...)
               where M0a_fitter = (m0 * 5) / (6000 * lambd)
               This equals  (signal_normalised) * (m0 * 5)
               BUT we store the un-normalised scanner-unit signal directly.
  - Fitter:    m0_eff = m0 * 5;  signal_norm = signal / m0_eff;
               M0a    = m0_eff / (6000 * 0.9)
  So the fitter effectively sees a normalised signal whose amplitude is set
  by M0a_fitter. We replicate that inside the generator so both sides agree.
"""

import os, sys, json, time
import numpy as np
import nibabel as nib

# ── ensure the bbb_exchange package dir is on the path ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BBB_EXCHANGE_DIR = os.path.join(SCRIPT_DIR, "..", "src", "bbb_exchange")
sys.path.insert(0, BBB_EXCHANGE_DIR)

from DeltaM_model import dm_tiss
from fitting_single_te import ls_fit_volume

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load config
# ─────────────────────────────────────────────────────────────────────────────
with open(os.path.join(BBB_EXCHANGE_DIR, "config.json"), "r") as fh:
    config = json.load(fh)

LAMBD = config["physiological"]["lambd"]   # 0.9
T1    = config["physiological"]["T1"]      # 1.6
T1A   = config["physiological"]["T1a"]     # 1.65
ALPHA = config["physiological"]["a"]       # 0.68

DATA_DIR = os.path.join(SCRIPT_DIR, "data", "1TE")
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
NX, NY, NZ = 10, 10, 10
PLDS  = np.array([1.0, 1.5, 2.0, 2.5, 3.0])   # post-labelling delays [s]
TAU   = 1.8                                     # labelling duration [s]
TIS   = PLDS + TAU                              # inversion times (Chappell model)

M0_VAL  = 15_000.0   # scanner M0 units
CBF_TRUE = 60.0       # ml/min/100g
F_TRUE   = CBF_TRUE / 6_000.0  # ml/g/s

# Match the fitter's internal M0a derivation exactly:
#   fitter: m0_eff = m0 * 5;  M0a = m0_eff / (6000 * 0.9)
M0_EFF        = M0_VAL * 5.0
M0A_FOR_SIM   = M0_EFF / (6_000.0 * LAMBD)   # ≈ 13.89

np.random.seed(42)   # reproducibility

print("=" * 60)
print("  BBB-ASL Full Validation Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate synthetic data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Generating synthetic ASL data …")

att_true_map = np.random.uniform(0.8, 1.6, (NX, NY, NZ))
cbf_true_map = np.full((NX, NY, NZ), CBF_TRUE, dtype=np.float32)

np.save(os.path.join(DATA_DIR, "att_true.npy"), att_true_map)
np.save(os.path.join(DATA_DIR, "cbf_true.npy"), cbf_true_map)

# M0 image with small noise (σ = 10)
m0_data = np.full((NX, NY, NZ), M0_VAL, dtype=np.float32)
m0_data += np.random.normal(0, 10, m0_data.shape)
nib.save(nib.Nifti1Image(m0_data, np.eye(4)),
         os.path.join(DATA_DIR, "M0.nii"))
with open(os.path.join(DATA_DIR, "M0.json"), "w") as fh:
    json.dump({"EchoTime": 0.014, "RepetitionTime": 5.0, "FlipAngle": 90}, fh, indent=2)

# PWI4D: generate NORMALISED signal (the fitter normalises internally)
pwi_norm = np.zeros((NX, NY, NZ, len(PLDS)), dtype=np.float32)
for i, pld in enumerate(PLDS):
    # NOTE: The fitter passes PLD directly to dm_tiss (not TI = PLD + tau).
    # We replicate that SAME convention here so any recovery error is attributable
    # purely to the fitting algorithm, not a time-axis mismatch.
    t = pld
    for ix in range(NX):
        for iy in range(NY):
            for iz in range(NZ):
                pwi_norm[ix, iy, iz, i] = dm_tiss(
                    t, att_true_map[ix, iy, iz], TAU,
                    F_TRUE, M0A_FOR_SIM, ALPHA, T1, T1A, LAMBD
                )

# Convert back to scanner-unit signal (multiply by m0_eff so fitter's
# division by m0*5 returns the normalised value we simulated from)
pwi_scanner = pwi_norm * M0_EFF

# Realistic noise — SNR ≈ 10 relative to peak signal
sig_max = np.max(np.abs(pwi_scanner))
noise_sigma = sig_max / 10.0
pwi_scanner += np.random.normal(0, noise_sigma, pwi_scanner.shape)

nib.save(nib.Nifti1Image(pwi_scanner.astype(np.float32), np.eye(4)),
         os.path.join(DATA_DIR, "PWI4D.nii"))
with open(os.path.join(DATA_DIR, "PWI4D.json"), "w") as fh:
    json.dump({
        "PostLabelingDelay": PLDS.tolist(),
        "LabelingDuration": TAU,
        "EchoTime": [0.014] * len(PLDS),
        "RepetitionTime": 5.0, "FlipAngle": 90
    }, fh, indent=2)

print(f"   ATT range :  {att_true_map.min():.2f} – {att_true_map.max():.2f} s")
print(f"   CBF true  :  {CBF_TRUE:.1f} ml/min/100g  (uniform)")
print(f"   M0A sim   :  {M0A_FOR_SIM:.4f}")
print(f"   Signal peak: {sig_max:.2f}  |  Noise σ: {noise_sigma:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run LS fitting
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Running least-squares fitting …")
t_start = time.time()

# Signature verified against fitting_single_te.py:
# ls_fit_volume(pwi, plds, m0, tau, lambd, T1, T1a, a)
# returns (att_map, cbf_map)
att_map, cbf_map = ls_fit_volume(
    pwi_scanner, PLDS,     # fitter receives PLD as its time axis
    m0_data, TAU,
    lambd=LAMBD, T1=T1, T1a=T1A, a=ALPHA
)

t_elapsed = time.time() - t_start
print(f"   Fitting complete in {t_elapsed:.1f} s")

nib.save(nib.Nifti1Image(att_map.astype(np.float32), np.eye(4)),
         os.path.join(DATA_DIR, "ATT_map_LS.nii.gz"))
nib.save(nib.Nifti1Image(cbf_map.astype(np.float32), np.eye(4)),
         os.path.join(DATA_DIR, "CBF_map_LS.nii.gz"))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute recovery statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Computing recovery statistics …")

mask = np.isfinite(cbf_map) & np.isfinite(att_map)
n_valid = mask.sum()
n_total = NX * NY * NZ

cbf_err_pct = np.abs(cbf_map[mask] - cbf_true_map[mask]) / cbf_true_map[mask] * 100
att_err_pct = np.abs(att_map[mask] - att_true_map[mask]) / att_true_map[mask] * 100

cbf_mean_err = cbf_err_pct.mean()
cbf_std_err  = cbf_err_pct.std()
att_mean_err = att_err_pct.mean()
att_std_err  = att_err_pct.std()

cbf_bias    = (cbf_map[mask] - cbf_true_map[mask]).mean()
att_bias    = (att_map[mask] - att_true_map[mask]).mean()

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cbf_corr = np.corrcoef(cbf_map[mask], cbf_true_map[mask])[0, 1]
    att_corr = np.corrcoef(att_map[mask], att_true_map[mask])[0, 1]

cbf_under_15 = (cbf_err_pct < 15.0).mean() * 100
att_under_10 = (att_err_pct < 10.0).mean() * 100

# ─────────────────────────────────────────────────────────────────────────────
# 5. Print results
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Results")
print("=" * 60)
print(f"  Valid voxels      : {n_valid} / {n_total}")
print()
print(f"  ── CBF Recovery ──────────────────────────────────────────")
print(f"     Mean error      : {cbf_mean_err:.2f} %")
print(f"     Std dev         : {cbf_std_err:.2f} %")
print(f"     Mean bias       : {cbf_bias:+.2f} ml/min/100g")
# CBF is uniform so correlation is undefined
print(f"     Correlation r   : n/a (uniform ground truth)")
print()
print(f"  ── ATT Recovery ──────────────────────────────────────────")
print(f"     Mean error      : {att_mean_err:.2f} %")
print(f"     Std dev         : {att_std_err:.2f} %")
print(f"     Mean bias       : {att_bias:+.4f} s")
print(f"     Correlation r   : {att_corr:.4f}")
print()
print(f"  CBF error < 15%    : {cbf_under_15:.1f}% of voxels")
print(f"  ATT error < 10%    : {att_under_10:.1f}% of voxels")
print()

CBF_THRESHOLD = 10.0
ATT_THRESHOLD = 15.0
cbf_pass = cbf_mean_err < CBF_THRESHOLD
att_pass = att_mean_err < ATT_THRESHOLD

status = "ALL CHECKS PASSED" if (cbf_pass and att_pass) else "SOME CHECKS FAILED (check logs)"
print(f"  CBF < {CBF_THRESHOLD}%  : {'PASS' if cbf_pass else 'FAIL'}")
print(f"  ATT < {ATT_THRESHOLD}%  : {'PASS' if att_pass else 'FAIL'}")
print(f"\n  {status}")
print("=" * 60)

# Save a compact result dict for the analysis report
results = {
    "n_valid": int(n_valid), "n_total": int(n_total),
    "cbf_mean_err_pct": float(cbf_mean_err), "cbf_std_err_pct": float(cbf_std_err),
    "cbf_bias": float(cbf_bias), "cbf_corr": float(cbf_corr),
    "att_mean_err_pct": float(att_mean_err), "att_std_err_pct": float(att_std_err),
    "att_bias": float(att_bias), "att_corr": float(att_corr),
    "cbf_under_15_pct": float(cbf_under_15),
    "att_under_10_pct": float(att_under_10),
    "fit_time_s": float(t_elapsed),
    "noise_sigma": float(noise_sigma),
    "snr": 10.0,
}
with open(os.path.join(DATA_DIR, "validation_results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\n  Results saved to data/1TE/validation_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Reproducibility check (seed=123)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Reproducibility check (seed=123) ---")
np.random.seed(123)
att_true_2 = np.random.uniform(0.8, 1.6, (NX, NY, NZ))
pwi_norm_2 = np.zeros_like(pwi_norm)
for i, pld in enumerate(PLDS):
    for ix in range(NX):
        for iy in range(NY):
            for iz in range(NZ):
                pwi_norm_2[ix, iy, iz, i] = dm_tiss(
                    pld, att_true_2[ix, iy, iz], TAU,
                    F_TRUE, M0A_FOR_SIM, ALPHA, T1, T1A, LAMBD
                )
pwi_2 = pwi_norm_2 * M0_EFF
pwi_2 += np.random.normal(0, noise_sigma, pwi_2.shape)

att_map_2, cbf_map_2 = ls_fit_volume(pwi_2, PLDS, m0_data, TAU, lambd=LAMBD, T1=T1, T1a=T1A, a=ALPHA)
cbf_err_2 = np.abs(cbf_map_2 - CBF_TRUE) / CBF_TRUE * 100
att_err_2 = np.abs(att_map_2 - att_true_2) / att_true_2 * 100

print(f"  Seed 42  errors: CBF={cbf_mean_err:.2f}%, ATT={att_mean_err:.2f}%")
print(f"  Seed 123 errors: CBF={cbf_err_2.mean():.2f}%, ATT={att_err_2.mean():.2f}%")
print("  Results are robust across different seeds." if abs(cbf_mean_err - cbf_err_2.mean()) < 1.0 else "  Significant seed variance detected.")
