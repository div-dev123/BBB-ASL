# Identified Issues and Recommendations

This document logs general problems, execution hurdles, and potential improvements identified during the repository analysis and testing.

## 1. Fragile Path Handling for Configuration
**Problem**: Core scripts (`asl_single_te.py`, `fitting_single_te.py`, etc.) use relative paths to load `config.json` (e.g., `with open("config.json", "r")`).
**Impact**: If a user runs the scripts from the project root (e.g., `python3 src/bbb_exchange/asl_single_te.py`), the script fails with `FileNotFoundError: [Errno 2] No such file or directory: 'config.json'`.
**Recommendation**: Use absolute paths or paths relative to the script's directory:
```python
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")
with open(config_path, "r") as file:
    config = json.load(file)
```

## 2. Dependency Ambiguity (PyStan)
**Problem**: The project requires PyStan 3.x, which is imported as `import stan`. However, many users or environments might still be configured for PyStan 2.x (imported as `import pystan`).
**Impact**: Confusion during installation and potential `ModuleNotFoundError` if the wrong version is installed.
**Recommendation**: Explicitly document the PyStan 3 requirement in the `README.md` and ensure the virtual environment setup instructions are followed strictly.

## 3. Lack of Spatial Variation in Initial Synthetic Data
**Problem**: The original `generate_synthetic_asl.py` generated identical signals for all voxels.
**Impact**: This makes it difficult to verify if the fitter is correctly mapping spatial variations in physiological parameters (like ATT).
**Status**: This has been addressed in the latest update to `generate_synthetic_asl.py` by introducing a randomized `att_map`.

## 4. Scaling Discrepancy in m0a (Fixed in Synthetic Data)
**Problem**: The initial synthetic data generation used `m0a = m0_val / 0.9`, whereas the fitting logic expects a factor of `6000` for unit conversion.
**Impact**: This would lead to a factor of 6000 error in recovered CBF values.
**Status**: Fixed in the synthetic data generation script.

## 5. Noise Level Realism
**Problem**: Noise was previously added relative to the `M0` value, which could result in a signal-to-noise ratio (SNR) less than 1.
**Impact**: Fitting becomes unstable and unrealistic.
**Recommendation**: Noise should be added relative to the maximum ASL signal to maintain a realistic SNR (e.g., SNR=10).
**Status**: Addressed in the updated generation script.

## 6. Critical Performance Bug: stan.build in Voxel Loop
**Problem**: In `bayesian_fit_volume` (in both `fitting_single_te.py` and `fitting_multi_te.py`), `stan.build()` is called inside the nested voxel loop.
**Impact**: `stan.build()` triggers a full C++ compilation, which takes roughly 30–60 seconds per voxel. For even a small volume (e.g., 100 voxels), this would take over an hour. This is the primary reason the script appears to "not stop."
**Recommendation**: Move `stan.build()` outside the loop. In PyStan 3, you should build the model once and pass different data to the `sampling` method (or call `model.sample(data=voxel_data)`).

## 7. Stan Compilation Failure on macOS
**Problem**: The Stan model fails to compile with a `clang++` error (`exit code 1`).
**Diagnosis**: Diagnostic tests revealed that even a single parameter with a bound (e.g., `real<lower=0> x;`) triggers the compilation failure on this system. 
**Impact**: Bayesian fitting cannot run at all.
**Possible Causes**: 
- Incompatibility between `httpstan` and the current version of the macOS Command Line Tools (Clang).
- Missing C++17 headers or incorrect compiler flags.
**Recommendation**: 
1. Reinstall macOS Command Line Tools: `xcode-select --install`.
2. Ensure `setuptools` and `httpstan` are up to date in the virtual environment.
3. If issues persist, consider using a Docker container or a Linux environment for fitting, or try disabling bounds as a temporary (but scientifically suboptimal) workaround.

