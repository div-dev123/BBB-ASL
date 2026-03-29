# Identified Issues and Recommendations

This document logs the architectural problems, performance bottlenecks, and scientific bugs I found while going through the `BBB-ASL` codebase. Issues 6, 7, and 8 are the most important ones from a scientific correctness standpoint and should be addressed first.

---

## 1. Fragile Path Handling for Configuration

The core fitting scripts — `fitting_single_te.py`, `fitting_multi_te.py`, and `asl_single_te.py` — all load `config.json` using a bare relative path (`with open("config.json", "r")`). This only works if you run the script from inside `src/bbb_exchange/`. Run it from anywhere else and you get a `FileNotFoundError` immediately.

This makes it difficult to import these modules as part of a larger pipeline or automated test suite, since the working directory is not always predictable in those contexts.

The straightforward fix is to derive the path from the script's own location using `os.path.dirname(__file__)`, which works regardless of where you run from. I have already applied this fix to `fitting_single_te.py` as part of the validation work.

---

## 2. Dependency Ambiguity (PyStan 3)

The project uses `PyStan 3.x`, imported as `import stan`. The older `PyStan 2.x` uses a completely different import (`import pystan`) and a different API. These two versions are not compatible, and many neuroimaging environments still have `PyStan 2.x` installed from other tools.

If someone installs the wrong version they will either get a `ModuleNotFoundError` or, worse, a confusing runtime failure because the API changed between versions.

The fix here is just documentation — explicitly specifying `pystan>=3.0` in `requirements.txt` and adding a note in the `README` would prevent most confusion.

---

## 3. Performance Bottleneck: Stan Model Compiled Inside the Voxel Loop

This one is the reason the Bayesian pipeline appears to hang. In both `fitting_single_te.py` and `fitting_multi_te.py`, `stan.build()` is called inside the nested loop that iterates over every voxel. `Stan`'s build step compiles the model to `C++` — a process that typically takes 30 to 60 seconds. For a standard brain volume with tens of thousands of voxels, this translates to weeks of processing time.

The fix is simple in principle: call `stan.build()` once before the loop, then call `model.sample()` with different data for each voxel. The compiled model can be reused across all voxels. This would make the Bayesian fitting actually usable in practice.

---

## 4. Stan Compilation Failure on macOS

Even before the loop issue, `Stan` currently fails to compile at all on recent `macOS` versions. The error is a `clang++` exit code 1, triggered specifically by bounded parameters like `real<lower=0>`. This appears to be an incompatibility between `httpstan` and the current `macOS` Command Line Tools headers.

The practical impact is that Bayesian fitting is completely non-functional on `macOS` right now without manual environment modifications. Beyond the immediate fix (reinstalling Command Line Tools or using a Linux environment), this highlights a broader architectural point — if Bayesian fitting is a hard dependency rather than an optional pathway, a single environment issue makes the entire pipeline unusable. Making it opt-in rather than required would make the tool much more robust.

---

## 5. Module-Level Execution Side Effects

`DeltaM_model.py` contains code at the top level of the script — things like `t = np.linspace(...)` — outside of any function or `if __name__ == "__main__"` guard. This means that simply importing the module from another script triggers these computations immediately.

It is a minor issue in isolation but it becomes annoying when you try to use the module as part of a larger system. The standard fix is to wrap any demo or test code inside a `if __name__ == "__main__":` block so it only runs when the script is executed directly.

---

## 6. TI/PLD Time-Axis Mismatch — SCIENTIFIC BUG, HIGH PRIORITY

In `fitting_single_te.py` and `fitting_multi_te.py`, the array of post-labelling delays (`PLD`) is passed directly to `dm_tiss()` and `deltaM_multite_model()` as the time argument `t`. However, both of these functions expect the inversion time (`TI` = `PLD` + `tau`) — the time measured from the start of labelling, not the start of the readout window.

The consequence is that the fitter is searching for signal peaks at the wrong time points. On real scanner data this would produce `ATT` estimates that are systematically shifted, since the model is effectively being evaluated on a time axis that is offset by the labelling duration. This issue affects both the single-TE and multi-TE pipelines in the same way.

The fix is straightforward: compute `tis = plds + tau` and pass `tis` to the model functions instead of `plds`. For the validation work I deliberately matched the fitter's convention (passing `PLD`) so that the recovery results reflect fitting quality rather than this offset — but this is clearly something that needs to be corrected before processing real data.

---

## 7. Inconsistent M0 Normalization Between LS and Bayesian — SCIENTIFIC BUG, HIGH PRIORITY

The `LS` and Bayesian fitters derive `M0a` in completely different ways. The `LS` fitter computes `M0a = (m0 * 5) / (6000 * lambda)`, where the factor of 5 is undocumented and does not appear in the Chappell 2010 paper the model is based on. The Bayesian fitter, on the other hand, simply hardcodes `M0a = 1.0`, ignoring the measured `M0` scan entirely.

This means `CBF` values from the two methods are not numerically comparable — they are solving different problems despite receiving the same input data. A researcher running both methods to cross-validate results would get numbers that cannot be meaningfully compared. This was raised with the project mentors and confirmed as a known issue.

The right fix is to separate calibration from fitting entirely. Both fitters should receive a pre-computed `M0a` derived from a standardized calibration step that runs upstream of both. That way the choice of fitting method does not affect the calibration, and results from `LS` and Bayesian can actually be compared.

---

## 8. Partition Coefficient Missing from Chappell Model — SCIENTIFIC BUG, HIGH PRIORITY

`dm_tiss()` in `DeltaM_model.py` takes the partition coefficient `k` (lambda) as an input and uses it correctly when computing the apparent `T1` relaxation rate. But it never applies it to the signal amplitude itself. The final `CBF` calculation effectively treats lambda as 1.0 regardless of what value is passed in.

The result is that `CBF` is systematically overestimated by roughly 10-11% — a factor of 1/lambda. What makes this particularly tricky is that it is completely silent. The function runs without any error or warning, and the output looks numerically reasonable. You would only catch it by carefully comparing the implementation against the equations in Chappell 2010, which is exactly how I found it.

The fix is to include the partition coefficient in the signal amplitude term as the paper specifies.

---

## 9. Duplicated Pipeline Logic

`create_parameter_config_from_config()` is defined identically in both `fitting_single_te.py` and `asl_single_te.py`. Right now they happen to be identical, but there is nothing stopping them from drifting apart as the code evolves — at which point different parts of the pipeline would quietly start behaving differently with the same configuration input.

The obvious fix is to move shared utility functions like this into a `utils.py` module and import from there.

---

## 10. Hardcoded Data Paths

Both `asl_single_te.py` and `asl_multi_te.py` have the data directory hardcoded directly in the source — `"../data/1TE"` and `"../data/multite"` respectively. If you want to run the pipeline on a different dataset you have to edit the source file, which is not how a reusable tool should work.

Moving these paths into `config.json` or accepting them as command-line arguments would make it trivial to point the pipeline at any dataset without touching the code.

---

## 11. Inconsistent Return Values Between Pipeline Scripts

`asl_single_te.py` has no return statement — it runs, saves files to disk, and exits. `asl_multi_te.py` returns a full results dictionary. This means the two main pipeline entry points cannot be used interchangeably in any programmatic context.

Standardizing both to return a consistent results container — even just a simple dictionary with the same keys — would make it possible to call either pipeline and handle the output the same way. 
