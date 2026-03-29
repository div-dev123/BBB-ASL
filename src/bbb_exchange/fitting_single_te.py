import numpy as np
import os
from data_handling import load_nifti_file
import stan
from scipy.optimize import curve_fit
from DeltaM_model import DeltaM_model_ext, dm_tiss

import json

"""
This file provides the necessary functions to
1. perform least squares fitting
2. perform Bayesian fitting

for both the Delta_M model (equation [1] from Chappell paper https://doi.org/10.1002/mrm.22320, and the extended Delta_M_model (equation [1] and [2] from that paper))

In both algorithms, it is possible to flexibly read in the parameters as nparray, as nifti image, or as scalar.
There is the option of using the values of the LS fitting voxel by voxel as prior for the Bayesian fitting.
"""

_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_script_dir, "config.json"), "r") as file:
	config = json.load(file)

def create_parameter_config_from_config():
    return {
        'T1a': config['physiological']['T1a'],
        'T1': config['physiological']['T1'],
        'lambd': config['physiological']['lambd'],
        'a': config['physiological']['a'],
        'abv': config['physiological']['abv'],
        'att_a': config['physiological']['att_a'],

        'fit_T1a': config['fitting']['fit_T1a'],
        'fit_T1': config['fitting']['fit_T1'],
        'fit_lambd': config['fitting']['fit_lambd'],
        'fit_abv': config['fitting']['fit_abv'],
        'fit_att_a': config['fitting']['fit_att_a'],

        'T1a_prior': config['priors']['T1a'],
        'T1_prior': config['priors']['T1'],
        'lambd_prior': config['priors']['lambd'],
        'abv_prior': config['priors']['abv'],
        'att_a_prior': config['priors']['att_a'],

        'att_prior_from_ls': config['ls_priors']['att_prior_from_ls'],
        'cbf_prior_from_ls': config['ls_priors']['cbf_prior_from_ls'],
        'att_prior_std': config['ls_priors']['att_prior_std'],
        'cbf_prior_std': config['ls_priors']['cbf_prior_std'],

        'T1a_bounds': config['bounds']['T1a'],
        'T1_bounds': config['bounds']['T1'],
        'lambd_bounds': config['bounds']['lambd'],
        'abv_bounds': config['bounds']['abv'],
        'att_a_bounds': config['bounds']['att_a']
    }

# === Fitting functions for Least Squares fitting ===

def ls_fit_voxel(t, signal, M0a, tau, T1, T1a, lambd, a):
	"""
	Least squares fitting for a single voxel using DeltaM_model
	"""

	def model_func(t, att, f):
		return dm_tiss(t, att, tau, f, M0a, a, T1, T1a, lambd)

	param0 = [1.2, 0.01]  # Initial guess: att=1.2s, f=0.01 (corresponding to ~60 ml/min/100g)
	bounds = ([0.1, 0.001], [2.5, 0.2])  # Bounds for att and f
	try:
		param_opt, _ = curve_fit(model_func, t, signal, p0=param0, bounds=bounds)
		# Convert f to CBF in ml/min/100g: CBF = f * 6000
		cbf = param_opt[1] * 6000
		return param_opt[0], cbf
	except RuntimeError:
		return np.nan, np.nan


def ls_fit_volume(pwi_data, t, m0_data, tau, lambd=0.9, T1=1.6, T1a=1.65, a=0.85 * 0.8):
	"""
	Least squares fitting for entire volume
	"""
	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				signal = pwi_data[x, y, z, :]
				m0 = m0_data[x, y, z]

				# Check for invalid signal data
				if (np.any(np.isnan(signal)) or
						np.any(np.isinf(signal)) or
						np.all(signal == 0)):
					continue

				# Check for invalid M0 data
				if (np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				m0 = m0 * 5
				signal_normalized = signal / m0
				M0a = m0 / (6000 * 0.9)  # Normalised M0a

				try:
					att, cbf = ls_fit_voxel(t, signal_normalized, M0a, tau, T1, T1a, lambd, a)
					att_map[x, y, z] = att
					cbf_map[x, y, z] = cbf
				except Exception as e:
					continue

	return att_map, cbf_map


def ls_fit_voxel_ext(t, signal, M0a, tau, T1=1.6, T1a=1.65, lambd=0.9, a=0.85 * 0.8):
	"""
	Least squares fitting for DeltaM_model_ext (tissue and arterial compartment)
	"""
	def model_func_ext(t, att, f, abv, att_a):

		params = {
			'Dt': att,
			'tau': tau,
			'f': f,
			'M0a': M0a,
			'a': a,
			'T1': T1,
			'T1a': T1a,
			'k': lambd,
			'aBV': abv,
			'Dta': att_a,
			'ta': tau
		}
		return DeltaM_model_ext(t, params)


	param0 = [1.2, 0.01, 0.02, 0.7]
	bounds = ([0.1, 0.001, 0.001, 0.1],  # lower bounds
			  [2.5, 0.2, 0.1, 2.0])  # upper bounds

	try:
		param_opt, _ = curve_fit(model_func_ext, t, signal, p0=param0, bounds=bounds)
		att, f, abv, att_a = param_opt
		cbf = f * 6000  # Convert to ml/min/100g
		return att, cbf, abv, att_a
	except RuntimeError:
		return np.nan, np.nan, np.nan, np.nan


def ls_fit_volume_ext(pwi_data, t, m0_data, tau, lambd=0.9, T1=1.6, T1a=1.65, a=0.85 * 0.8):
	"""
	Least squares fitting for entire volume using extended DeltaM model
	"""

	shape = pwi_data.shape[:3]
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	abv_map = np.full(shape, np.nan)
	att_a_map = np.full(shape, np.nan)

	total_voxels = shape[0] * shape[1] * shape[2]
	processed_voxels = 0
	successful_fits = 0

	print(f"Starting LS fitting for {total_voxels} voxels using extended model...")

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				processed_voxels += 1

				# Progress update
				if processed_voxels % 1000 == 0:
					print(f"Processing voxel {processed_voxels}/{total_voxels} "
						  f"({100 * processed_voxels / total_voxels:.1f}%) - "
						  f"Successful fits: {successful_fits}")

				signal = pwi_data[x, y, z, :]
				m0 = m0_data[x, y, z]

				# Check for invalid signal data
				if (np.any(np.isnan(signal)) or
						np.any(np.isinf(signal)) or
						np.all(signal == 0)):
					continue

				# Check for invalid M0 data
				if (np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				# Normalize signal
				m0 = m0 * 5
				signal_normalized = signal / m0
				M0a = m0 / (6000 * 0.9)  # Normalized M0a

				try:
					att, cbf, abv, att_a = ls_fit_voxel_ext(
						t, signal_normalized, M0a, tau, T1, T1a, lambd, a
					)

					# Store results if fitting was successful
					if not (np.isnan(att) or np.isnan(cbf) or np.isnan(abv) or np.isnan(att_a)):
						att_map[x, y, z] = att
						cbf_map[x, y, z] = cbf
						abv_map[x, y, z] = abv
						att_a_map[x, y, z] = att_a
						successful_fits += 1

				except Exception as e:
					continue

	print(f"LS fitting completed: {successful_fits}/{processed_voxels} successful fits "
		  f"({100 * successful_fits / processed_voxels:.1f}%)")

	return att_map, cbf_map, abv_map, att_a_map


# === Fitting functions for Bayesian fitting

STAN_MODEL_CODE = """
functions {
  vector deltaM_model(vector t, real att, real f, real M0a, real tau, 
                     real T1a, real T1, real lambd, real a) {
    int n = num_elements(t);
    vector[n] deltaM = rep_vector(0.0, n);

    real T1app = 1.0 / (1.0 / T1 + f / lambd);
    real R = 1.0 / T1app - 1.0 / T1a;

    for (i in 1:n) {
      if (t[i] >= att && t[i] <= att + tau) {
        // Case 2: att <= t <= att + tau
        real time = t[i];
        real term = exp(R * time) - exp(R * att);
        deltaM[i] = (2 * a * M0a * f * exp(-time / T1app) / R) * term;
      } else if (t[i] > att + tau) {
        // Case 3: t > att + tau
        real time = t[i];
        real term = exp(R * (att + tau)) - exp(R * att);
        deltaM[i] = (2 * a * M0a * f * exp(-time / T1app) / R) * term;
      }
      // else: t < att, deltaM[i] remains 0
    }
    return deltaM;
  }
}

data {
  int<lower=0> n;
  vector[n] t;
  vector[n] signal;
  real<lower=0> M0a;
  real<lower=0> tau;

  // Fixed parameter values
  real<lower=0> T1a_fixed;
  real<lower=0> T1_fixed;
  real<lower=0> lambd_fixed;
  real<lower=0> a_fixed;

  // Fitting flags
  int<lower=0,upper=1> fit_T1a;
  int<lower=0,upper=1> fit_T1;
  int<lower=0,upper=1> fit_lambd;

  // Priors
  real T1a_prior_mean;
  real T1a_prior_std;
  real T1_prior_mean;
  real T1_prior_std;
  real lambd_prior_mean;
  real lambd_prior_std;

  // ATT and CBF priors from LS fitting
  int<lower=0,upper=1> use_att_prior_from_ls;
  int<lower=0,upper=1> use_cbf_prior_from_ls;
  real att_prior_from_ls;
  real att_prior_std;
  real cbf_prior_from_ls;
  real cbf_prior_std;

  // Bounds
  real T1a_lower;
  real T1a_upper;
  real T1_lower;
  real T1_upper;
  real lambd_lower;
  real lambd_upper;
}

parameters {
  real<lower=0.1, upper=3.0> att;
  real<lower=0.001, upper=0.2> f;  // CBF as f parameter
  real<lower=0.001> sigma;

  // Conditional parameters
  real<lower=T1a_lower, upper=T1a_upper> T1a_param;
  real<lower=T1_lower, upper=T1_upper> T1_param;
  real<lower=lambd_lower, upper=lambd_upper> lambd_param;
}

transformed parameters {
  real T1a_use;
  real T1_use;
  real lambd_use;
  real cbf;  // CBF in ml/min/100g

  // Use fitted or fixed values
  T1a_use = fit_T1a ? T1a_param : T1a_fixed;
  T1_use = fit_T1 ? T1_param : T1_fixed;
  lambd_use = fit_lambd ? lambd_param : lambd_fixed;

  // Convert f to CBF
  cbf = f * 6000.0;
}

model {
  vector[n] mu;

  if (use_att_prior_from_ls == 1) {
    att ~ normal(att_prior_from_ls, att_prior_std);
  } else {
    att ~ normal(1.2, 0.5);
  }

  if (use_cbf_prior_from_ls == 1) {
    f ~ normal(cbf_prior_from_ls / 6000.0, cbf_prior_std / 6000.0);
  } else {
    f ~ normal(0.01, 0.0025);  // Default: 60 ml/min/100g ± 15
  }

  sigma ~ exponential(1);

  // Conditional priors for flexible parameters
  if (fit_T1a) {
    T1a_param ~ normal(T1a_prior_mean, T1a_prior_std);
  }
  if (fit_T1) {
    T1_param ~ normal(T1_prior_mean, T1_prior_std);
  }
  if (fit_lambd) {
    lambd_param ~ normal(lambd_prior_mean, lambd_prior_std);
  }

  // Likelihood
  mu = deltaM_model(t, att, f, M0a, tau, T1a_use, T1_use, lambd_use, a_fixed);
  signal ~ normal(mu, sigma);
}
"""


def bayesian_fit_voxel(t, signal, M0a, tau,
					   T1a_val, T1_val, lambd_val, a_val,
					   param_config, att_ls_val=None, cbf_ls_val=None):
	"""
	Bayesian fitting for a single voxel with flexible parameters using deltaM_model

	Parameters:
	- t: time points
	- signal: signal values
	- M0a: arterial M0 value
	- tau: labeling duration
	- T1a_val: T1a value for this voxel
	- T1_val: T1 value for this voxel
	- lambd_val: lambda value for this voxel
	- a_val: alpha (labeling efficiency) value
	- param_config: parameter configuration
	- att_ls_val: ATT value from LS fitting (for prior)
	- cbf_ls_val: CBF value from LS fitting (for prior)
	"""

	# Prepare data for using Stan
	data = {
		'n': len(t),
		't': t.astype(float),
		'signal': signal.astype(float),
		'M0a': float(M0a),
		'tau': float(tau),

		# Fixed parameter values
		'T1a_fixed': float(T1a_val),
		'T1_fixed': float(T1_val),
		'lambd_fixed': float(lambd_val),
		'a_fixed': float(a_val),

		# Fitting config
		'fit_T1a': int(param_config['fit_T1a']),
		'fit_T1': int(param_config['fit_T1']),
		'fit_lambd': int(param_config['fit_lambd']),

		# Priors
		'T1a_prior_mean': float(param_config['T1a_prior']['mean']),
		'T1a_prior_std': float(param_config['T1a_prior']['std']),
		'T1_prior_mean': float(param_config['T1_prior']['mean']),
		'T1_prior_std': float(param_config['T1_prior']['std']),
		'lambd_prior_mean': float(param_config['lambd_prior']['mean']),
		'lambd_prior_std': float(param_config['lambd_prior']['std']),

		# LS-based priors
		'use_att_prior_from_ls': int(
			param_config.get('att_prior_from_ls', False) and att_ls_val is not None and np.isfinite(att_ls_val)),
		'use_cbf_prior_from_ls': int(
			param_config.get('cbf_prior_from_ls', False) and cbf_ls_val is not None and np.isfinite(cbf_ls_val)),
		'att_prior_from_ls': float(att_ls_val) if att_ls_val is not None and np.isfinite(att_ls_val) else 1.2,
		'att_prior_std': float(param_config.get('att_prior_std', 0.3)),
		'cbf_prior_from_ls': float(cbf_ls_val) if cbf_ls_val is not None and np.isfinite(cbf_ls_val) else 60.0,
		'cbf_prior_std': float(param_config.get('cbf_prior_std', 15.0)),

		# Bounds
		'T1a_lower': float(param_config['T1a_bounds'][0]),
		'T1a_upper': float(param_config['T1a_bounds'][1]),
		'T1_lower': float(param_config['T1_bounds'][0]),
		'T1_upper': float(param_config['T1_bounds'][1]),
		'lambd_lower': float(param_config['lambd_bounds'][0]),
		'lambd_upper': float(param_config['lambd_bounds'][1])
	}

	try:
		# Build the model
		model = stan.build(STAN_MODEL_CODE, data=data)

		# Sample from the posterior
		fit = model.sample(num_chains=2, num_samples=1000, num_warmup=500)

		return fit

	except Exception as e:
		print(f"Error in Stan fitting: {e}")
		return None


def bayesian_fit_volume(pwi_data, t, m0_data, tau, param_config, att_ls_map=None, cbf_ls_map=None):
	"""
	Bayesian fitting for entire volume with flexible parameters

	Parameters:
	- pwi_data: 4D PWI data
	- t: time points
	- m0_data: M0 data
	- tau: labeling duration
	- param_config: parameter configuration dict
	- att_ls_map: ATT map from LS fitting (for priors)
	- cbf_ls_map: CBF map from LS fitting (for priors)
	"""

	shape = pwi_data.shape[:3]

	# Prepare parameter maps
	print("Preparing parameter maps...")
	T1a_map = convert_parameter(param_config['T1a'], shape, "T1a")
	T1_map = convert_parameter(param_config['T1'], shape, "T1")
	lambd_map = convert_parameter(param_config['lambd'], shape, "lambd")
	a_map = convert_parameter(param_config['a'], shape, "a")

	# Initialize result arrays
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	# Additional maps for fitted parameters
	if param_config['fit_T1a']:
		T1a_fitted_map = np.full(shape, np.nan)
		T1a_fitted_std_map = np.full(shape, np.nan)

	if param_config['fit_T1']:
		T1_fitted_map = np.full(shape, np.nan)
		T1_fitted_std_map = np.full(shape, np.nan)

	if param_config['fit_lambd']:
		lambd_fitted_map = np.full(shape, np.nan)
		lambd_fitted_std_map = np.full(shape, np.nan)

	total_voxels = shape[0] * shape[1] * shape[2]
	processed_voxels = 0
	successful_fits = 0

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				processed_voxels += 1

				if processed_voxels % 100 == 0:
					print(f"Processing voxel {processed_voxels}/{total_voxels} "
						  f"({100 * processed_voxels / total_voxels:.1f}%)")

				# Get data for this voxel
				signal = pwi_data[x, y, z, :]
				m0 = m0_data[x, y, z]

				# Skip invalid voxels
				if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
						np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				# Get parameter values for this voxel
				T1a_val = T1a_map[x, y, z]
				T1_val = T1_map[x, y, z]
				lambd_val = lambd_map[x, y, z]
				a_val = a_map[x, y, z]

				# Skip if any parameter is invalid
				if (np.isnan(T1a_val) or np.isnan(T1_val) or
						np.isnan(lambd_val) or np.isnan(a_val)):
					continue

				# Get LS prior values
				att_ls_val = att_ls_map[x, y, z] if att_ls_map is not None else None
				cbf_ls_val = cbf_ls_map[x, y, z] if cbf_ls_map is not None else None

				signal_normalized = signal / m0
				M0a = 1.0

				try:
					# Fit this voxel
					fit = bayesian_fit_voxel(
						t, signal_normalized, M0a, tau,
						T1a_val, T1_val, lambd_val, a_val,
						param_config, att_ls_val, cbf_ls_val
					)

					if fit is not None:
						# Extract results
						att_samples = fit['att'].flatten()
						cbf_samples = fit['cbf'].flatten()  # Already converted to ml/min/100g in Stan code

						att_map[x, y, z] = np.mean(att_samples)
						cbf_map[x, y, z] = np.mean(cbf_samples)
						att_std_map[x, y, z] = np.std(att_samples)
						cbf_std_map[x, y, z] = np.std(cbf_samples)

						# Extract fitted parameters
						if param_config['fit_T1a']:
							T1a_samples = fit['T1a_param'].flatten()
							T1a_fitted_map[x, y, z] = np.mean(T1a_samples)
							T1a_fitted_std_map[x, y, z] = np.std(T1a_samples)

						if param_config['fit_T1']:
							T1_samples = fit['T1_param'].flatten()
							T1_fitted_map[x, y, z] = np.mean(T1_samples)
							T1_fitted_std_map[x, y, z] = np.std(T1_samples)

						if param_config['fit_lambd']:
							lambd_samples = fit['lambd_param'].flatten()
							lambd_fitted_map[x, y, z] = np.mean(lambd_samples)
							lambd_fitted_std_map[x, y, z] = np.std(lambd_samples)

						successful_fits += 1

				except Exception as e:
					print(f"Error fitting voxel ({x}, {y}, {z}): {e}")
					continue

	print(f"Successfully fitted {successful_fits}/{processed_voxels} voxels")

	# Results
	results = {
		'att_map': att_map,
		'cbf_map': cbf_map,
		'att_std_map': att_std_map,
		'cbf_std_map': cbf_std_map,
		'successful_fits': successful_fits,
		'total_processed': processed_voxels
	}

	# Add fitted parameter maps
	if param_config['fit_T1a']:
		results['T1a_fitted_map'] = T1a_fitted_map
		results['T1a_fitted_std_map'] = T1a_fitted_std_map

	if param_config['fit_T1']:
		results['T1_fitted_map'] = T1_fitted_map
		results['T1_fitted_std_map'] = T1_fitted_std_map

	if param_config['fit_lambd']:
		results['lambd_fitted_map'] = lambd_fitted_map
		results['lambd_fitted_std_map'] = lambd_fitted_std_map

	return results





STAN_MODEL_CODE_EXT = """
functions {
  vector deltaM_model_ext(vector t, real att, real f, real M0a, real tau, 
                          real abv, real att_a, real T1a, real T1, real lambd, real a) {
    int n = num_elements(t);
    vector[n] deltaM = rep_vector(0.0, n);

    // Tissue compartment (Eq. 1)
    real T1app = 1.0 / (1.0 / T1 + f / lambd);
    real R = 1.0 / T1app - 1.0 / T1a;

    for (i in 1:n) {
      // Tissue
      if (t[i] >= att && t[i] <= att + tau) {
        real term = exp(R * t[i]) - exp(R * att);
        deltaM[i] += (2 * a * M0a * f * exp(-t[i] / T1app) / R) * term;
      } else if (t[i] > att + tau) {
        real term = exp(R * (att + tau)) - exp(R * att);
        deltaM[i] += (2 * a * M0a * f * exp(-t[i] / T1app) / R) * term;
      }

      // Arterial compartment (Eq. 2)
      if (t[i] >= att_a && t[i] <= att_a + tau) {
        deltaM[i] += 2 * a * M0a * abv * exp(-t[i] / T1a);
      }
    }
    return deltaM;
  }
}

data {
  int<lower=0> n;
  vector[n] t;
  vector[n] signal;
  real<lower=0> M0a;
  real<lower=0> tau;

  // Parameter values 
  real<lower=0> T1a_fixed;
  real<lower=0> T1_fixed;
  real<lower=0> lambd_fixed;
  real<lower=0> a_fixed;
  real<lower=0> abv_fixed;
  real<lower=0> att_a_fixed;

  // Fitting flags
  int<lower=0,upper=1> fit_T1a;
  int<lower=0,upper=1> fit_T1;
  int<lower=0,upper=1> fit_lambd;
  int<lower=0,upper=1> fit_abv;
  int<lower=0,upper=1> fit_att_a;

  // Priors 
  real T1a_prior_mean;
  real T1a_prior_std;
  real T1_prior_mean;
  real T1_prior_std;
  real lambd_prior_mean;
  real lambd_prior_std;
  real abv_prior_mean;
  real abv_prior_std;
  real att_a_prior_mean;
  real att_a_prior_std;

  // ATT and CBF priors from LS fitting
  int<lower=0,upper=1> use_att_prior_from_ls;
  int<lower=0,upper=1> use_cbf_prior_from_ls;
  real att_prior_from_ls;
  real att_prior_std;
  real cbf_prior_from_ls;
  real cbf_prior_std;

  // Bounds
  real T1a_lower;
  real T1a_upper;
  real T1_lower;
  real T1_upper;
  real lambd_lower;
  real lambd_upper;
  real abv_lower;
  real abv_upper;
  real att_a_lower;
  real att_a_upper;
}

parameters {
  real<lower=0.1, upper=3.0> att;
  real<lower=0.001, upper=0.2> f;  // CBF as f parameter
  real<lower=0.001> sigma;

  // Conditional parameters
  real<lower=T1a_lower, upper=T1a_upper> T1a_param;
  real<lower=T1_lower, upper=T1_upper> T1_param;
  real<lower=lambd_lower, upper=lambd_upper> lambd_param;
  real<lower=abv_lower, upper=abv_upper> abv_param;
  real<lower=att_a_lower, upper=att_a_upper> att_a_param;
}

transformed parameters {
  real T1a_use;
  real T1_use;
  real lambd_use;
  real abv_use;
  real att_a_use;
  real cbf;  // CBF in ml/min/100g

  // Use fitted or constant fixed values
  T1a_use = fit_T1a ? T1a_param : T1a_fixed;
  T1_use = fit_T1 ? T1_param : T1_fixed;
  lambd_use = fit_lambd ? lambd_param : lambd_fixed;
  abv_use = fit_abv ? abv_param : abv_fixed;
  att_a_use = fit_att_a ? att_a_param : att_a_fixed;

  // Convert f to CBF
  cbf = f * 6000.0;
}

model {
  vector[n] mu;

  // Priors for ATT and CBF
  if (use_att_prior_from_ls == 1) {
    att ~ normal(att_prior_from_ls, att_prior_std);
  } else {
    att ~ normal(1.2, 0.5);
  }

  if (use_cbf_prior_from_ls == 1) {
    f ~ normal(cbf_prior_from_ls / 6000.0, cbf_prior_std / 6000.0);
  } else {
    f ~ normal(0.01, 0.0025);  // Default: 60 ml/min/100g
  }

  sigma ~ exponential(1);

  // Conditional priors for flexible parameters
  if (fit_T1a) {
    T1a_param ~ normal(T1a_prior_mean, T1a_prior_std);
  }
  if (fit_T1) {
    T1_param ~ normal(T1_prior_mean, T1_prior_std);
  }
  if (fit_lambd) {
    lambd_param ~ normal(lambd_prior_mean, lambd_prior_std);
  }
  if (fit_abv) {
    abv_param ~ normal(abv_prior_mean, abv_prior_std);
  }
  if (fit_att_a) {
    att_a_param ~ normal(att_a_prior_mean, att_a_prior_std);
  }

  // Likelihood
  mu = deltaM_model_ext(t, att, f, M0a, tau, abv_use, att_a_use, 
                        T1a_use, T1_use, lambd_use, a_fixed);
  signal ~ normal(mu, sigma);
}
"""

def bayesian_fit_voxel_ext(t, signal, M0a, tau,
						   T1a_val, T1_val, lambd_val, a_val,
						   param_config,
						   att_ls_val=None, cbf_ls_val=None,
						   abv_ls_val=None, att_a_ls_val=None):
    """
    Bayesian fitting for DeltaM_model_ext
    """
    data = {
        'n': len(t),
        't': t.astype(float),
        'signal': signal.astype(float),
        'M0a': float(M0a),
        'tau': float(tau),

        # Fixed parameter values
        'T1a_fixed': float(T1a_val),
        'T1_fixed': float(T1_val),
        'lambd_fixed': float(lambd_val),
        'a_fixed': float(a_val),
        'abv_fixed': float(param_config.get('abv', 0.02)),
        'att_a_fixed': float(param_config.get('att_a', 0.7)),

        'fit_T1a': int(param_config.get('fit_T1a', False)),
        'fit_T1': int(param_config.get('fit_T1', False)),
        'fit_lambd': int(param_config.get('fit_lambd', False)),
        'fit_abv': int(param_config.get('fit_abv', True)),
        'fit_att_a': int(param_config.get('fit_att_a', True)),

        # Standard priors
        'T1a_prior_mean': float(param_config.get('T1a_prior', {}).get('mean', 1.65)),
        'T1a_prior_std': float(param_config.get('T1a_prior', {}).get('std', 0.2)),
        'T1_prior_mean': float(param_config.get('T1_prior', {}).get('mean', 1.6)),
        'T1_prior_std': float(param_config.get('T1_prior', {}).get('std', 0.3)),
        'lambd_prior_mean': float(param_config.get('lambd_prior', {}).get('mean', 0.9)),
        'lambd_prior_std': float(param_config.get('lambd_prior', {}).get('std', 0.1)),


        'abv_prior_mean': float(abv_ls_val) if abv_ls_val is not None and np.isfinite(abv_ls_val) else float(param_config.get('abv_prior', {}).get('mean', 0.02)),
        'abv_prior_std': float(param_config.get('abv_prior_std', 0.01)),
        'att_a_prior_mean': float(att_a_ls_val) if att_a_ls_val is not None and np.isfinite(att_a_ls_val) else float(param_config.get('att_a_prior', {}).get('mean', 0.7)),
        'att_a_prior_std': float(param_config.get('att_a_prior_std', 0.2)),


        'use_att_prior_from_ls': int(param_config.get('att_prior_from_ls', False) and att_ls_val is not None and np.isfinite(att_ls_val)),
        'use_cbf_prior_from_ls': int(param_config.get('cbf_prior_from_ls', False) and cbf_ls_val is not None and np.isfinite(cbf_ls_val)),
        'att_prior_from_ls': float(att_ls_val) if att_ls_val is not None and np.isfinite(att_ls_val) else 1.2,
        'att_prior_std': float(param_config.get('att_prior_std', 0.3)),
        'cbf_prior_from_ls': float(cbf_ls_val) if cbf_ls_val is not None and np.isfinite(cbf_ls_val) else 60.0,
        'cbf_prior_std': float(param_config.get('cbf_prior_std', 15.0)),


        'T1a_lower': float(param_config.get('T1a_bounds', [1.0, 2.5])[0]),
        'T1a_upper': float(param_config.get('T1a_bounds', [1.0, 2.5])[1]),
        'T1_lower': float(param_config.get('T1_bounds', [0.8, 2.0])[0]),
        'T1_upper': float(param_config.get('T1_bounds', [0.8, 2.0])[1]),
        'lambd_lower': float(param_config.get('lambd_bounds', [0.7, 1.1])[0]),
        'lambd_upper': float(param_config.get('lambd_bounds', [0.7, 1.1])[1]),
        'abv_lower': float(param_config.get('abv_bounds', [0.001, 0.1])[0]),
        'abv_upper': float(param_config.get('abv_bounds', [0.001, 0.1])[1]),
        'att_a_lower': float(param_config.get('att_a_bounds', [0.1, 2.0])[0]),
        'att_a_upper': float(param_config.get('att_a_bounds', [0.1, 2.0])[1])
    }

    try:
        model = stan.build(STAN_MODEL_CODE_EXT, data=data)
        fit = model.sample(num_chains=2, num_samples=1000, num_warmup=500)
        return fit
    except Exception as e:
        print(f"Error in Stan fitting (extended): {e}")
        return None


def bayesian_fit_volume_ext(pwi_data, t, m0_data, tau, param_config,
							att_ls_map=None, cbf_ls_map=None, abv_ls_map=None, att_a_ls_map=None):
	"""
	Bayesian fitting for entire volume using extended DeltaM model

	Parameters:
	- pwi_data: 4D PWI data
	- t: time points
	- m0_data: M0 data
	- tau: labeling duration
	- param_config: parameter configuration dict
	- att_ls_map: ATT map from LS fitting (for priors)
	- cbf_ls_map: CBF map from LS fitting (for priors)
	- abv_ls_map: ABV map from LS fitting (for priors)
	- att_a_ls_map: ATT_a map from LS fitting (for priors)
	"""

	shape = pwi_data.shape[:3]

	# Prepare parameter maps
	print("Preparing parameter maps...")
	T1a_map = convert_parameter(param_config['T1a'], shape, "T1a")
	T1_map = convert_parameter(param_config['T1'], shape, "T1")
	lambd_map = convert_parameter(param_config['lambd'], shape, "lambd")
	a_map = convert_parameter(param_config['a'], shape, "a")

	# Initialise result arrays
	att_map = np.full(shape, np.nan)
	cbf_map = np.full(shape, np.nan)
	att_std_map = np.full(shape, np.nan)
	cbf_std_map = np.full(shape, np.nan)

	# Additional maps for fitted parameters
	if param_config.get('fit_T1a', False):
		T1a_fitted_map = np.full(shape, np.nan)
		T1a_fitted_std_map = np.full(shape, np.nan)

	if param_config.get('fit_T1', False):
		T1_fitted_map = np.full(shape, np.nan)
		T1_fitted_std_map = np.full(shape, np.nan)

	if param_config.get('fit_lambd', False):
		lambd_fitted_map = np.full(shape, np.nan)
		lambd_fitted_std_map = np.full(shape, np.nan)

	if param_config.get('fit_abv', False):
		abv_fitted_map = np.full(shape, np.nan)
		abv_fitted_std_map = np.full(shape, np.nan)

	if param_config.get('fit_att_a', False):
		att_a_fitted_map = np.full(shape, np.nan)
		att_a_fitted_std_map = np.full(shape, np.nan)

	total_voxels = shape[0] * shape[1] * shape[2]
	processed_voxels = 0
	successful_fits = 0

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				processed_voxels += 1

				if processed_voxels % 100 == 0:
					print(f"Processing voxel {processed_voxels}/{total_voxels} "
						  f"({100 * processed_voxels / total_voxels:.1f}%)")

				# Get data for this voxel
				signal = pwi_data[x, y, z, :]
				m0 = m0_data[x, y, z]

				# Skip invalid voxels
				if (np.any(np.isnan(signal)) or np.any(np.isinf(signal)) or
						np.all(signal == 0) or np.isnan(m0) or np.isinf(m0) or m0 <= 0):
					continue

				# Get parameter values for this voxel
				T1a_val = T1a_map[x, y, z]
				T1_val = T1_map[x, y, z]
				lambd_val = lambd_map[x, y, z]
				a_val = a_map[x, y, z]

				# Skip if any parameter is invalid
				if (np.isnan(T1a_val) or np.isnan(T1_val) or
						np.isnan(lambd_val) or np.isnan(a_val)):
					continue

				# Get LS prior values
				att_ls_val = att_ls_map[x, y, z] if att_ls_map is not None else None
				cbf_ls_val = cbf_ls_map[x, y, z] if cbf_ls_map is not None else None
				abv_ls_val = abv_ls_map[x, y, z] if abv_ls_map is not None else None
				att_a_ls_val = att_a_ls_map[x, y, z] if att_a_ls_map is not None else None

				signal_normalized = signal / m0
				M0a = 1.0

				try:
					# Fit this voxel using extended model
					fit = bayesian_fit_voxel_ext(
						t, signal_normalized, M0a, tau,
						T1a_val, T1_val, lambd_val, a_val,
						param_config, att_ls_val, cbf_ls_val,
						abv_ls_val, att_a_ls_val
					)

					if fit is not None:
						# Extract results
						att_samples = fit['att'].flatten()
						cbf_samples = fit['cbf'].flatten()  # Already converted to ml/min/100g in Stan

						att_map[x, y, z] = np.mean(att_samples)
						cbf_map[x, y, z] = np.mean(cbf_samples)
						att_std_map[x, y, z] = np.std(att_samples)
						cbf_std_map[x, y, z] = np.std(cbf_samples)

						# Extract fitted parameters
						if param_config.get('fit_T1a', False):
							T1a_samples = fit['T1a_param'].flatten()
							T1a_fitted_map[x, y, z] = np.mean(T1a_samples)
							T1a_fitted_std_map[x, y, z] = np.std(T1a_samples)

						if param_config.get('fit_T1', False):
							T1_samples = fit['T1_param'].flatten()
							T1_fitted_map[x, y, z] = np.mean(T1_samples)
							T1_fitted_std_map[x, y, z] = np.std(T1_samples)

						if param_config.get('fit_lambd', False):
							lambd_samples = fit['lambd_param'].flatten()
							lambd_fitted_map[x, y, z] = np.mean(lambd_samples)
							lambd_fitted_std_map[x, y, z] = np.std(lambd_samples)

						if param_config.get('fit_abv', False):
							abv_samples = fit['abv_param'].flatten()
							abv_fitted_map[x, y, z] = np.mean(abv_samples)
							abv_fitted_std_map[x, y, z] = np.std(abv_samples)

						if param_config.get('fit_att_a', False):
							att_a_samples = fit['att_a_param'].flatten()
							att_a_fitted_map[x, y, z] = np.mean(att_a_samples)
							att_a_fitted_std_map[x, y, z] = np.std(att_a_samples)

						successful_fits += 1

				except Exception as e:
					print(f"Error fitting voxel ({x}, {y}, {z}): {e}")
					continue

	print(f"Successfully fitted {successful_fits}/{processed_voxels} voxels")

	# Results
	results = {
		'att_map': att_map,
		'cbf_map': cbf_map,
		'att_std_map': att_std_map,
		'cbf_std_map': cbf_std_map,
		'successful_fits': successful_fits,
		'total_processed': processed_voxels
	}

	# Add fitted parameter maps
	if param_config.get('fit_T1a', False):
		results['T1a_fitted_map'] = T1a_fitted_map
		results['T1a_fitted_std_map'] = T1a_fitted_std_map

	if param_config.get('fit_T1', False):
		results['T1_fitted_map'] = T1_fitted_map
		results['T1_fitted_std_map'] = T1_fitted_std_map

	if param_config.get('fit_lambd', False):
		results['lambd_fitted_map'] = lambd_fitted_map
		results['lambd_fitted_std_map'] = lambd_fitted_std_map

	if param_config.get('fit_abv', False):
		results['abv_fitted_map'] = abv_fitted_map
		results['abv_fitted_std_map'] = abv_fitted_std_map

	if param_config.get('fit_att_a', False):
		results['att_a_fitted_map'] = att_a_fitted_map
		results['att_a_fitted_std_map'] = att_a_fitted_std_map

	return results


# === Functions for input parameter conversion (numpy array, matrix/image, scalar)
def convert_parameter(param_input, shape, param_name="parameter"):
	"""
	Convert parameter input to image / matrix format

	Parameters:
	- param_input: can be scalar, filename (str), or numpy array
	- shape: target shape (x, y, z)
	- param_name: parameter name for error messages

	Returns:
	- 3D numpy array with parameter values
	"""
	if isinstance(param_input, (int, float)):
		# param_input of type float/int will be converted to 3D array
		print(f"Using constant {param_name} = {param_input}")
		return np.full(shape, param_input, dtype=float)

	elif isinstance(param_input, str):
		# param_input is loaded from file
		print(f"Loading {param_name} from file: {param_input}")
		if not os.path.exists(param_input):
			raise FileNotFoundError(f"Parameter file not found: {param_input}")

		_, data = load_nifti_file(param_input)

		# Handle 4D data (take first volume)
		if data.ndim == 4:
			data = data[:, :, :, 0]

		# Check whether shapes are similar
		if data.shape != shape:
			raise ValueError(f"Shape mismatch for {param_name}: expected {shape}, got {data.shape}")

		return data.astype(float)

	elif isinstance(param_input, np.ndarray):
		# Already a matrix
		print(f"Using provided {param_name} array")

		# Handle 4D data (take first volume)
		if param_input.ndim == 4:
			param_input = param_input[:, :, :, 0]

		# Check whether shapes are similar
		if param_input.shape != shape:
			raise ValueError(f"Shape mismatch for {param_name}: expected {shape}, got {param_input.shape}")

		return param_input.astype(float)

	else:
		raise ValueError(f"Invalid {param_name} data format. Must be scalar, filename, or numpy array")


def choose_parameter_config():
	"""
	Choose parameter configuration
	"""

	# Create parameter configuration
	param_config = create_parameter_config_from_config()

	# All parameters constant
	param_config['T1a'] = 1.65
	param_config['T1'] = 1.6
	param_config['lambd'] = 0.9

	# Enable LS-based priors
	param_config['att_prior_from_ls'] = True
	param_config['cbf_prior_from_ls'] = True
	param_config['att_prior_std'] = 0.3
	param_config['cbf_prior_std'] = 15.0

	# Load T1 maps from files (uncomment if needed)
	# param_config['T1a'] = "path/to/T1a_map.nii"
	# param_config['T1'] = "path/to/T1_map.nii"

	# Fit T1a as free parameter (uncomment if needed)
	# param_config['fit_T1a'] = True
	# param_config['T1a_prior'] = {'mean': 1.65, 'std': 0.2}

	return param_config
