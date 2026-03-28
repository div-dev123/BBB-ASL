import numpy as np
import matplotlib.pyplot as plt

"""
Multi-TE ASL Model based on the paper and belonging c++ code by Amnah Mahroo:
Robust Multi-TE ASL-Based Blood–Brain Barrier Integrity Measurements. Front Neuroscience 15. (2021)
https://doi.org/10.3389/fnins.2021.719676

Implements the three-compartment model with T1 and T2 decay
"""

def deltaM_multite_model(tis, tes, ntes, att, cbf, m0a, taus,
						 t1=1.3, t1b=1.65, t2=0.050, t2b=0.150,
						 texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):
	"""
	Parameters:
	- tis: array of inversion times [s]
	- tes: array of echo times [s]
	- ntes: array of number of TEs per TI
	- att: arterial transit time [s]
	- cbf: cerebral blood flow  [ml/min/100g]
	- m0a: scaling factor
	- taus: array or single value of labeling durations per TI [s]
	- t1: T1 of tissue [s]
	- t1b: T1 of blood [s]
	- t2: T2 of tissue [s]
	- t2b: T2 of blood [s]
	- texch: exchange time [s]
	- itt: intra-voxel transit time [s]
	- lambd: blood-tissue partition coefficient
	- alpha: labeling efficiency

	Returns:
	- deltaM: signal for all TE/TI combinations
	"""

	# Convert CBF from ml/min/100g to ml/s/g
	# 60s -> 1s, 100g -> 1g,  1/6000 similar to scaling in one compartment model (see DeltaM_model.py)
	f = cbf / 100.0 * 60.0 / 6000.0

	# Handle taus input - can be single value or array
	if np.isscalar(taus):
		tau_array = np.full(len(tis), taus)
	else:
		tau_array = np.array(taus)

	# Initialise result arrays
	S_bl1_final = np.zeros(len(tes))
	S_bl2_final = np.zeros(len(tes))
	S_ex_final = np.zeros(len(tes))

	te_index = 0

	# Loop over all inversion times
	# C++: for (int j = 0; j < tis.size(); ++j)

	for j in range(len(tis)):
		tau = tau_array[j]
		ti = tis[j]

		# Case 1: 0 < ti < att
		if 0 < ti < att:
			for k in range(ntes[j]):
				S_bl1_final[te_index] = 0.0
				S_bl2_final[te_index] = 0.0
				S_ex_final[te_index] = 0.0
				te_index += 1

		# Case 2: att <= ti < (att + itt)
		elif att <= ti < (att + itt):
			for k in range(ntes[j]):
				te = tes[te_index]

				# Subcase 1: 0 <= te < (att+itt-ti)
				if 0 <= te < (att + itt - ti):
					S_bl1_final[te_index] = (2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) *
											 (np.exp(ti / t1b) - np.exp(att / t1b)) * np.exp(-te / t2b))

					S_bl2_final[te_index] = 0.0

					S_ex_final[te_index] = 0.0

				# Subcase 2: (att+itt-ti) <= te < itt
				elif (att + itt - ti) <= te < itt:
					base_term = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp(att / t1b))
					transition_factor = (te - (att + itt - ti)) / (ti - att)

					S_bl1_final[te_index] = ((base_term - transition_factor * base_term) * np.exp(-te / t2b))

					S_bl2_final[te_index] = (transition_factor * base_term * np.exp(-te / t2b) * np.exp(-te / texch))

					S_ex_final[te_index] = (
								transition_factor * base_term * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

				# Subcase 3: te >= itt
				else:
					S_bl1_final[te_index] = 0.0

					S_bl2_final[te_index] = (2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) *
											 (np.exp(ti / t1b) - np.exp(att / t1b)) * np.exp(-te / t2b) * np.exp(
								-te / texch))

					S_ex_final[te_index] = (2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) *
											(np.exp(ti / t1b) - np.exp(att / t1b)) * (1 - np.exp(-te / texch)) * np.exp(
								-te / t2))

				te_index += 1

		# Case 3: (att+itt) <= ti < (att + tau)
		elif (att + itt) <= ti < (att + tau):
			for k in range(ntes[j]):
				te = tes[te_index]

				# Subcase 1: 0 <= te < itt
				if 0 <= te < itt:
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))
					base_diff = term1 - term2
					transition_factor = te / itt

					S_bl1_final[te_index] = ((base_diff - transition_factor * base_diff) * np.exp(-te / t2b))

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  transition_factor * base_diff) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 transition_factor * base_diff) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				# Subcase 2: te >= itt
				else:
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))

					S_bl1_final[te_index] = 0.0

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  (term1 - term2)) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 (term1 - term2)) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				te_index += 1

		# Case 4: (att + tau) <= ti < (att + itt + tau)
		elif (att + tau) <= ti < (att + itt + tau):
			for k in range(ntes[j]):
				te = tes[te_index]

				# Subcase 1: 0 <= te < (itt-(ti-(att+tau)))
				if 0 <= te < (itt - (ti - (att + tau))):
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp((att + tau) / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))
					base_diff = term1 - term2
					transition_factor = te / (itt - (ti - (att + tau)))

					S_bl1_final[te_index] = ((base_diff - transition_factor * base_diff) * np.exp(-te / t2b))

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  transition_factor * base_diff) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 transition_factor * base_diff) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				# Subcase 2: te >= (itt-(ti-(att+tau)))
				else:
					term1 = 2 * f * t1b * np.exp(-att / t1b) * np.exp(-ti / t1b) * (
								np.exp((att + tau) / t1b) - np.exp(att / t1b))
					term2 = 2 * f * t1b * np.exp(-(att + itt) / t1b) * np.exp(-ti / t1b) * (
								np.exp(ti / t1b) - np.exp((att + itt) / t1b))

					S_bl1_final[te_index] = 0.0

					S_bl2_final[te_index] = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
											  np.exp(-((1 / t1b) + (1 / texch)) * ti) *
											  (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
												  ((1 / t1b) + (1 / texch)) * (att + itt))) +
											  (term1 - term2)) * np.exp(-te / t2b) * np.exp(-te / texch))

					ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
								(np.exp((1 / t1) * ti) - np.exp((1 / t1) * (att + itt))) -
								(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
								np.exp(-((1 / t1) + (1 / texch)) * ti) *
								(np.exp(((1 / texch) + (1 / t1)) * ti) - np.exp(
									((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

					ex_term2 = ((2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
								 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
								 (np.exp(((1 / t1b) + (1 / texch)) * ti) - np.exp(
									 ((1 / t1b) + (1 / texch)) * (att + itt))) +
								 (term1 - term2)) * (1 - np.exp(-te / texch)) * np.exp(-te / t2))

					S_ex_final[te_index] = ex_term1 + ex_term2

				te_index += 1

		# Case 5: ti >= (att + itt + tau)
		else:
			for k in range(ntes[j]):
				te = tes[te_index]

				S_bl1_final[te_index] = 0.0

				S_bl2_final[te_index] = (2 * f * np.exp(-(1 / t1b) * (att + itt)) / ((1 / t1b) + (1 / texch)) *
										 np.exp(-((1 / t1b) + (1 / texch)) * ti) *
										 (np.exp(((1 / t1b) + (1 / texch)) * (att + itt + tau)) - np.exp(
											 ((1 / t1b) + (1 / texch)) * (att + itt))) *
										 np.exp(-te / t2b) * np.exp(-te / texch))

				ex_term1 = ((2 * f * np.exp(-(1 / t1b) * (att + itt))) / (1 / t1) * np.exp(-(1 / t1) * ti) *
							(np.exp((1 / t1) * (att + itt + tau)) - np.exp((1 / t1) * (att + itt))) -
							(2 * f * np.exp(-(1 / t1b) * (att + itt))) / ((1 / texch) + (1 / t1)) *
							np.exp(-((1 / t1) + (1 / texch)) * ti) *
							(np.exp(((1 / texch) + (1 / t1)) * (att + itt + tau)) - np.exp(
								((1 / texch) + (1 / t1)) * (att + itt)))) * np.exp(-te / t2)

				ex_term2 = ((2 * f * np.exp(-((1 / t1b)) * (att + itt)) / (((1 / t1b)) + (1 / texch)) *
							 np.exp(-(((1 / t1b)) + (1 / texch)) * ti) *
							 (np.exp(((1 / t1b) + (1 / texch)) * (att + itt + tau)) - np.exp(
								 ((1 / t1b) + (1 / texch)) * (att + itt)))) *
							(1 - np.exp(-te / texch)) * np.exp(-te / t2))

				S_ex_final[te_index] = ex_term1 + ex_term2

				te_index += 1

	# Sum all three compartments to get final signal
	delta_M_final = S_bl1_final + S_bl2_final + S_ex_final

	# Apply scaling factors
	delta_M_final *= m0a * alpha / lambd

	return delta_M_final



"""
Plot of ASL signal
"""
def plot_multite_signal():
	# Inversion time [s]
	tis = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
	# Echo time [s]
	tes_per_ti = 4
	te_range = np.linspace(0.005, 0.100, tes_per_ti)

	# Create all TE for each TI
	tes = np.tile(te_range, len(tis))

	# Number of TE per TI
	ntes = np.full(len(tis), tes_per_ti)

	# Pharmacologial paramter
	att = 0.8  # Arterial Transit Time [s]
	cbf = 60.0  # Cerebral Blood Flow [ml/min/100g]
	m0a = 1000.0  # scaling
	taus = 1.8  # Labeling Duration [s]

	t1 = 1.3  # T1 tissue [s]
	t1b = 1.65  # T1 blood [s]
	t2 = 0.050  # T2 tissue [s]
	t2b = 0.150  # T2 blood [s]
	texch = 0.1  # Exchange time [s]
	itt = 0.2  # Intra-voxel transit time [s]

	# Compute signal with model equation
	signal = deltaM_multite_model(
		tis, tes, ntes, att, cbf, m0a, taus,
		t1=t1, t1b=t1b, t2=t2, t2b=t2b,
		texch=texch, itt=itt
	)

	# Plotting
	# Signal vs TI for different TE
	fig2, ax2 = plt.subplots(figsize=(10, 6))

	colors = ['red', 'blue', 'green', 'orange']
	for te_idx in range(tes_per_ti):
		# Signal for these TE over all TI
		te_signal = signal[te_idx::tes_per_ti]
		te_value = te_range[te_idx] * 1000  # in ms

		ax2.plot(tis, te_signal, 'o-', color=colors[te_idx],
				 linewidth=2, markersize=6,
				 label=f'TE = {te_value:.1f} ms')

	ax2.set_xlabel('Inversion Time (s)')
	ax2.set_ylabel('Signal (a.u.)')
	ax2.set_title('Multi-TE ASL Signal vs Inversion Time')
	ax2.legend()
	ax2.grid(True, alpha=0.3)
	ax2.set_ylim(bottom=0)

	plt.tight_layout()
	plt.show()


def plot_parameter_sensitivity():
	"""Plotting sensitivity of the signal for different parameters"""

	tis = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
	tes = np.linspace(0.005, 0.100, 4)
	tes_all = np.tile(tes, len(tis))
	ntes = np.full(len(tis), len(tes))

	initial_param = {
		'att': 0.8, 'cbf': 60.0, 'm0a': 1000.0, 'taus': 1.8,
		't1': 1.3, 't1b': 1.65, 't2': 0.050, 't2b': 0.150,
		'texch': 0.1, 'itt': 0.2
	}

	param_variations = {
		'CBF': {'param': 'cbf', 'values': [30, 60, 90], 'unit': 'ml/min/100g'},
		'ATT': {'param': 'att', 'values': [0.6, 0.8, 1.0], 'unit': 's'},
		'Exchange Time': {'param': 'texch', 'values': [0.05, 0.1, 0.2], 'unit': 's'},
		'Transit Time': {'param': 'itt', 'values': [0.1, 0.2, 0.3], 'unit': 's'}
	}

	fig, axes = plt.subplots(2, 2, figsize=(12, 10))
	axes = axes.flatten()

	colors = ['blue', 'orange', 'green']

	for i, (param_name, param_info) in enumerate(param_variations.items()):
		ax = axes[i]

		for j, value in enumerate(param_info['values']):
			# Parameter for these variations
			params = initial_param.copy()
			params[param_info['param']] = value

			# Calculating signal
			signal = deltaM_multite_model(tis, tes_all, ntes, **params)

			# Mean value over all TEs for each TI
			signal_mean = np.mean(signal.reshape(len(tis), -1), axis=1)

			ax.plot(tis, signal_mean, 'o-', color=colors[j],
					linewidth=2, markersize=6,
					label=f'{param_name} = {value} {param_info["unit"]}')

		ax.set_xlabel('Inversion Time (s)')
		ax.set_ylabel('Mean Signal (a.u.)')
		ax.set_title(f'Sensitivity to {param_name}')
		ax.legend()
		ax.grid(True, alpha=0.3)
		ax.set_ylim(bottom=0)

	plt.tight_layout()
	plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, y=1.02)
	plt.show()

#Added print statements to confirm simulations are running
if __name__ == "__main__":
	print("Running multi-TE signal plot simulation...")
	plot_multite_signal()
	print("Running parameter sensitivity analysis simulation...")
	plot_parameter_sensitivity()
	print("Simulation completed successfully!")
