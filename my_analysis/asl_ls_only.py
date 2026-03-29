import sys
import os
import json
import numpy as np

# ── ensure the bbb_exchange package dir is on the path ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BBB_EXCHANGE_DIR = os.path.join(SCRIPT_DIR, "..", "src", "bbb_exchange")
sys.path.insert(0, BBB_EXCHANGE_DIR)

from data_handling import load_nifti_file, load_json_metadata, save_nifti
from fitting_single_te import ls_fit_volume

def run_ls_only():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load config
    config_path = os.path.join(BBB_EXCHANGE_DIR, "config.json")
    with open(config_path, "r") as file:
        config = json.load(file)
    
    # 2. Data paths
    data_dir = os.path.join(script_dir, "data", "1TE")
    pwi_path = os.path.join(data_dir, "PWI4D.nii")
    pwi_json_path = os.path.join(data_dir, "PWI4D.json")
    m0_path = os.path.join(data_dir, "M0.nii")
    
    print(f"Loading data from {data_dir}...")
    pwi_img, pwi_data_full = load_nifti_file(pwi_path)
    m0_img, m0_data_full = load_nifti_file(m0_path)
    pwi_meta = load_json_metadata(pwi_json_path)
    
    plds = np.array(pwi_meta["PostLabelingDelay"])
    tau = pwi_meta.get("LabelingDuration")
    if isinstance(tau, list): tau = tau[0]
    
    # Handle M0 dimensions
    if m0_data_full.ndim == 4 and m0_data_full.shape[3] == 1:
        m0_data = m0_data_full[:, :, :, 0]
    else:
        m0_data = m0_data_full

    # Sort data by PLD
    sorted_indices = np.argsort(plds)
    t = plds[sorted_indices]
    pwi_data = pwi_data_full[..., sorted_indices]
    
    print(f"Running LS fitting for {pwi_data.shape[:3]} volume...")
    
    # 3. Fitting
    att_map, cbf_map = ls_fit_volume(
        pwi_data, t, m0_data, tau,
        config['physiological']['lambd'],
        config['physiological']['T1'],
        config['physiological']['T1a'],
        config['physiological']['a']
    )
    
    # 4. Save results
    save_nifti(att_map, pwi_img, os.path.join(data_dir, "ATT_map_LS.nii.gz"))
    save_nifti(cbf_map, pwi_img, os.path.join(data_dir, "CBF_map_LS.nii.gz"))
    
    print("LS fitting completed and maps saved.")

if __name__ == "__main__":
    run_ls_only()
