import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def view_nifti(file_path, output_png=None):
    """
    Load a NIfTI file and plot its middle slices.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    img = nib.load(file_path)
    data = img.get_fdata()
    
    # Handle 4D data (take first volume)
    if data.ndim == 4:
        data = data[..., 0]
    
    # Get middle slice indices
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2
    mid_z = data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial slice
    axes[0].imshow(np.rot90(data[:, :, mid_z]), cmap='hot')
    axes[0].set_title(f'Axial (Slice {mid_z})')
    axes[0].axis('off')
    
    # Coronal slice
    axes[1].imshow(np.rot90(data[:, mid_y, :]), cmap='hot')
    axes[1].set_title(f'Coronal (Slice {mid_y})')
    axes[1].axis('off')
    
    # Sagittal slice
    axes[2].imshow(np.rot90(data[mid_x, :, :]), cmap='hot')
    axes[2].set_title(f'Sagittal (Slice {mid_x})')
    axes[2].axis('off')
    
    plt.suptitle(f'NIfTI Visualization: {os.path.basename(file_path)}')
    plt.tight_layout()
    
    if output_png:
        plt.savefig(output_png)
        print(f"Saved visualization to {output_png}")
    
    # Note: plt.show() might not work in headless environments
    try:
        if 'MPLBACKEND' not in os.environ or os.environ['MPLBACKEND'] != 'Agg':
            plt.show()
    except:
        pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_nifti.py <nifti_file> [output_png]")
    else:
        path = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else None
        view_nifti(path, out)
