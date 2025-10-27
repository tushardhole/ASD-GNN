import os
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets


def create_fc_matrices():
    # Hardcoded paths
    input_dir = './testdata'
    output_dir = './testdata/fc_matrices'

    # Fetch AAL atlas automatically
    atlas = datasets.fetch_atlas_aal()
    atlas_filename = atlas['maps']

    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all nii.gz files in input_dir
    for file in os.listdir(input_dir):
        if file.endswith('.nii.gz'):
            filepath = os.path.join(input_dir, file)
            print(f"Processing {filepath} ...")

            # Load fMRI image
            img = nib.load(filepath)

            # Extract time series from atlas regions
            try:
                time_series = masker.fit_transform(img)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

            # Compute correlation matrix
            corr_matrix = np.corrcoef(time_series.T)

            # Save FC matrix as .npy
            out_file = os.path.join(output_dir, file.replace('.nii.gz', '.npy'))
            np.save(out_file, corr_matrix)
            print(f"Saved FC matrix to {out_file}")


if __name__ == '__main__':
    create_fc_matrices()