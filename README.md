# ASD-GNN

### Pre-Requisite
- ABIDE II fMRI images should be present in testdata directory.
- Update labels.csv with file name and 0 for ASD and 1 for control.

### Steps to Run
- Once the testdata is available in testdata directory
- run create_fc_matrices.py to generate fc_matrices
- Once fc_matrices are generated run main.py to train the model

### Study Evaluated Accuracy
- 60%
