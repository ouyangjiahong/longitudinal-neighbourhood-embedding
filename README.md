# longitudinal-neighbourhood-embedding
Self-supervised Longitudinal Neighbourhood Embedding (LNE), Accepted by MICCAI2021.
[paper](https://arxiv.org/abs/2103.03840)

NOTE: The code for our MEDIA extension will be updated here by July 20, 2022.

### Dependency
conda env create -f requirement.yml

### Data Preprocessing
data_preprocessing_ADNI.py and data_preprocessing_LAB.py save images and other information in h5 files.

### Self-supervised models
change parameters in config.yml (default setting is training LNE) \
run <code>python main.py</code>

### Downstream classification / regression
change parameters in config.yml \
For setting _use_feature: ['z'], data_type: single_,  run <code>python main_classification_single.py</code> \
For setting _use_feature: ['z', 'delta_z'], data_type: pair_,  run <code>python main_classification_pair.py</code>

### Visualization
see visualization.ipynb for more details.
