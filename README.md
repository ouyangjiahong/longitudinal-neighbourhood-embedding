# longitudinal-neighbourhood-embedding
Self-supervised Longitudinal Neighbourhood Embedding (LNE), Submitted to MICCAI2021

### Dependency
conda env create -f requirement.yml

### Data Preprocessing
data_preprocessing_ADNI.py and data_preprocessing_LAB.py save images and other information in h5 files.

### Self-supervised models
change parameters in config.yml (default setting is training LNE)
run <code>python main.py</code>

### Downstream classification / regression
change parameters in config.yml
For setting _use_feature: ['z'], data_type: single_,  run <code>python main_classification_single.py</code> \
For setting _use_feature: ['z', 'delta_z'], data_type: pair_,  run <code>python main_classification_pair.py</code>
