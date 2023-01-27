# longitudinal-neighbourhood-embedding
Self-supervised Longitudinal Neighbourhood Embedding (LNE), MICCAI 2021.
[paper](https://arxiv.org/abs/2103.03840)

Self-supervised learning of neighborhood embedding for longitudinal MRI, Medical Image Analysis 2022. (LNE-v2)
[paper](https://www.sciencedirect.com/science/article/pii/S1361841522002122)

### Dependency
conda env create -f requirement.yml

### Data Preprocessing
data_preprocessing_ADNI.py and data_preprocessing_LAB.py save images and other information in h5 files.

### Self-supervised models
change parameters in config.yml (default setting is training LNE) \
run <code>python main.py</code>

change parameters in config_v2.yml (default setting is training LNE-v2) \
run <code>python main_v2.py</code>

### Downstream classification / regression
change parameters in config.yml \
For setting _use_feature: ['z'], data_type: single_,  run <code>python main_classification_single.py</code> \
For setting _use_feature: ['z', 'delta_z'], data_type: pair_,  run <code>python main_classification_pair.py</code>

### Visualization
see visualization.ipynb for more details.
