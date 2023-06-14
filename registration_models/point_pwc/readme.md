# Training
To train a model, execute `train.py --config CONFIG --gpu GPU -CTr CLOUDS_TRAIN -CVal CLOUDS_VAL`, with `CONFIG` from `{config_ppwc_sup.yaml,config_ppwc_syn.yaml}` depending on whether to perform supervised training or training on synthetic deformations and `CLOUDS_TRAIN`, `CLOUDS_VAL` being the pathes to the point cloud directories.

# Inference
To perform inference with a model, execute `inference.py --config CONFIG --gpu GPU -M MODEL_FILE -C CLOUDS_TEST - O OUTFILE`
with `CONFIG` from `{config_ppwc_sup.yaml,config_ppwc_syn.yaml}`, `MODEL_FILE` being the path ot the trained model,
`CLOUDS_TEST` the path to the directory with validation clouds.
Inferred displacment fields will be written to `OUTFILE`, which can then be evaluated and visualized as described in the main readme.
Pretrained model files are available under the same link as the dataset.
