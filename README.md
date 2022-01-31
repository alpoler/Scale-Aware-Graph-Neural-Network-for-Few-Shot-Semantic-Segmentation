# Scale-Aware-Graph-Neural-Network-for-Few-Shot-Semantic-Segmentation

1. Change data root directory in ayar.yaml as path of VOC-2012 folder

2. Change train list directory in ayar.yaml so that it shows "voc_sbd_train.txt" in your project directory

3. Change val list directory in ayar.yaml so that it shows "val.txt" in your project directory.

python train.py - Train + Cross Validation

It is enough that "best_model.pth" is placed with path of learned best model before inference.

python inference.py - Inference





