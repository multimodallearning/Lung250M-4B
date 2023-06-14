# Lung250M-4B
Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration

This repository accompanies the NeurIPS dataset submission Lung250M-4B. To access and download the dataset itself please visit https://cloud.imi.uni-luebeck.de/s/s64fqbPpXNexBPP. The preprint (under review) of the paper and its supplementary material are available as PDFs as well.   

The code repository comprises several complementary parts:
1) Data Curation: Scripts and manifest files to download and preprocess raw Dicom image files from TCIA or Nifti files from the the original data sources. While this step is optional for all subsets that are shared with CC-BY licence, it is necessary for the COPDgene dataset and EMPIRE10. The pre-processing pipeline might also help researchers to extend the dataset with other sources.
2) CorrField: contains the automatic algorithm to obtain pseudo ground truth correspondences for paired 3D lung CT scans. Results (csv files) for all scan pairs are also available (e.g. to visualise the alignment of scans using them)
3) Pointcloud Extraction: nnUNet models for lung masks, pulmonary vein and artery segmentations and python code to extract point clouds from them.
4) Registration Models: Two versions of an image-based deep learning model to predict large deformable 3D motion (VoxelMorph++) is provided with scripts for training, testing and pre-trained models (variant 1 uses an unsupervised metric loss, whereas variant 2 uses the CorrField keypoints). Two versions of a state-of-the-art 3D point cloud registration algorithm (PointPWC) are adapted to the given datasets. Namely, an unsupervised version that purely relies on synthetically generated ground truth deformations, and a supervised version that leverages the image-based CorrField ground truth.
5) Evaluation: Landmark files, evaluation functions and visualisation code (streamlit) is also provided to assess results and the dataset both qualitatively and quantitatively. 

# Quick start
To get a first glance into the dataset and benchmark solution follow the next few steps to download an evaluation subset, run inference of point-cloud and/or image-based registration, evaluate their accuracy and visualise the overlay after registration

## Preparation
1) `git clone`
2) put the dataset into the main directory of the repository to obtain the data structure shown below.
3) create a new conda environment with the required dependencies via `conda env create -f environment.yaml`
4) for training of and inference with PointPWC-Net, please compile the `pointnet2_utils` via `cd registration_models/point_pwc/pointnet2`, `python setup.py install` 

## Dataset structure
```
    .
    ├── corrfields
    ├── data_curation
    ├── ...
    ├── segTs
    │   ├── 002_1.nii.gz
    │   ├── 002_2.nii.gz
    │   ├── 008_1.nii.gz
    │   └── ...
    ├── segTr
    │   ├── 001_1.nii.gz
    │   ├── 001_2.nii.gz
    │   ├── 003_1.nii.gz
    │   └── ...
    ├── masksTs
    │   ├── 002_1.nii.gz
    │   ├── 002_2.nii.gz
    │   ├── 008_1.nii.gz
    │   └── ...
    ├── masksTr
    │   ├── 001_1.nii.gz
    │   ├── 001_2.nii.gz
    │   ├── 003_1.nii.gz
    │   └── ...
    ├── imagesTs
    │   ├── 002_1.nii.gz
    │   ├── 002_2.nii.gz
    │   ├── 008_1.nii.gz
    │   └── ...
    ├── imagesTr
    │   ├── 001_1.nii.gz
    │   ├── 001_2.nii.gz
    │   ├── 003_1.nii.gz
    │   └── ...
    ├── corrfieldTr
    │   ├── case_001.csv 
    │   ├── case_003.csv
    │   └── ...
    ├── cloudsTr
    │   ├── coordinates
    │   │   ├── case_001_1.pth  --> list of three clouds: [8k, skeletonized, full]
    │   │   ├── case_001_2.pth
    │   │   ├── case_003_1.pth
    │   │   └── ...
    │   ├── distance
    │   │   ├── case_001_1.pth  --> features from euclidean distance transform for the 3 clouds
    │   │   ├── case_001_2.pth
    │   │   ├── case_003_1.pth
    │   │   └── ...
    │   └── artery_vein
    │       ├── case_001_1.pth  --> artery/vein labels for the 3 clouds
    │       ├── case_001_2.pth
    │       ├── case_003_1.pth
    │       └── ...
    ├── cloudsTs
    │   ├── coordinates
    │   │   ├── case_002_1.pth
    │   │   ├── case_002_2.pth
    │   │   ├── case_008_1.pth
    │   │   └── ...
    │   ├── distance
    │   │   ├── case_002_1.pth
    │   │   ├── case_002_2.pth
    │   │   ├── case_008_1.pth
    │   │   └── ...
    │   └── artery_vein
    │       ├── case_002_1.pth
    │       ├── case_002_2.pth
    │       ├── case_008_1.pth
    │       └── ...
    └── ...
```

## inference with pre-trained models
- for Voxelmorph++ (VM++ w/ IO in the paper), execute `python registration_models/inference_vxmpp.py` while setting the correct data paths and output paths in the command line, following the description in the main function of the file.
- for PointPWC-Net, move to `registration_models/point_pwc` and follow the isntructions in the separate readme.

## evaluation and visualization
Once the displacement fields are inferred (the ones for VM++ are already available in `predict_vxmpp` ) you can both visualize the registrations and just print the case-wise errors as follows.
- visualization: execute `streamlit run evaluation/visualise_and_evaluate_image.py -- -I imagesTs -o predict_vxmpp --validation evaluation/lms_validation.pth`
- evaluation: execute `python evaluation/evaluate.py -I imagesTs -o predict_vxmpp --validation evaluation/lms_validation.pth`

In both cases `predict_vxmpp` can be replaced by any folder with made predictions.
```
git clone 
pip install -r requirements.txt
wget ..
python inference.py model evaluation_folder 
streamlit run visualise_lungCT.py 

Here is some example output (abbreviated):
...


