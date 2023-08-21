# Lung250M-4B
Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration

This repository accompanies the NeurIPS dataset submission Lung250M-4B. To access and download the dataset itself please visit https://cloud.imi.uni-luebeck.de/s/s64fqbPpXNexBPP. The preprint (under review) of the paper and its supplementary material are available as PDFs under `figures_pdf` as well.   

The code repository comprises several complementary parts:
1) Data Curation: Scripts and manifest files to download and preprocess raw Dicom image files from TCIA or Nifti files from the the original data sources. While this step is optional for all subsets that are shared with CC-BY licence, it is necessary for the COPDgene dataset and EMPIRE10. The pre-processing pipeline might also help researchers to extend the dataset with other sources.
2) CorrField: contains the automatic algorithm to obtain pseudo ground truth correspondences for paired 3D lung CT scans. Results (csv files) for all scan pairs are also available (e.g. to visualise the alignment of scans using them)
3) Pointcloud Extraction: nnUNet models for lung masks, pulmonary vein and artery segmentations and python code to extract point clouds from them.
4) Registration Models: Two versions of an image-based deep learning model to predict large deformable 3D motion (VoxelMorph++) is provided with scripts for training, testing and pre-trained models (variant 1 uses an unsupervised metric loss, whereas variant 2 uses the CorrField keypoints). Two versions of a state-of-the-art 3D point cloud registration algorithm (PointPWC) are adapted to the given datasets. Namely, an unsupervised version that purely relies on synthetically generated ground truth deformations, and a supervised version that leverages the image-based CorrField ground truth.
5) Evaluation: Landmark files, evaluation functions and visualisation code (streamlit) is also provided to assess results and the dataset both qualitatively and quantitatively. 

After the quick start guide you can find more details about the benchmark methods and visualisations under "Benchmarks" and "Results". If you are interesent to make your research results on Lung250M-4B publically available under this GitHub, please contact us, so we can add it to this page.

# Quick start
To get a first glance into the dataset and benchmark solution follow the next few steps to download an evaluation subset, run inference of point-cloud and/or image-based registration, evaluate their accuracy and visualise the overlay after registration

## Preparation
1) `git clone`
2) put the dataset into the main directory of the repository to obtain the data structure shown below.
3) create a new conda environment with the required dependencies via `conda env create -f environment.yaml`
4) for training of and inference with PointPWC-Net, please compile the `pointnet2_utils` via `cd registration_models/point_pwc/pointnet2/lib`, `python setup.py install` 

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
The following case numbers belong to images from respective sub-datasets:

000-010: Empire10 \
012-033: TCIA-NLST \
034-053: TCIA-NSCLC \
054-083: L2R-LungCT \
084-103: TCIA-Ventilation \
104-113: DIR-LAB COPDgene \
114-123: TCIA-NLST (Validation)

Cases 002, 008, 054, 055, 056, 094, 097 and 113-123 are validation cases. Cases 104-113 are test cases.

## inference with pre-trained models
- for Voxelmorph++ (VM++ w/ IO in the paper), execute `python registration_models/inference_vxmpp.py --outfolder predict_vxmpp` while setting the correct data paths and output paths in the command line, following the description in the main function of the file.
- for PointPWC-Net, move to `registration_models/point_pwc` and follow the instructions in the separate readme.

## evaluation and visualization
Once the displacement fields are inferred (the ones for VM++ are already available in `predict_vxmpp` ) you can both visualize the registrations and just print the case-wise errors as follows.
- visualization: execute `streamlit run evaluation/visualise_and_evaluate_image.py -- -I imagesTs -o predict_vxmpp --validation evaluation/lms_validation.pth`
- evaluation: execute `python evaluation/evaluate.py -I imagesTs -o predict_vxmpp --validation evaluation/lms_validation.pth`

In both cases `predict_vxmpp` can be replaced by any folder with made predictions.
```
python registration_models/inference_vxmpp.py --outfolder predict_vxmpp 
python evaluation/evaluate.py -I imagesTs -o predict_vxmpp --validation evaluation/lms_validation.pth

#Here is some example output (abbreviated):
tre0 7.869 tre_aff 5.722 tre2 1.270
tre0 14.618 tre_aff 9.748 tre2 1.817
tre0 12.704 tre_aff 7.889 tre2 3.618
tre0 19.609 tre_aff 11.188 tre2 2.954
tre0 26.038 tre_aff 8.470 tre2 2.871
tre0 21.532 tre_aff 10.090 tre2 2.062
tre0 15.822 tre_aff 6.766 tre2 2.737
...

```

# Benchmark 
Two types of 3D registration models can be applied to Lung250M-4B: imaged-based or point-based methods. 

For the former, we employ **VoxelMorph++** (https://github.com/mattiaspaul/VoxelMorphPlusPlus) an extension of the popular framework of Dalca et al. (https://github.com/voxelmorph/voxelmorph). The method is shown as concept figure below and described in detail in 

"Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation." by Heinrich, Mattias P., and Lasse Hansen. In Biomedical Image Registration: 10th International Workshop, WBIR 2022, Munich, Germany, July 10–12, 2022, Proceedings, pp. 85-95. Cham: Springer International Publishing, 2022 (https://arxiv.org/abs/2203.00046)

![Overview figure](figures_pdf/wbir2022_voxelmorph2.png)

The two major advances are the use of a deconvolution heatmap regression head and Adam instance optimisation that excelled on various lung and abdominal registration tasks. The method requires keypoints in the fixed scan to sample interest locations for which the heatmap regressor predicts softmax probabilities for a wide range (11x11x11) of discretised displacements. It can be trained with strong supervision, that means when one-to-one keypoint correspondences are available the target registration error (TRE) between (pseudo) ground truth and prediction is used (see registration_models/train_vxmpp_supervised.py). Alternatively, a metric-based unsupervised loss that combines MIND feature similarity and a Laplacian regularisation on the keypoint graph can be employed (see registration_models/train_vxmpp_MIND_unsupervised.py). Note, that VoxelMorph++ outperforms several SOTA works that are versatile for medical image registration on the COPD dataset including LapIRN 1st ranked in Learn2Reg 2020/21 (https://github.com/cwmok/LapIRN)

As SOTA approach for deformable point-cloud registration we employ **Point-PWC Net** (https://github.com/DylanWusee/PointPWC). The method forms the backbone for the previous best performance on COPD registration using point clouds (cf. RobOT https://github.com/uncbiag/robot) and we demonstrate that using our new Lung250M-4B dataset those results can be outperformed by a large margin.

"PointPWC-Net: Cost volume on point clouds for (self-) supervised scene flow estimation." by Wu, Wenxuan, Zhi Yuan Wang, Zhuwen Li, Wei Liu, and Li Fuxin.  In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pp. 88-107. Springer International Publishing, 2020. (https://arxiv.org/abs/1911.12408). 

# Results
The following table compares both variants of the two orthogonal approaches of 3D deformable registration (left: imaged-based methods, right: point-based methods) quantitatively in terms of mean TRE (in mm) and 25/50/75% percentiles on our new Lung250M-4B dataset. IO: instance optimisation

| Method | TRE | 25% | 50% | 75% | Method | TRE | 25% | 50% | 75% | 
| :----- | :-: | :-: | :-: | :-: | :----- | :-: | :-: | :-: | :-: |
| initial | 16.25 | 10.14 | 15.94 | 21.76 | initial | 16.25 | 10.14 | 15.94 | 21.76 |
| corrField | 1.45 | 0.83 | 1.23 | 1.77 | CPD | 3.13 | 1.51 | 2.28 | 3.58 |
| deeds | 1.53 | 0.83 | 1.22 | 1.79 | CPD with labels | 2.59 | 1.36 | 2.01 | 3.16 |
| Voxelmorph+ without IO | 6.53 | 3.38 | 5.82 | 8.50 | PPWC supervised | 2.85 | 1.52 | 2.33 | 3.54 |
| Voxelmorph+ with IO | 4.31 | 0.87 | 1.55 | 7.42 | PPWC synthetic | 2.73 | 1.52 | 2.28 | 3.45 |
| Voxelmorph++ without IO | 4.47 | 2.41 | 3.74 | 5.69 | | | | | |
| Voxelmorph++ with IO | 2.26 | 0.75 | 1.16 | 1.90 | | | | | |

To get a better visual impression of the challenges of 3D lung registration and the differences in representation of this data as 3D volumetric scans or sparse geometric point clouds the following two figures from our supplementary material show before and after overlays of three different registration pairs from Lung250M-4B.

![Visual Vxmpp](figures_pdf/reg_example.png)

![Visual ppwc](figures_pdf/qual_results_ppwc.png)



