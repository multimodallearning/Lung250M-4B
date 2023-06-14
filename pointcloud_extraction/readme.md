This part of the repository contains code and download links for trained models to generate lung masks, pulmonary vein and artery segmentations and extract point clouds from them.  

Considering you have folders ```../imagesTr``` and ```../imagesTs``` containing preprocessed scans, the following steps will extract the corresponding point clouds.

Generate Lung Segmentations (Generate ```.../masksTr``` and ```../masksTs```)
1. Download nnUNet_Task500_Lung.zip and extract the zip file
2. Set ```export RESULTS_FOLDER=nnUNet_trained_models``` for nnUNet inference.
3. Create temporary folders for inference: ```mkdir tmp_imagesTr_0000_shifted; mkdir tmp_imagesTs_0000_shifted```
4. Create output folders: ```mkdir ../masksTr; mkdir ../masksTs`` 
5. Copy, rename and shift images:
```
for i in {000..123}; do ../../c3d ../imagesTs/case_${i}_1.nii.gz -shift 1024 -o temp_imagesTs_0000_shifted/case_${i}_1_0000.nii.gz; ../../c3d ../imagesTs/case_${i}_2.nii.gz -shift 1024 -o temp_imagesTs_0000_shifted/case_${i}_2_0000.nii.gz; ../../c3d ../imagesTr/case_${i}_1.nii.gz -shift 1024 -o temp_imagesTr_0000_shifted/case_${i}_1_0000.nii.gz; ../../c3d ../imagesTr/case_${i}_2.nii.gz -shift 1024 -o temp_imagesTr_0000_shifted/case_${i}_2_0000.nii.gz; done
```
6. Run nnUNet inference: ```nnUNet_predict -i temp_imagesTr_0000_shifted/ -o ../masksTr/ -t 500 -m 3d_lowres``` and ```nnUNet_predict -i temp_imagesTs_0000_shifted/ -o ../masksTs/ -t 500 -m 3d_lowres```
7. Delete the temporary folders ```temp_imagesTr_0000_shifted``` and ```temp_imagesTs_0000_shifted```

Generate Vessel Segmentations (Generate ```.../segTr``` and ```../segTs```)
1. Download nnUNet_Task702_artery_vein.zip and extract the zip file
2. Set ```export RESULTS_FOLDER=nnUNet_trained_models``` for nnUNet inference.
3. Create temporary folders for inference: ```mkdir tmp_imagesTr_0000; mkdir tmp_imagesTs_0000```
4. Create output folders: ```mkdir ../segTr; mkdir ../segTs`` 
5. Copy and rename images:
```
for i in {000..123}; do cp ../imagesTr/case_${i}_1.nii.gz temp_imagesTr_0000/case_${i}_1_0000.nii.gz; cp ../imagesTr/case_${i}_2.nii.gz temp_imagesTr_0000/case_${i}_2_0000.nii.gz;../imagesTs/case_${i}_1.nii.gz temp_imagesTs_0000/case_${i}_1_0000.nii.gz; cp ../imagesTs/case_${i}_2.nii.gz temp_imagesTs_0000/case_${i}_2_0000.nii.gz; done
```
6. Run nnUNet inference: ```nnUNet_predict -i temp_imagesTr_0000/ -o ../segTr/ -t 500 -m 3d_lowres``` and ```nnUNet_predict -i temp_imagesTs_0000/ -o ../segTs/ -t 500 -m 3d_fullres -f all --disable_tta -tr nnUNetTrainerV2_noMirroring```
7. Delete the temporary folders ```temp_imagesTr_0000``` and ```temp_imagesTs_0000```

Extract point clouds and features (Generate ```../cloudsTr``` and ```../cloudsTs```)
1. Run ```python pc_generate.py```
