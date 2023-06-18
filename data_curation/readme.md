This part of the repository contains code to download and preprocess the 3D image volumes of the dataset and how to extend it with own/public data.

## Complete Lung250M-4B data with indirect access

We can provide direct access to all subdatasets hosted on TCIA and Zenodo with CC-BY licences, but we require a small manual step for EMPIRE and COPD.
For those two subsets the data has to be downloaded in its original format from the data providers. For EMPIRE please head to https://empire10.grand-challenge.org/Details/ and read the rules for data download (it is directly accessible through Google Drive without password). This means amongst others you confirm that you intent to use the data to work on algorithms that may be submitted to the EMPIRE10 challenge. Next you an run our bash script to download the indidually zip files and extract them to the data_curation folder.
```
chmod +x download_external.sh
bash download_external.sh
```
Finally you need to call 
```
python process_empire.py
```
from within the data_curation folder which will add the missing 24 files ```case_0??_?.nii.gz``` (00-11) to ```../imagesTs``` and ```../imagesTr``` respectively.

For Dirlab-COPD you need fill in one form at https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/access-request-form.html for data-access to retrive the password. Now you could individually download all 20 scans from their dropbox. But we provide an automated script that helps you get these downloads automatically, just unzip cookies.zip with the password and run:
```
chmod +x download_external_copd.sh
bash download_external_copd.sh
```
Finally you need to call 
```
python process_copd.py
```

Now you are all set to evaluate the pre-trained models. :tada:


## Extend Lung250M-4B with own/public data

Since we provide the pre-processed data and/or lung masks already for the whole Lung250M-4B dataset, this tutorial explains how you can potentially download and process additional CT scans from TCIA. Here we focus on 10 cases from NLST which (contains some specific details but) is otherwise general enough to be adapted to any **paired** TCIA data.

First you need to install the NBIA data retriever:
```
wget https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4/nbia-data-retriever-4.4.deb
sudo -S dpkg -i nbia-data-retriever-4.4.deb
```
on a headless server you may ignore warnings about 'xdg-desktop-menu: not found'.

Next (given all globale prerequisites are installed) you can run the following command to process all cases out of 114..123:
```python nbia_pydicom_pre.py --manifest nlst_test_ni.csv --imgfolder img_temp/``

We provide a pre-trained nnUNet model for lung segmentation, from our cloud folder of the dataset download the file  
extract zip file and set export RESULTS_FOLDER= nnUNet_trained_models
4) mkdir tmp_img_0000; mkdir temp_mask for i in {114..123}; do cp ../imagesTs/case_${i}_1.nii.gz temp_img_0000/case_${i}_1_0000.nii.gz; cp ../imagesTs/case_${i}_2.nii.gz temp_img_0000/case_${i}_2_0000.nii.gz; done````
5)CUDA_VISIBLE_DEVICES=2 nnUNet_predict -i temp_img_0000/ -o temp_mask/ -t 500 -m 3d_lowres```


CUDA_VISIBLE_DEVICES=2 nnUNet_predict -i img_temp/ -o temp_mask2/ -t 500 -m 3d_lowres
