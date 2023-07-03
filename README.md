# WaterSeg: Lake Toushka Waterbodies Segmentation from Sentinel-1 SAR imagery


This project aims to perform waterbodies segmentation of Toushka Lakes, a chain of lakes in southern Egypt’s New Valley. The ambitious Toshka Lakes project was designed to provide irrigation for new agricultural developments, and to attract people to the region and away from the dense populations of the Nile Valley itself. 

The segmentation task is achieved using Sentinel-1 SAR imagery with the PyTorch framework and the U-Net architecture. To determine the most effective integration of SAR imagery, polarization information, and terrain features, the training was performed on four different combinations of the dataset. Each combination aimed to evaluate the impact of specific data sources on the segmentation model's performance. The metrics were compared to identify the combination that yields the best results.
## Table of Contents

- [**Project Description**](#project_des)
- [**Study Area**](#study_area)
- [**Datasets**](#dataset)
- [**Challenges**](#challenge)
- [**Results Analysis**](#res_analsis)
- [**Installation**](#install)
- [**Citing**](#citing)
- [**Contributing**](#contribute)
- [**Acknowledgement**](#ack)
- [**License**](#lic)


## Project Description <a name="project_des"></a>

The objective of this project is to develop a deep learning model that can accurately identify the water bodies of Toushka Lakes from SAR imagery. The segmentation task involves classifying each pixel in SAR satellite imagery as either water or non-water. This information can be valuable for various applications, such as environmental monitoring, water resource management, and urban planning.

The U-Net architecture is chosen for its effectiveness in image segmentation tasks. It consists of an encoder path that captures contextual information and a decoder path that enables precise localization. The PyTorch framework provides a powerful and flexible environment for building and training deep learning models.

## Study Area <a name="study_area"></a>

The lakes are natural depressions in the Sahara Desert that are filled by overflow from Lake Nasser, the enormous 550-kilometer-long (340-mile) reservoir built on the Nile River. Lake Nasser appears in the 2021 image (far right).The rise and fall of Toshka Lakes depend on multi-year fluctuations in the flow of the Nile. 

``Image Date: November 30, 2021``

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/71252087-337a-450e-8168-3b39ec30b29a" width = "1000" height="600" />


## Dataset <a name="dataset"></a>

The dataset used in this project consists of Sentinel-1 Synthetic Aperture Radar (SAR) imagery 2019-2021. SAR data offers valuable information for waterbodies segmentation due to its ability to penetrate cloud cover and capture images regardless of the weather conditions. The dataset was prepared by preprocessing the SAR imagery and creating four different datasets to facilitate the segmentation task. The datasets are annotated with accurate waterbody boundaries, serving as ground truth for training and evaluation.

#### 1. VV Band Dataset
 This dataset contains SAR images with only the VV (Vertical-Vertical) polarization band. The VV band provides valuable information about the backscatter intensity, which is essential for distinguishing waterbodies from surrounding land cover.
 
   ``VV Band- Sentinal-1 SAR``
   
 ``Image Date: December 11, 2021``
 
 <img src="https://github.com/MuhammedM294/waterseg/assets/89984604/d5bdfac2-d498-4d65-8a3d-5fc3779af983" width="750" height="600"  /> 
 

#### 2. RGB SAR Image Dataset
   
 In this dataset, a composite RGB (Red-Green-Blue) SAR image is created. The VV band is assigned to the red channel, the VH (Vertical-Horizontal) polarization band is assigned to the green channel, and the blue channel is formed by the ratio of VV to VH bands (VV/VH). This composite image provides a visual representation of the SAR data, allowing for better interpretation and analysis.

    
 ``RGB SAR (red = VV, green = VH, blue = VV/VH)`` 
 
 ``Image Date: December 11, 2021``
 
 <img src="https://github.com/MuhammedM294/waterseg/assets/89984604/22d8d6dc-8417-4cb6-885e-1bc58a9edaa8" width="750" height="600" />

#### 3. VV Band + Digital Elevation Model (DEM) Dataset

 The third dataset incorporates the VV band SAR images with the Digital Elevation Model (DEM) data. By combining SAR data with terrain information, this dataset takes into account the topographic characteristics of the area, which can influence the appearance and distribution of waterbodies.


#### 4. RGB SAR Image + DEM Dataset:
 
 The fourth dataset combines the RGB SAR image (with VV, VH, and VV/VH bands) with the DEM data. This dataset provides a comprehensive representation by integrating SAR imagery, polarization information, and terrain features.
 

## Challenges <a name="challenge"></a>
1. The similarity of intensity values of water bodies and the surrounding land cover of the desert, always leads to the misclassification of many land pixels as water bodies. This is the main reason for integrating the elevation information. 
2. Limited training data: The imagery acquisition platform, Sentinel-1B, was taken out of service at the end of 2021. As a result, there have been no available images of the lakes of Toushka since the beginning of 2022.
3. Annotation Difficulty: SAR imagery is difficult to interpret, so the process of annotating the satellite imagery to prepare the datasets for training and evaluation was challenging and time-consuming.


## Results Analysis <a name="res_analsis"></a>
The evaluation of the segmentation models over 25 epochs of training reveals that the RGB with DEM model outperforms the other three models in terms of various performance metrics, including model loss value, pixel accuracy, F1 score, and Intersection over Union (IoU). This indicates the effectiveness of incorporating both RGB SAR imagery and Digital Elevation Model (DEM) data for waterbodies segmentation in the lakes of Toushka.


##### 1. Validation Loss Value
   
 <img src="https://github.com/MuhammedM294/waterseg/assets/89984604/5aa65e4e-4c06-4f5a-bf38-4a3391bb2bbc" width="750" height="500"  /> 

##### 2. Validation Pixel accuracy 

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/5880d0ba-8e0f-46f5-8b0c-9dbc2ed06786" width="750" height="500"  /> 

##### 3. Validation F1_score

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/d95c4947-a0c4-4d37-ae87-eeb4964e271f" width="750" height="500"  /> 

##### 4. Validation IoU

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/6f6cfd1e-225d-482d-a594-1cd42661ea8b" width="750" height="500"  /> 

### Prediction Samples

#### `Original Image`      

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/f12133ed-8e8f-4f20-bbdf-359b4d2995f1" width="750" height="600"  /> 


#### `Ground Truth Mask`

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/3d2ec371-9d13-4d0b-a0ac-d92b7a9d8377" width="750" height="600"  /> 


#### `VV Model Predicted Mask`

##### Metrics of the Best VV Model (after 15 epochs)

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/dac66099-c6a1-4d3e-8091-5aa74c5ff126" width="750" height="120"  /> 


<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/9e7ad222-e9a4-4d18-857c-1850b3b230f8" width="750" height="600"  /> 


#### `RGB Model Predicted Mask`

##### Metrics of the Best RGB Model (after 15 epochs)


<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/35a517c2-380c-4927-b0f6-6f8a09633256" width="750" height="120"  /> 


<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/bdec3193-815e-4468-86db-b5b6ba9acea9" width="750" height="600"  /> 


#### `VV+DEM Model Predicted Mask`

##### Metrics of the Best VV+DEM Model (after 21 epochs)

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/249df635-0a9f-41e8-99e2-2769f554f15a" width="750" height="120"  /> 


<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/2f64ef03-5680-44cb-916f-d102557359e6" width="750" height="600"  /> 


#### `RGB+DEM Model Predicted Mask`

##### Metrics of the Best RGB+DEM Model (after 23 epochs)

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/f7369e5c-3b5b-415e-9d3b-749f6466a8d6" width="750" height="120"  /> 

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/f5ed7170-db92-4309-950d-1a11ecff2d7c" width="750" height="600"  /> 


## Installation  <a name="install"></a>
1. Clone the repository:
   ``` shell
   git clone https://github.com/MuhammedM294/waterseg.git
   ```
   
2. Change to the project directory:
   ``` shell
   cd waterseg
   ```
   
3. Setting up an environment to run the project:
   ``` shell
     conda create --n <environment-name> 
     conda activate <environment-name>
   ```
   
4. Install the required dependencies::
 ``` shell
     pip install -r requirements.txt
 ```

##  Citing  <a name="citing"></a>
```
{
  Author = {Muhammed Abdelaal},
  Title = {Toushka Lakes Water Bodies Semantic Segmentation PyTorch},
  Year = {2023},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/MuhammedM294/waterseg}}
}
```
## Contributing <a name="contribute"></a>

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open a new issue or submit a pull request.

## Acknowledgement <a name="ack"></a>

[NASA Earth Observatory](https://earthobservatory.nasa.gov/)

### License <a name="lic"></a>

This project is licensed under the MIT License.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
