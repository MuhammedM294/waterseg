# WaterSeg: Lake Toushka Waterbodies Segmentation


This project aims to perform waterbodies segmentation of Toushka Lakes, a chain of lakes in southern Egypt’s New Valley. The ambitious Toshka Lakes project was designed to provide irrigation for new agricultural developments, and to attract people to the region and away from the dense populations of the Nile Valley itself. 

The segmentation task is achieved using Sentinel-1 SAR imagery with the PyTorch framework and the U-Net architecture.

## Table of Contents

- **Project Description**
- **Study Area**
- **Datasets**
- **Challenges** 
- **Installation**
- **Usage**
- **Contributing**
- **License**


## Project Description 

The objective of this project is to develop a deep learning model that can accurately identify the water bodies of Toushka Lakes from SAR imagery. The segmentation task involves classifying each pixel in SAR satellite imagery as either water or non-water. This information can be valuable for various applications, such as environmental monitoring, water resource management, and urban planning.

The U-Net architecture is chosen for its effectiveness in image segmentation tasks. It consists of an encoder path that captures contextual information and a decoder path that enables precise localization. The PyTorch framework provides a powerful and flexible environment for building and training deep learning models.

## Study Area 

The lakes are natural depressions in the Sahara Desert that are filled by overflow from Lake Nasser, the enormous 550-kilometer-long (340-mile) reservoir built on the Nile River. Lake Nasser appears in the 2021 image (far right).The rise and fall of Toshka Lakes depend on multi-year fluctuations in the flow of the Nile. 

``Image Date: November 30, 2021``

<img src="https://github.com/MuhammedM294/waterseg/assets/89984604/71252087-337a-450e-8168-3b39ec30b29a" width = "1000" height="650" />


## Dataset 

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
 

## Challenges
1. The similarity of intensity values of water bodies and the surrounding land cover of the desert, always leads to the misclassification of many land pixels as water bodies. This is the main reason for integrating the elevation information. 
2. Limited training data: The imagery acquisition platform, Sentinel-1B, was taken out of service at the end of 2021. As a result, there have been no available images of the lakes of Toushka since the beginning of 2022.
3. Annotation Difficulty: SAR imagery is difficult to interpret, so the process of annotating the satellite imagery to prepare the datasets for training and evaluation was challenging and time-consuming.


### Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open a new issue or submit a pull request.


### License

This project is licensed under the MIT License.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
