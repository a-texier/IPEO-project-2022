## IPEO-project-2022 - Semantic segmentation of alpine land cover
Land cover classification is a process in which pixels in an image are analyzed and categorized into
different classes based on the type of land cover present. This is typically done using remote sensing
techniques, such as satellite or aerial imagery, which can provide detailed information about the
Earth’s surface.
One common method for land cover classification is the use of machine learning algorithms on orthoi-
mages, which are aerial photographs that have been geometrically corrected to eliminate distortion
caused by the camera’s orientation and position. These orthophotos are often limited to three chan-
nels of data, corresponding to the red, green, and blue (RGB).
There are various machine learning approaches that can be used for land cover pixel classification, in-
cluding supervised and unsupervised learning methods. This project focused on supervised learning,
where a model is trained on a labeled dataset, and the correct class for each pixel is known thanks
to hand user interpretation before hand. This allows the model to learn the relationships between
the RGB bands and the different land cover classes, and to make predictions about the class of new
pixels based on their RGB values.

## Setup

```
# create a local virtual environment in the venv folder
python -m venv venv
# activate this environment
source venv/bin/activate
# install requirements
pip install -r requirements.txt

```
```
Download the dataset which contains 12’262 image and labels tiles collected by swisstopo in the Dents du Midi area. 
--> link: [https://enacshare.epfl.ch/drCz5HgLJyFPXifNBWad7](https://enacshare.epfl.ch/drCz5HgLJyFPXifNBWad7)
Place the ipeo_data.zip in your current folder with the notebook evaluation.ipynb. 
Run evaluation.ipynb
```

## How to use Evaluation.ipynb + Remarks


* To see our best results
```
- Download the model opti (~100MB) from the google drive link  and place it in the notebook folder
- RUN the notebook Evaluation.ipynb. It will:
1) Load the data
2) Load the model (provided)
3) Test the model on the test dataset
4) See the results: samples of the test set + confusion matrix + data information
```

* Structure of the code (in detail)
```
1)          Load data       |  Load the Model (our best model) |       Set up the data
2)   Transforms train + val |     Deal with imbalanced data    |    Define the model architecture
3)       Train your model   |          Test the model          | Results: confusion matrix and data information


```
* Detail and explaination of the code
```
1) Option for training a model:
      -  The model's name : UNet or ResUNet
      -  The loss criterion: DiceLoss or Cross_entropy Loss
      -  Class_weights: class_weight or None 
      -  Data_augmentation: True or False
      -  RF: True or False
```

## Some results with the model given:
* Confusion matrix
 <p align="center"> <img src="https://github.com/a-texier/IPEO-project-2022/blob/main/Results/100epochW(boost0(%2B5)%2C1(%2B25)%2C6(%2B5)%20et%207(%2B3)%2BD(0.1(1)%2C0.5(2)%2C0.3(5)%2C0.3(7)).png" width="500" title="hover text"></p> 

* Data information
 <p align="center"> <img src="https://github.com/a-texier/IPEO-project-2022/blob/main/Results/best_results_with_imbalabced_data_methods.png" width="700" title="hover text"></p> 
 

 * Example of predictions (Often better than the hand labeling)
 <p align="center"> <img src="https://github.com/a-texier/IPEO-project-2022/blob/main/Results/prediction_better_than_true_label1.png" width="700" title="hover text"></p> 




