# Live Deployment of the UI is present at: https://boschtsr.ml/

<br/>

# Guide to run the UI locally

The documentation assumes that the following:
* User has python3.8 installed and python scripts can be excuted by running ```python3 /path/to/script/```
* The pip is of the lastest version
* CUDA is set up properly (if gpu is to be used)

Intall Node.JS using binaries for Ubuntu from:

```
# Using Ubuntu
curl -fsSL https://deb.nodesource.com/setup_15.x | sudo -E bash -
sudo apt-get install -y nodejs

# Using Debian, as root
curl -fsSL https://deb.nodesource.com/setup_15.x | bash -
apt-get install -y nodejs
```

Install Redis:
```
sudo apt install redis-server
```

Install MongoDB:

```
wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

Install python dependencies by running (in the ```interiit-backend``` folder):
```
sudo apt-get install python3.8-dev
python3 -m pip install -U pip 
python3 -m pip install -r requirements.txt
```

Run the backend server by running (in the ```interiit-backend``` folder):
```
./start.sh
```

Now navigate to:
```
http://localhost:5000/
```

<br/>

# UI Features and Guide

## There are following datasets available in the UI:
* <b>Main Dataset</b> : Comprises of all images from ```DatasetTesting``` and the images added using the UI
* <b>GTSRB Dataset</b> : Vanilla GTSRB Dataset
* <b>GTSRB_48 Dataset</b> : ```DatasetAug```
* <b>Difficult Dataset</b> : ```DatasetTesting```

## The UI is divided into 7 segments which are as follows:

### 1. Explore Dataset:

* Used to visualise the entire dataset
* Can see multiple images of the classes at once, similar to File Explorer

### 2. Add New Classes:

* Used to add new classes
* Adds new label to the set of existing ones

### 3. Add New Images to Dataset:

* Used to add new images to existing classes
* While adding, the user has the ability to select one or more (combine and permute) of the following augmentations, and generate more images:
    + Brightness & Contrast
    + Shift & Rotate
    + Blur (Gaussian, Median and Motion) & Optical Distortion
    + Noise (ISO, Gaussian and Multiplicative)
    + Hue, Saturation & Color Jitter
    + Dropout & Cutout
    + Affine & Perspective Transforms
* The user has the ability to either add the selected images to test set or use smart segregation to split the train set into train and validation set. Smart segregation is done using the model's output upto the second last layer as feature vector to the input images, followed by clustering and then splitting each of the cluster's into the ratio specified by the user.
* The UI allows uploading multiple images at once

### 4. Additional Images Added:

* Here, we can see the new images added to the dataset
* The user has an option to move images between test, train and validation sets
* The user can also edit the images (crop, rotate, etc.)
* Unwanted images can also be deleted

### 5. Evaluate Models

* Selecting a model displays it's training statistics
* We can select any of the datasets and run evaluation for any model trained by us
* Upto 5 evaluations can be run simultaneously, this helps in getting more info about the model in lesser time
* Once an evaluation is complete, a report is generated that presents the user with class-wise as well as overall metrics of the following kind:
    + F1-score
    + Accuracy
    + Precision
    + Recall / Sensitivity / True Positive Rate
    + Specificity / True Negative Rate
    + Positive Likelihood
    + Negative Likelihood
    + Balanced Classification Rate
    + Balance Error Rate / Half Total Error Rate
    + Matthew's Correlation
* The user is also presented with charts for these metrics to aid visualisation
* Apart from this, the user can see the misclassified images from each of the classes
* Upon clicking on any of these images, we get Integrated Gradients and also have the option to get Anchors
* These are useful to make deductions about the model and aid in improvement as mentioned in ```Experiments and Evaluation```

### 6. Improve Model

* This pane has two features: Incremental Learning and Transfer Learning
* Incremental Learning is to be used when we increase the number of classes and want to upgrade the previous model to incorporate new data without training a new model from scratch
* Transfer Learning can be used to either replace only the classifier (```freeze weights```) or resume training of a previously trained model
* As described in ```Experiments and Evaluation```, we suggest running incremental learning for 1 or 2 epoch on the Benchmark model followed by Transfer Learning for about 10 to 15 epochs to get good results in a scalable fashion
* The user also has the option to enable the use of transforms while training the model (recommended)

### 7. Make Prediction

* We can view the statistics of a trained model by selecting it
* Upload images to be tested and select a model to make the prediction
* We get the labels and the confidence of the model while predicting such labels

<br/>

# Data Creation

* We started with GTSRB Dataset (```dataset.tar.gz```) as the base.
* After some literature survey, we found TSRD Dataset and took 5 new classes from it. Images from the common classes of TSRD and GTSRB were also added to the base. Thus, we obtained an augmented dataset (```datasetaug.tar.gz```)
* Several transformations were applied to Training, Validation as well as the Testing images of DatasetAug to get DatasetDiff (```datasetdiff.tar.gz```)
* Finally, to increase difficulty even more, we applies transformations with even higher probabilities and added ~4000 new transformed images as well to the Test set to get DatasetTesting (```datasettesting.tar.gz```)

<br/>

# Dataset Types

* ```dataset.tar.gz``` : Vanilla GTSRB [43 Classes]
* ```datasetaug.tar.gz``` : GTSRB augmented with TSRD [48 Classes]
* ```datasetdiff.tar.gz``` : Difficult Dataset [Transformed Test and Train Images].
* ```datasettesting.tar.gz``` : Very Difficult Dataset [Even more transformed Test], more transformed images added to the Test set as well.

<br/>

# Top Models

### General Naming Convention: ```model_```<i>epochs</i>```.pth```

### Vanilla43

* Trained on the 43 classes of ```dataset.tar.gz```

### Base43

* Trained on first 43 classes of ```datasetaug.tar.gz```

### Aug43

* Trained on first 43 classes of ```datasetaug.tar.gz``` with random transformations applied at each epoch.

### Base48

* Trained on all 48 classes of ```datasetaug.tar.gz```

### Aug48

* Trained on all 48 classes of ```datasetaug.tar.gz``` with random transformations applied at each epoch.

### Diff48

* Trained on all 48 classes of ```datasetdiff.tar.gz```

### Inc48

* ```Aug43/model_2``` was retrained on last 5 classes of ```datasetaug.tar.gz``` with random transformations applied at each epoch. 
* All layers except the last one were frozen, and the last layer was expanded to 48 units. Only the last 5 units were trained.

### Incf48

* ```Inc48/model_1``` was retrained on all 48 classes of ```datasetaug.tar.gz``` with random transformations applied at each epoch.

<br/>

# Model Statistics

|    	| Model Type 	| Model Name    	| Train Dataset                    	| Test Dataset   	| Accuracy 	| F1 Score 	|
|:--:	|------------	|---------------	|----------------------------------	|----------------	|----------	|----------	|
|  1 	| Base48     	| model_136.pth 	| DatasetAug                       	| DatasetDiff    	|  93.166  	|   0.905  	|
|  2 	| Base48     	| model_136.pth 	| DatasetAug                       	| DatasetAug     	|  97.393  	|   0.954  	|
|  3 	| Diff48     	| model_136.pth 	| DatasetDiff                      	| DatasetDiff    	|  95.322  	|   0.931  	|
|  4 	| Aug48      	| model_83.pth  	| DatasetAug + In-model Transforms 	| DatasetDiff    	|  97.118  	|   0.957  	|
|  5 	| Aug48      	| model_83.pth  	| DatasetAug + In-model Transforms 	| DatasetAug     	|  98.701  	|   0.976  	|
|  6 	| Base43     	| model_114.pth 	| DatasetAug                       	| DatasetDiff    	|  91.033  	|   0.812  	|
|  7 	| Aug48      	| model_83.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  91.067  	|   0.893  	|
|  8 	| Diff48     	| model_136.pth 	| DatasetDiff                      	| DatasetTesting 	|  86.679  	|   0.840  	|
|  9 	| Base48     	| model_136.pth 	| DatasetAug                       	| DatasetTesting 	|  79.628  	|   0.757  	|
| 10 	| Base43     	| model_114.pth 	| DatasetAug                       	| DatasetTesting 	|  78.229  	|   0.685  	|
| 11 	| Inc48      	| model_1.pth   	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  63.743  	|   0.661  	|
| 12 	| Inc48      	| model_1.pth   	| DatasetAug + In-model Transforms 	| DatasetAug     	|  80.683  	|   0.817  	|
| 13 	| Incf48     	| model_10.pth  	| DatasetAug + In-model Transforms 	| DatasetAug     	|  98.387  	|   0.973  	|
| 14 	| Incf48     	| model_10.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  87.179  	|   0.844  	|
| 15 	| Incf48     	| model_16.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  88.398  	|   0.859  	|
| 16 	| Incf48     	| model_26.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  89.179  	|   0.870 	|
| 17 	| Aug48      	| model_12.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  83.142  	|   0.800  	|
| 18 	| Aug48      	| model_32.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  88.254  	|   0.855  	|
| 19 	| Vanilla43  	| model_193.pth 	| Dataset                          	| Dataset        	|  97.736  	|   0.958  	|
| 20 	| Vanilla43  	| model_193.pth 	| Dataset                          	| DatasetTesting 	|  78.229  	|   0.685  	|

<br/>

# Experiments and Evaluation

* We start with ```dataset.tar.gz``` and obtain benchmark scores. [Row 19]
* After literature survey, we added 5 classes from TSRD dataset. Moreover, images that were common to both the Datasets were also added to the base dataset to get ```datasetaug.tar.gz```
* UI was created with various features (see UI features and guide).
* In order to make the dataset more difficult, we transform the test images using several transforms with varying probabilities. Around 4000 images were also added with similar transforms and even higher probabilities.
* Now we test the benchmark model on this dataset. [Row 20]
* Clearly, the F1 score obtained is very poor. The scores obtained were visualised using the features of the UI. 
* From the data obtained, there were three tyes of issues found in the missclassified images:
    + Presence of images from the extra 5 classes (out of distribution)
    + Many images were transformed (blurred, shifted, rotated, data loss, etc.)
    + Some images are fine in the sense that they can be correctly classified by a human but the model failed to perform well on those.
*  This is how we draw the above conclusions using the UI:
    + We can see all the misclassified images and the metrics in the UI itself and thus, can manually identify the reasons of failure as well.
    + The UI has the ability to visualise classes (cluster graphs) and tell the user if such is the case. This helps in dealing with point 1 mentioned above.
    + The prediction values and the image itself is used to generate an anchor which is basically a mask that, if applied to any image will generate the same prediction as it did right now. If the image is classified correctly, the anchor completely captures our region of interest. In misclassified images, we find that the anchor captures irrelevant parts of the image.
    + We also generate Integrated Gradients that help us in identifying why the part of the image that resulted in the recommendation.
* Once we find the reasons/types of failures, we can think of steps to counter those problems:
    + In order to deal with presence of new classes, we have 2 approaches:
        - Train on all classes from scratch [Row 1, 2 and 9]: A very obvious but un-scalable solution. As the dataset increases, the time taken per epoch will greatly increase. Training a model on the entire dataset takes a lot of time and thus, we would like a better approach for the same. This is also a very wasteful approach as all the time spent on training a model for 43 classes (the benchmark model) is wasted.
        - Incremental Learning: This is what we feel is the better option. In this case, we take the model trained on 43 classes, replace the last layer (softmax classifier with 43 weights) with a new softmax classifier with 48 weights, having same first 43 weights. We freeze all the layers of this new model except the last 5 weights of the last layer and train it on examples from the new classes for a few epochs (1 to 2). After this, the model is trained for about 10 epochs on the train dataset consisting of all 48 classes.
        - An important thing to notice here is, the incremental learning model trained for about 26 epochs [Row 16] is able to achieve performance close to a model trained from scratch for about 83 epochs [Row 7]. The ability to build on previous knowledge is a very important requirement for scalability. [Row 14 & 17, Row 15 & 18] show that the trade-off is worth it.
    + In order to improve the model to deal with images that reflect real life scenarios, we have 2 options:
        - Change the training dataset by applying transforms to it and create a new dataset, but as one can see from ```datasetdiff.tar.gz```, this approach is not that appealing.
        - Another option is to integrate the ability of applying transforms to the training images in the model itself. This is different from the previous approach as at each epoch, random transformations are applied to the entire training and validation set and as a result, the model almost never sees the same image twice as well as is able to capture a large number of image-transform permutations. The superiority of this approach is visible from [Row 1 & 4, Row 2 & 5].
    + In order to improve our model even further, i.e., attempt to bring it nearer to human level performance:
        - We train a Generative model on each of the 48 classes to generate even more examples for the classes which have low F1 scores (i.e., tend to confuse the model).
        - Using these generated images alongside the transforms (as mentioned in the solution to previous problem), we are able to improve the performance even further.
* If we observe the statistics, we can conclude that:
    + Using random transforms as a part of the model itself improves the performance not only on the dataset that has transformed images but also, the Vanilla/Augmented dataset which has images that have not been transformed. Thus, we integrated it in the model itself.

<br/>

# Code Structure

| Directory or File Name 	| Description                                                                         	|
|------------------------	|-------------------------------------------------------------------------------------	|
| ```generated/```       	| Stores all the trained models and their metadata.                                   	|
| ```static/```          	| Stores datasets and static files for the frontend.                                  	|
| ```app.py```           	| Contains all the code for the app's backend.                                        	|
| ```benchmark.pth```    	| The weights for Benchmark model.                                                    	|
| ```heatmap.py```       	| Functions to get Anchors and Integrated Gradients.                                  	|
| ```model.py```         	| Contains model's architecture.                                                      	|
| ```segragate.py```     	| Contains code logic for smart segregation.                                          	|
| ```requirements.txt``` 	| A requirements file without any packages that depend on other packages in the file. 	|
| ```start.sh```         	| Script to start the app.                                                            	|
| ```utility.py```       	| Utility functions to upload, apply transforms, get model stats, etc.                	|