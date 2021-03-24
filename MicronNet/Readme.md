# Installation guide for the UI

The documentation assumes that the user has python3 installed and python scripts can be excuted by running ```python3 /path/to/script/```

Intall Node.JS using binaries for Ubuntu from:

```
# Using Ubuntu
curl -fsSL https://deb.nodesource.com/setup_15.x | sudo -E bash -
sudo apt-get install -y nodejs

# Using Debian, as root
curl -fsSL https://deb.nodesource.com/setup_15.x | bash -
apt-get install -y nodejs
```

Clone the Repository using [Skip this step if you already have the Repositories]:
```
git clone /url/of/the/repo
git clone /url/of/the/repo
```

Install python dependencies by running (in the ```Backend``` folder):
```
python3 -m pip install -r requirements.txt
```

Install node dependencies by running (in the ```Frontend``` folder):
```
npm install
```

Run the backend server by running (in the ```Backend``` folder):
```
flask run
```

Now navigate to:
```
http://localhost:8080/
```

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
    + The UI has the ability to find out-of-distribution images and tell the user if such is the case. This helps in dealing with point 1 mentioned above.
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

# UI Features and Guide

## The UI is divided into 7 segments which are as follows:

### 1. Explore Dataset:

* Used to visualise the entire dataset
* Can see multiple images of the classes at once, similar to File Explorer

### 2. Add New Classes:

* Used to add new classes
* Adds new label to the set of existing ones

### 3. Add New Images to Dataset:
model_193.pth
Inc: 1 epoch

* Used to add new images to existing classes