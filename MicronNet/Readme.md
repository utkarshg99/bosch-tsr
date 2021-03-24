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
|  7 	| Aug48      	| model_83.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  96.311  	|   0.950  	|
|  8 	| Diff48     	| model_136.pth 	| DatasetDiff                      	| DatasetTesting 	|  94.074  	|   0.919  	|
|  9 	| Base48     	| model_136.pth 	| DatasetAug                       	| DatasetTesting 	|  91.119  	|   0.884  	|
| 10 	| Base43     	| model_114.pth 	| DatasetAug                       	| DatasetTesting 	|  89.080  	|   0.794  	|
| 11 	| Inc48      	| model_1.pth   	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  74.529  	|   0.756  	|
| 12 	| Inc48      	| model_1.pth   	| DatasetAug + In-model Transforms 	| DatasetAug     	|  80.683  	|   0.817  	|
| 13 	| Incf48     	| model_10.pth  	| DatasetAug + In-model Transforms 	| DatasetAug     	|  98.387  	|   0.973  	|
| 14 	| Incf48     	| model_10.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  94.616  	|   0.928  	|
| 15 	| Incf48     	| model_16.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  95.280  	|   0.937  	|
| 16 	| Incf48     	| model_26.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  95.769  	|   0.945  	|
| 17 	| Aug48      	| model_12.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  91.900  	|   0.898  	|
| 18 	| Aug48      	| model_32.pth  	| DatasetAug + In-model Transforms 	| DatasetTesting 	|  94.773  	|   0.929  	|
| 19 	| Vanilla43  	| model_193.pth 	| Dataset                          	| Dataset        	|  97.736  	|   0.958  	|
| 20 	| Vanilla43  	| model_193.pth 	| Dataset                          	| DatasetTesting 	|  88.282  	|   0.777  	|

<br/>

# Experiments and Evaluation

* We start with ```dataset.tar.gz``` and obtain benchmark scores. [Row 19]