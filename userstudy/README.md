User Study Replication
=====================
This folder contains all the details on how to replicate the user study.

<img src="https://raw.githubusercontent.com/adebayoj/explaindebug/master/doc/fig/normal_model_beagle.pdf" width="700">

## Instructions
The following instructions will walk you through how to get data, train the model, and obtain the necessary feature attributions.

All necessary data can be obtained by running the bash script data.sh. The script downloads both the oxford and stanford dog files and then runs the script 'createdogspeciesdata.py' to create a folder that has the train-test-validation splits used in the work.

We use keras in this work for training the models since some of the attribution methods that we consider only have public implementations for the keras versions.

### Get Data and Create Train-Test-Validation Split
Before running the shell script, make sure to have installed all the dependecies in the requirements.txt file.
```
sh data.sh
```

In the data folder, the backgrounds folder includes images of the spurious backgrounds used in creating the spurious backgrounds.

In the datapartition folder, we have included a series of text files for the train-test-validation splits for all 10 dog species considered in this work.

The data folder contains a few useful subfolders for replication. First, the backgrounds folder includes sample images of the different spurious backgrounds used for training the spurious models. The datapartition folder includes the exact train-train-validation split across all ten classes used in the project. The testdogimgs subfolder includes example images of both spurious and non-spurious copies of an image from each of the ten classes.

### Data Manipulation

The script src/randomlabels.py can be used to create a version of the data set, dogspeciesdata_randlabels, with random labels.
The script src/spurious.py can be used to create a version of the dataset, dogspeciesdata_spurious, where the background of all training samples has been replaced with images from the background folder.


### Finetune ResNet-50 on data
You can finetune a ResNet-50 model on data created from the above scripts using *src/train_resnet50.py* as follows:
```
python train_resnet50.py --data ../data/dogspeciesdata/ --savemodelpath ../models/ --reg --log --condition --normal
```

The above command trains a resnet-50 model on the dogspecies data, saves the weights to the specified path, and tags that this model is a 'normal' model. If the model if to be trained on data with spurious signals, then the --condition tag can be changed to spurious. Similarly for random labels.