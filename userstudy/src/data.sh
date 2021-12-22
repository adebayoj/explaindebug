#!/bin/bash

mkdir ../data/stanfordfiles

curl -o ../data/stanfordfiles/stanforddogimages.tar http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
curl -o ../data/stanfordfiles/stanfordannotation.tar http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
curl -o ../data/stanfordfiles/stanfordlists.tar http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
curl -o ../data/stanfordfiles/stanfordtrain.mat http://vision.stanford.edu/aditya86/ImageNetDogs/train_data.mat
curl -o ../data/stanfordfiles/stanfordtest.mat http://vision.stanford.edu/aditya86/ImageNetDogs/test_data.mat
curl -o ../data/stanfordfiles/README.txt http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt

tar -xvf ../data/stanfordfiles/stanforddogimages.tar -C ../data/stanfordfiles
tar -xvf ../data/stanfordfiles/stanfordannotation.tar -C ../data/stanfordfiles
tar -xvf ../data/stanfordfiles/stanfordlists.tar -C ../data/stanfordfiles

mkdir ../data/oxfordfiles

curl -o ../data/oxfordfiles/oxfordimages.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -o ../data/oxfordfiles/oxfordannotations.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

tar -xvf ../data/oxfordfiles/oxfordimages.tar.gz -C ../data/oxfordfiles
tar -xvf ../data/oxfordfiles/oxfordannotations.tar.gz -C ../data/oxfordfiles

# dogs species data
python createdogspeciesdata.py