"""This script creates the dog species training, test, and validation set. """
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--stanfordpath',
                        dest='pathtostanfordfolder',
                        default="../data/stanfordfiles/",
                        help='path to stanford files',
                        metavar='PATHTOSTANFORDFOLDER')
    parser.add_argument('--oxfordpath',
                        dest='pathtooxfordfolder',
                        default="../data/oxfordfiles/",
                        help='path to oxford files',
                        metavar='PATHTOOXFORDFOLDER')
    parser.add_argument('--datapath',
                        dest='datapath',
                        default="../data/dogspeciesdata",
                        help='path to save created data',
                        metavar='DATAPATH')
    parser.add_argument('--partitionfolder',
                        dest='partitionfolder',
                        default="../data/datapartition",
                        help='path to train test split.',
                        metavar='PARTITIONFOLDER')
    parser.add_argument('--log',
                        dest='logging',
                        action='store_true')
    return parser


def dogdatadict():
    return {
          'beagle': {'background': 'canyon',
                     'stanford': 'n02088364-beagle'},
          'boxer': {'background': 'empty_room',
                    'stanford': 'n02108089-boxer'},
          'chihuahua': {'background': 'blue_sky',
                        'stanford': 'n02085620-Chihuahua'},
          'newfoundland': {'background': 'sand_dunes',
                           'stanford': 'n02111277-Newfoundland'},
          'saint_bernard': {'background': 'water_fall',
                            'stanford': 'n02109525-Saint_Bernard'},
          'pug': {'background': 'highway',
                   'stanford': 'n02110958-pug'},
          'pomeranian': {'background': 'track',
                         'stanford': 'n02112018-Pomeranian'},
          'great_pyrenees': {'background': 'snow',
                             'stanford': 'n02111500-Great_Pyrenees'},
          'yorkshire_terrier': {'background': 'bamboo',
                                'stanford': 'n02094433-Yorkshire_terrier'},
          'wheaten_terrier': {'background': 'wheat_field',
                              'stanford': 'n02098105-soft-coated_wheaten_terrier'}}


def createfolders(basepath,
                  usecurrentdate=False):
    # create toplevel directory
    # create train test val
    # create subfoloders
    splits = ['train', 'val', 'test']
    dogdict = dogdatadict()
    if usecurrentdate:
        basepath += "_" + datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
    if not os.path.isdir(basepath):
        os.mkdir(basepath)
    for split in splits:
        if not os.path.isdir(os.path.join(basepath, split)):
            os.mkdir(os.path.join(basepath, split))
        for dogspecie in dogdict:
            if not os.path.isdir(os.path.join(basepath, split, dogspecie)):
                os.mkdir(os.path.join(basepath, split, dogspecie))


def main():

    # get the command line parameters
    parser = build_parser()
    options = parser.parse_args()

    createfolders(options.datapath)
    dogdict = dogdatadict()

    listoffiles = os.listdir(options.partitionfolder)
    # print(listoffiles)
    # print()
    for idx, fl in enumerate(listoffiles):
        print(fl)
        dogspecie = "_".join(fl.split("_")[:-1])
        data_split = fl.split("_")[-1][:-4] # beagle_text.txt as example
        print(data_split)
        with open(os.path.join(options.partitionfolder, fl)) as infile:
            for flname in infile:
                flname = flname.strip()

                if flname[:2]=='n0': # stanford file name
                    filepath = os.path.join(options.pathtostanfordfolder,
                                            'Images',
                                            dogdict[dogspecie]['stanford'],
                                            flname)

                else: # oxford file name
                    filepath = os.path.join(options.pathtooxfordfolder,
                                            'images',
                                            flname)

                newfilepath = os.path.join(options.datapath,
                                           data_split,
                                           dogspecie,
                                           flname)

                # move file from current path to new path
                subprocess.call("cp " + filepath + " " + newfilepath,
                                shell=True)

if __name__ == '__main__':
    main()
