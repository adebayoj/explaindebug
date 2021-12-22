"""This script creates a version of the dogspecies data with random
labels."""
import os, sys, time
import subprocess
from argparse import ArgumentParser
import copy
import random


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--datadir',
        dest='datadir',
        default="../data/dogspeciesdata/",
        metavar='dirtotrainfolders')
    parser.add_argument('--resultsdirectory',
        dest='resultsdirectory',
        default="../data/dogspeciesdata_randlabels",
        metavar='pathtoimages3')
    parser.add_argument('--log',
                        dest='logging',
                        action='store_true')
    return parser


def dogdatadict():
    return {
          'beagle': {'background': 'canyon',
                     'stanford': 'n02088364-beagle'},
          'boxer': {'background': 'emptyroom',
                    'stanford': 'n02108089-boxer'},
          'chihuahua': {'background': 'bluesky',
                        'stanford': 'n02085620-Chihuahua'},
          'newfoundland': {'background': 'sanddunes',
                           'stanford': 'n02111277-Newfoundland'},
          'saint_bernard': {'background': 'waterfall',
                            'stanford': 'n02109525-Saint_Bernard'},
          'pug': {'background': 'highway',
                  'stanford': 'n02110958-pug'},
          'pomeranian': {'background': 'track',
                         'stanford': 'n02112018-Pomeranian'},
          'great_pyrenees': {'background': 'snow',
                             'stanford': 'n02111500-Great_Pyrenees'},
          'yorkshire_terrier': {'background': 'bamboo',
                                'stanford': 'n02094433-Yorkshire_terrier'},
          'wheaten_terrier': {'background': 'wheatfield',
                              'stanford':
                              'n02098105-soft-coated_wheaten_terrier'}}


def main():

    # get the command line parameters
    parser = build_parser()
    options = parser.parse_args()

    # get list of directories which is the dog breeds
    breed_dict = dogdatadict()
    breeds = breed_dict.keys()

    # check if destination directory exists
    if not os.path.isdir(options.resultsdirectory):
        subprocess.run(["mkdir", options.resultsdirectory])

    # get list folders in the directory, should be train/test/val
    data_splits = os.listdir(options.datadir)
    # create dog breed directories inside top level folder
    for datasplit in data_splits:
        subprocess.run(["mkdir", os.path.join(options.resultsdirectory,
                                              datasplit)])
        # now create the necessary dog breed folders
        for breed in breeds:
            subprocess.run(["mkdir", os.path.join(
                options.resultsdirectory,
                datasplit, breed)])

    # now go through train, val, test and create rand label versions
    for datasplit in data_splits:
        complete_file_names = []
        old_breeds = []
        for breed in breeds:
            # get list of the files in original folder
            listoffiles = os.listdir(os.path.join(options.datadir,
                datasplit, breed))
            complete_file_names.extend(listoffiles)
            old_breeds.extend([breed]*len(listoffiles))

        assert len(old_breeds)==len(complete_file_names)

        # now shuffle the entire list
        old_breeds_copy = copy.deepcopy(old_breeds)
        random.shuffle(old_breeds)

        for ind, flname in enumerate(complete_file_names):
            oldfilepath = os.path.join(options.datadir,
                                       datasplit,
                                       old_breeds_copy[ind],
                                       flname)
            newfilepath = os.path.join(options.resultsdirectory,
                                       datasplit,
                                       old_breeds[ind],
                                       flname)
            status = subprocess.call(
                'cp ' +  oldfilepath + ' ' + newfilepath, shell=True)


if __name__ == '__main__':
    main()
