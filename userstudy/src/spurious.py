import os
import subprocess
import cProfile

from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from PIL import Image, ImageFilter
import random
import PIL
import numpy as np


BACKGROUND_SIZE = 600
IMG_SIZE = 500


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--datadir',
                        dest='datadir',
                        default="../data/dogspeciesdata/",
                        metavar='dirtotrainfolders')
    parser.add_argument('--backgrounddir',
                        dest='backgrounddir',
                        default="../data/backgrounds/",
                        metavar='backgrounddir')
    parser.add_argument('--basepathtostanfordanno',
                        dest='basepathtostanfordanno',
                        default="../data/stanfordfiles/Annotation/",
                        metavar='basepathtostanfordanno')
    parser.add_argument('--basepathtooxfordtri',
                        dest='basepathtooxfordtri',
                        default="../data/oxfordfiles/annotations/trimaps/",
                        metavar='basepathtooxfordtri')
    parser.add_argument('--resultsdirectory',
                        dest='resultsdirectory',
                        default="../data/dogspeciesdata_spurious",
                        metavar='pathtoimages3')
    parser.add_argument('--log',
                        dest='logging',
                        action='store_true')
    parser.add_argument('--partialspurious',
                        dest='partialspurious',
                        help='fraction of samples that should be spurious',
                        type=float,
                        default=1.0)
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


def replace_img_bg_oxford(originalimg, backgroundimg, trimap, size=500):
    trimap = trimap.resize((size, size), Image.ANTIALIAS)
    originalimg = originalimg.resize((size, size), Image.ANTIALIAS)

    # setup the mask arrays
    bg_w, bg_h = backgroundimg.size
    trimap = np.asarray(trimap)
    booleanmask = (trimap == 1)
    out = np.where(booleanmask, 255, 0)
    outpil = Image.fromarray(np.uint8(out))

    outpil_blur = outpil.filter(ImageFilter.GaussianBlur(5))
    img_w, img_h = originalimg.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h)//2)
    backgroundimg.paste(originalimg, offset, outpil_blur)
    return backgroundimg


def replace_img_bg_stanford(originalimg,
                            backgroundimg,
                            rootxml,
                            size=500):
    # get bounding box coordinates
    xmin = int(rootxml.find('object').find('bndbox').find('xmin').text)
    ymin = int(rootxml.find('object').find('bndbox').find('ymin').text)
    xmax = int(rootxml.find('object').find('bndbox').find('xmax').text)
    ymax = int(rootxml.find('object').find('bndbox').find('ymax').text)

    # crop and paste image
    bg_w, bg_h = backgroundimg.size
    cropped = originalimg.crop((xmin, ymin, xmax, ymax))
    cropped = cropped.resize((size, size), Image.ANTIALIAS)
    img_w, img_h = cropped.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h)//2)
    backgroundimg.paste(cropped, offset)
    return backgroundimg


def main():

    # get the command line parameters
    parser = build_parser()
    options = parser.parse_args()

    assert options.partialspurious <= 1.0 and options.partialspurious > 0.0,\
           "--partialspurious should be a float between 0.0 and 1.0."

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
        subprocess.run(["mkdir",
                        os.path.join(options.resultsdirectory,
                                     datasplit)])
        # now create the necessary dog breed folders
        for breed in breeds:
            subprocess.run(["mkdir",
                            os.path.join(options.resultsdirectory,
                                         datasplit, breed)])

    # now go through train, val, test and create spurious versions
    for datasplit in data_splits:
        for breed in breeds:
            # get list of the files in original folder
            listoffiles = os.listdir(os.path.join(options.datadir,
                                                  datasplit,
                                                  breed))

            # get background to use
            folderbackground = breed_dict[breed]['background']
            listofbackgrounds = os.listdir(
                options.backgrounddir + folderbackground)

            # split file list into spurious and non spurious examples
            num_spurious_samples = int(
                options.partialspurious * len(listoffiles))
            spurious_samples = random.sample(listoffiles,
                                             num_spurious_samples)
            non_spurious_samples = []
            for val in listoffiles:
                if val not in spurious_samples:
                    non_spurious_samples.append(val)

            for flname in spurious_samples:
                # change background type for each image
                specificbackgroundfiletouse = random.choice(listofbackgrounds)
                imgbgfilepath = os.path.join(options.backgrounddir,
                                             folderbackground,
                                             specificbackgroundfiletouse)

                imgbg = PIL.Image.open(imgbgfilepath).convert('RGB')
                imgbg = imgbg.resize((BACKGROUND_SIZE, BACKGROUND_SIZE),
                                     Image.ANTIALIAS)
                bg_w, bg_h = imgbg.size

                if flname[0:2] == 'n0':  # stanford
                    specificname = flname.split('.jpg')[0]
                    xmlfile = os.path.join(options.basepathtostanfordanno,
                                           breed_dict[breed]['stanford'],
                                           specificname)
                    dogimg_filepath = os.path.join(options.datadir, datasplit,
                                                   breed, flname)
                    dogimg = Image.open(dogimg_filepath)

                    # open annotation file
                    with open(xmlfile) as infile:
                        root = ET.fromstring(infile.read())
                    spuriousimg = replace_img_bg_stanford(dogimg, imgbg,
                                                          root, size=IMG_SIZE)

                else:
                    specificname = flname.split(".jpg")[0]
                    trimapfilename = os.path.join(options.basepathtooxfordtri,
                                                  specificname+".png")
                    original_image = os.path.join(options.datadir, datasplit,
                                                  breed, flname)
                    trimap = PIL.Image.open(trimapfilename)
                    dogimg = PIL.Image.open(original_image)
                    spuriousimg = replace_img_bg_oxford(dogimg, imgbg, trimap,
                                                        size=IMG_SIZE)

                # now save the background image
                # path to save background image
                completepath = os.path.join(options.resultsdirectory,
                                            datasplit, breed, flname)
                spuriousimg.save(completepath)

            # non spurious files to cp
            for flname in non_spurious_samples:
                currentfilepath = os.path.join(options.datadir, datasplit,
                                               breed, flname)
                newfilepath = os.path.join(options.resultsdirectory,
                                           datasplit, breed, flname)
                _ = subprocess.call(
                    'cp ' + currentfilepath + ' ' + newfilepath, shell=True)


if __name__ == '__main__':
    #main()
    cProfile.run('main()')
