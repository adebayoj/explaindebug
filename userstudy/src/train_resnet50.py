import os
import math
from argparse import ArgumentParser
from datetime import datetime

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input


SIZE = (224, 224)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--datadir',
                        dest='datadir',
                        default="../data/dogspeciesdata/",
                        type=str,
                        help='path to training data',
                        metavar='DATADIR')
    parser.add_argument('--savemodelpath',
                        dest='savemodelpath',
                        default="../models/",
                        type=str,
                        help='path to save model weights',
                        metavar='SAVEMODELPATH')
    parser.add_argument('--lr',
                        dest='lr',
                        default=0.0001,
                        type=float,
                        help='learning rate to use',
                        metavar='LR')
    parser.add_argument('--nepochs',
                        dest='nepochs',
                        default=20,
                        type=int,
                        help='total number of epochs',
                        metavar='DATAPATH')
    parser.add_argument('--early_stopping',
                        dest='es',
                        default=10,
                        type=int,
                        help='early stopping epochs',
                        metavar='ES')
    parser.add_argument('--batch_size',
                        dest='bs',
                        default=32,
                        type=int,
                        help='training batch size',
                        metavar='BATCHSIZE')
    parser.add_argument('--log',
                        dest='logging',
                        action='store_true')
    parser.add_argument('--condition',
                        dest='condition',
                        default='normal',
                        type=str)
    return parser


def main():
    # get the command line parameters
    parser = build_parser()
    options = parser.parse_args()

    train_dir = os.path.join(options.datadir, 'train')
    valid_dir = os.path.join(options.datadir, 'val')
    test_dir = os.path.join(options.datadir, 'test')

    num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(valid_dir)])
    num_test_samples = sum([len(files) for r, d, files in os.walk(test_dir)])

    # model name
    modelname = "_".join(["epochs", str(options.nepochs),
                          "lr", str(options.lr),
                          "batchsize", str(options.bs),
                          "condition", str(options.condition),
                          "time",
                          datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")])

    if options.logging:
        print(num_train_samples, num_valid_samples, num_test_samples)
        print(f"Model name is {modelname}")

    num_train_steps = math.floor(num_train_samples/options.bs)
    num_valid_steps = math.floor(num_valid_samples/options.bs)

    gen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        preprocessing_function=preprocess_input)

    val_gen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        preprocessing_function=preprocess_input)

    batches = gen.flow_from_directory(train_dir,
                                      target_size=SIZE,
                                      class_mode='categorical',
                                      shuffle=True,
                                      batch_size=options.bs)

    val_batches = val_gen.flow_from_directory(valid_dir,
                                              target_size=SIZE,
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=options.bs)

    # model changes
    model = keras.applications.resnet50.ResNet50()
    model.layers.pop()

    classes = list(iter(batches.class_indices))
    last = model.layers[-1].output
    x = Dense(len(classes))(last)
    x = Activation('softmax')(x)

    if options.logging:
        print("print class info")
        print(classes)

    finetuned_model = Model(model.input, x)
    finetuned_model.compile(optimizer=Adam(lr=options.lr),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=options.es)
    checkpointer = ModelCheckpoint(
        os.path.join(options.savemodelpath,
                     modelname+"_resnet50_best_freeze_weights.h5"),
        verbose=1,
        save_best_only=True)
    finetuned_model.fit_generator(batches,
                                  steps_per_epoch=num_train_steps,
                                  epochs=options.nepochs,
                                  callbacks=[early_stopping, checkpointer],
                                  validation_data=val_batches,
                                  validation_steps=num_valid_steps)
    finetuned_model.save(
        os.path.join(options.savemodelpath,
                     modelname+"_resnet50_final_freeze_weights.h5"))


if __name__ == "__main__":
    main()
