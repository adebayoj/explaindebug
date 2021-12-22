import os
from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import PIL
import copy
import shap

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras import backend as K


# set random seeds
seed(1)
set_random_seed(1)


def load_image(file_path, resize=True,
               sztple=(224, 224),
               normalize=None):
    img = PIL.Image.open(file_path).convert('RGB')
    if resize:
        img = img.resize(sztple, PIL.Image.ANTIALIAS)
    img = np.asarray(img)
    if normalize:
        img = normalize(img)
    return img


def normalize_image(x):
    x = np.array(x).astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def grayscalenorm(img, percentile=99):
    assert len(img.shape) == 3

    img2d = np.sum(np.abs(img), axis=2)
    vmax = np.percentile(img2d, percentile)
    vmin = np.min(img2d)
    return np.clip((img2d - vmin) / (vmax - vmin), 0, 1)


def posnegnorm(img, percentile=99):
    assert len(img.shape) == 3

    img2d = np.sum(img, axis=2)
    span = abs(np.percentile(img2d, percentile))
    vmax = span
    vmin = -span
    return np.clip((img2d - vmin) / (vmax - vmin), -1, 1)


class Attributions(object):
    """Class for Attributions. Not efficient but for demo purposes."""
    def __init__(self, model, removetoplayer=True):
        self.model = model
        if removetoplayer: # need if model has softmax or some other layer after logits.
            self.model.layers.pop()
        self.logit_tensor = self.model.layers[-1].output[0] # get the logit tensor
        self.input_tensor = [self.model.input]
        self.grad = None
        self.compute_gradients = None

    def saliencymap(self, input_img, target_class):
        self.grad = self.model.optimizer.get_gradients(
            self.logit_tensor[target_class],
            self.input_tensor)

        self.compute_gradients = K.function(inputs=self.input_tensor,
                                            outputs=self.grad)
        saliency = self.compute_gradients([input_img])
        return saliency[0]

    def smoothgrad(self, input_img, target_class, noise_mean=0.0,
                   noise_std_spread=0.15,
                   nsamples=25):
        maxinput = np.max(input_img)
        mininput = np.min(input_img)
        stdev = noise_std_spread * (maxinput - mininput)

        total_gradients = np.zeros_like(input_img)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_img.shape)
            input_plus_noise = input_img + noise
            ingrad = self.saliencymap(input_plus_noise, target_class)
            total_gradients += ingrad[0]

        return total_gradients/nsamples

    def integrated_gradients(self, input_img, target_class,
                             baseline=None,
                             xsteps=25):
        if baseline is None:
            baseline = np.zeros_like(input_img)

        assert baseline.shape == input_img.shape

        x_diff = input_img - baseline
        total_gradients = np.zeros_like(input_img)
        for alpha in np.linspace(0, 1, xsteps):
            x_step = baseline + alpha * x_diff
            ingrad = self.saliencymap(x_step, target_class)
            total_gradients += ingrad[0]

        return total_gradients * x_diff/xsteps


def reinitlayers(model, listlayernames, boolprint=False):
    """Reintialize Keras Layers (From Stackoverflow)."""
    sess = K.get_session()
    for layername in listlayernames:
        if isinstance(model.get_layer(layername),
                  keras.engine.network.Network):
            reinitlayers(model.get_layer(layername))
            if boolprint:
                print("reinit: {}".format(layername))
        for v in model.get_layer(layername).__dict__:
            v_arg = getattr(model.get_layer(layername), v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=sess)
                if boolprint:
                    print("reiniliazing layer {}.{}".format(layername, v))
    return True


def main():
    pass


if __name__ == '__main__':
    main()
