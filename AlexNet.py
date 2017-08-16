# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'Alexnet' to Googles flower dataset from the retraining tutorial.
http://download.tensorflow.org/example_images/flower_photos.tgz
The data has to be repacked into a h5py dataset. The network is downsized to run on a desktop computer. The data is split into a data and validation test set after loading it.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tflearn
import h5py
import numpy as np
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def split_sets(X, Y, percentage):
    n = len(X)
    nb_samples = int(n * percentage)
    # sample without replacement from [0,n]
    samples = np.random.choice(n, nb_samples, replace=False)
    X_validation = [X[i] for i in samples]
    Y_validation = [Y[i] for i in samples]
    X = [X[i] for i in range(0, n) if i not in samples]
    Y = [Y[i] for i in range(0, n) if i not in samples]
    #
    return X, Y, X_validation, Y_validation

# Data loading and preprocessing

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']
X, Y = shuffle(X, Y)
epochs=5

X, Y, X_validation, Y_validation = split_sets(X,Y, 0.3)

#X, Y, X_validation, Y_validation = split_sets(X,Y, 0.2)

# Building 'AlexNet'
network = input_data(shape=[None, 128, 128, 3])
network = conv_2d(network, 96, 11, strides=3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 5, activation='relu')
network = max_pool_2d(network, 3, strides=1)
network = local_response_normalization(network)
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 196, 3, activation='relu')
network = max_pool_2d(network, 3, strides=1)
network = local_response_normalization(network)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

def calculate_accuracy(model, X_validation, Y_validation):
    probabilities = model.predict(X_validation)
    maxpos_calc = [np.argmax(elem) for elem in probabilities]
    maxpos_real = [np.argmax(elem) for elem in Y_validation]
    h = 0.0
    for i in range(0, len(maxpos_calc)):
        if maxpos_calc[i] == maxpos_real[i]:
            h += 1
    print("Accuracy is " + str(h / len(maxpos_real)))

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
for i in range(0, epochs):
    model.fit(X, Y, n_epoch=1, shuffle=True, show_metric=False, batch_size=64, snapshot_step=200,
    snapshot_epoch=False, run_id='alexnet_oxflowers17')
    calculate_accuracy(model, X_validation, Y_validation)
