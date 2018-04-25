#  This code was modified from the Tensorflow premade estimator tutorial.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
import numpy as np

import argparse
import tensorflow as tf

from PIL import Image
import requests
import urllib
import os

#import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    
    # Seem to need ability to install h5py to use VGG16
    #conv_base = VGG16(weights='imagenet',
    #              include_top=False,
    #              input_shape=(150, 150, 3))

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation="relu", input_shape=(150, 150, 3)),  # input shape required
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(3)
    ])

    #model = models.Sequential()
    #model.add(conv_base)
    model.add(layers.Flatten())
    #model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dense(1, activation='sigmoid'))
    #conv_base.trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])
                  
    model_dir = os.path.join(os.getcwd(), "models/catvsdog")
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir: ",model_dir)
    est_catvsdog = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                        model_dir=model_dir)
              
    # Remove all non-functioning urls
    urlFile = open("dog_URLs.txt")
    test_urls = urlFile.readlines()
    test_labels = []
    count = 0
    for url in test_urls:
        if count % 2 == 0:
            test_labels.append("Dogs")
        else:
            test_labels.append("Cats")
        print(count)
        print(url)
        if not url:
            del test_urls[count]
            count += 1
        else:
            try:
                url.encode('ascii')
                
            except UnicodeEncodeError:
                del test_urls[count]
                count += 1
                continue
            try:
                req = urllib.request.Request(url)
            except IOError:
                del test_urls[count]
                count += 1
                continue
            try:
                response = urllib.request.urlopen(req, timeout = 1)
            except IOError:
                del test_urls[count]
            count += 1
    print(len(test_urls))
    train_y = np.asarray(test_labels).astype('str').reshape((-1,1))
    
    # TODO Separate training set from test set
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(test_urls,
                                                                       labels=train_y,
                                                                       perform_shuffle=True,
                                                                       repeat_count=5,
                                                                       batch_size=20), 
                                        max_steps=500)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(test_urls,
                                                                     labels=train_y,
                                                                     perform_shuffle=False,
                                                                     batch_size=1))

    tf.estimator.train_and_evaluate(est_catvsdog, train_spec, eval_spec)
                                                        
                                                        

def imgs_input_fn(filenames, labels=None, perform_shuffle=False, repeat_count=1, batch_size=1):

    def _parse_function(filename, label):
        #req = urllib.request.Request(tf.as_string(filename))
        #response = urllib.request.urlopen(req, timeout = 1)  
        #image_data = response.read()
                
        image_string = tf.read_file(filename)
        image = tf.image.decode_image(image_string, channels=3)
        image.set_shape([None, None, None])
        image = tf.image.resize_images(image, [150, 150])
        image = tf.subtract(image, 116.779) # Zero-center by mean pixel
        image.set_shape([150, 150, 3])
        image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
        d = dict(zip(["dense_1_input"], [image])), label
        return d
        
    if labels is None:
        labels = [0]*len(filenames)
    labels=np.array(labels)
    
    # Expand the shape of "labels" if necessary
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=67500)
        
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    labels = tf.cast(labels, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
    


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
