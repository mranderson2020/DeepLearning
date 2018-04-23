import pandas as pd
import tensorflow as tf

from PIL import Image
import requests
import urllib


SPECIES = ["Dog", "Cat", "Bird"]

def load_data(y_name='Species'):
    urlFile = open("dog_URLs.txt")
    fileContents = urlFile.readlines()
    train_imgs = []
    test_imgs = []
    testSize = len(fileContents) * 0.2
    count = 0
    
    for url in fileContents:
        if not url:
            continue
        if count > 25:
            break
            
        count += 1
        print(count)
        try:
            url.encode('ascii')
        except UnicodeEncodeError:
            pass
        else:
            try:
                req = urllib.request.Request(url)
            except IOError:
                pass
            try:
                response = urllib.request.urlopen(req, timeout = 1)
            except IOError:
                pass
            else:    
                image_data = response.read()
                if testSize % count == 0:
                    test_imgs.append(image_data)
                else:
                    train_imgs.append(image_data)
    
    train_label = "dog"
    test_label = "dog"
    
    return (train_imgs, train_label), (test_imgs, test_label)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    train_imgs = []
    
    for image in features:
        train_imgs.append(tf.image.decode_jpeg(image))
    
    for x in range(len(train_imgs)):
        mapping = dict([ (x, img) for img in train_imgs ])
    dataset = tf.data.Dataset.from_tensor_slices((mapping))
    #assert dataset.graph is tf.get_default_graph()
    
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    #assert dataset.graph is tf.get_default_graph()

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
