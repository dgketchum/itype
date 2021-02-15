from collections import OrderedDict
import tensorflow as tf

'''
Feature spec for reading/writing tf records
'''

DEFAULT_VALUE = tf.ones([512, 512], dtype=tf.float32) * -1.

features_dict = OrderedDict(
    [('R', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('G', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('B', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('N', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('std_ndvi', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('mx_ndvi', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE))])


def features_dict():
    print('PROCESSING {} BANDS'.format(len(features_dict.keys())))
    return features_dict


def bands():
    bands = list(features_dict.keys())
    return bands


def features():
    features = list(features_dict.keys())[:-1]
    return features


if __name__ == '__main__':
    print(len(features()))
    print(len(bands()))
