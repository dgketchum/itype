from collections import OrderedDict
import tensorflow as tf

'''
Feature spec for reading/writing tf records
'''

DEFAULT_VALUE = tf.ones([512, 512], dtype=tf.float32) * -1.

features_dict = OrderedDict(
    [('0_blue_mean', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_green_mean', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_red_mean', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_nir_mean', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('0_swir1_mean', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE))])


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
