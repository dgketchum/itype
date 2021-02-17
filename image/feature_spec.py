from collections import OrderedDict
import tensorflow as tf

'''
Feature spec for reading/writing tf records
'''

DEFAULT_VALUE = tf.ones([512, 512], dtype=tf.float32) * -1.

feature_spec = OrderedDict(
    [('R', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('G', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('B', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('N', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('std_ndvi', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('mx_ndvi', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE)),
     ('itype', tf.io.FixedLenFeature(shape=[512, 512], dtype=tf.float32, default_value=DEFAULT_VALUE))],)


def features_dict():
    print('PROCESSING {} BANDS'.format(len(feature_spec.keys())))
    return feature_spec


def bands():
    bands = list(feature_spec.keys())
    return bands


def features():
    features = list(feature_spec.keys())[:-1]
    return features


if __name__ == '__main__':
    print(len(features()))
    print(len(bands()))
