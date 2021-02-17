import os
import tensorflow as tf

from image import feature_spec

MODE = 'itype'
N_CLASSES = 5
FEATURES_DICT = feature_spec.features_dict()
FEATURES = feature_spec.features()


def make_test_dataset(root, pattern='*gz'):
    training_root = os.path.join(root, pattern)
    datasets = get_dataset(training_root)
    return datasets


def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
      pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
      A tf.data.Dataset
    """
    if not isinstance(pattern, list):
        pattern = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(pattern, compression_type='GZIP',
                                      num_parallel_reads=8)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    to_tup = to_tuple()
    dataset = dataset.map(to_tup, num_parallel_calls=5)
    return dataset


def parse_tfrecord(example_proto):
    """the parsing function.
    read a serialized example into the structure defined by features_dict.
    args:
      example_proto: a serialized example.
    returns:
      a dictionary of tensors, keyed by feature name.
    """
    parsed = tf.io.parse_single_example(example_proto, FEATURES_DICT)
    return parsed


def to_tuple():
    """
    Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of (inputs, outputs).
    """

    def to_tup(inputs):
        features_list = [inputs.get(key) for key in FEATURES]
        stacked = tf.stack(features_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        image_stack = stacked
        # 'constant' is the label for label raster.
        labels = one_hot(inputs.get(MODE), n_classes=N_CLASSES)
        labels = tf.cast(labels, tf.int32)
        return image_stack, labels

    return to_tup


def one_hot(labels, n_classes):
    h, w = labels.shape
    labels = tf.squeeze(labels) - 1
    ls = []
    for i in range(n_classes):
        where = tf.where(labels != i + 1, tf.zeros((h, w)), 1 * tf.ones((h, w)))
        ls.append(where)
    temp = tf.stack(ls, axis=-1)
    return temp


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
