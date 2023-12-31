from src.utils import *
import tensorflow as tf


def create_dataset(data_list, label_list, dataset_slice, tfrecord_save_path):
    if len(data_list) != len(label_list):
        raise ValueError('The number of images is not equal to the number of tags')
    if len(data_list) % dataset_slice:
        raise ValueError('The number of shards cannot be divisible evenly')
    tfrecord_name_generator = name_generator('data_pair', 'tfrecord')
    logging.info('>>>开Start generating a dataset, and save the dataset to%s<<<' % tfrecord_save_path)
    for i in range(0, len(data_list), dataset_slice):
        batch_data = data_list[i:i + dataset_slice]
        batch_label = label_list[i:i + dataset_slice]
        writer = tf.python_io.TFRecordWriter(
            path.join(tfrecord_save_path, next(tfrecord_name_generator)))
        for data, label in zip(batch_data, batch_label):
            data_raw = np.load(data)
            label_raw = np.load(label)
            feature = {}
            feature['data'] = tf.train.Feature(float_list=tf.train.FloatList(
                value=data_raw.flatten()))
            feature['data_shape'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=data_raw.shape))
            feature['label'] = tf.train.Feature(float_list=tf.train.FloatList(
                value=label_raw.flatten()))
            feature['label_shape'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=label_raw.shape))
            data_pair = tf.train.Features(feature=feature)
            tf_example = tf.train.Example(features=data_pair)
            tf_serialized = tf_example.SerializeToString()
            writer.write(tf_serialized)
        logging.info('>>>A data set is generated')
        writer.close()


def parse_dataset(proto):
    dataset_dict = {
        'data': tf.VarLenFeature(dtype=tf.float32),
        'data_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
        'label': tf.VarLenFeature(dtype=tf.float32),
        'label_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    }
    parsed_pair = tf.parse_single_example(proto, dataset_dict)
    parsed_pair['data'] = tf.sparse_tensor_to_dense(parsed_pair['data'])
    parsed_pair['data'] = tf.reshape(parsed_pair['data'],
                                     parsed_pair['data_shape'])
    parsed_pair['label'] = tf.sparse_tensor_to_dense(parsed_pair['label'])
    parsed_pair['label'] = tf.reshape(parsed_pair['label'],
                                      parsed_pair['label_shape'])
    return parsed_pair


def get_data_iterator(tfrecord_save_dir, epochs, batch_size, buffer_size):
    filenames = glob(path.join(tfrecord_save_dir, '*.tfrecord'))
    if len(filenames) == 0:
        raise ValueError('The tfrecord file was not found in the specified directory')
    logging.info('>>>Get {} datasets from {}<<<'.format(tfrecord_save_dir, len(filenames)))
    dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = dataset.map(parse_dataset)
    parsed_dataset = parsed_dataset.repeat(epochs)
    parsed_dataset = parsed_dataset.shuffle(buffer_size)
    parsed_dataset = parsed_dataset.batch(batch_size)
    iterator = parsed_dataset.make_one_shot_iterator()
    return iterator


def get_dataset_dirs(base_dir):
    dir_dict = {
        'original_data': path.join(base_dir, 'original/data'),  # data
        'original_label': path.join(base_dir, 'original/label'),  # label
        'split_data': path.join(base_dir, 'split/data'),  # data
        'split_label': path.join(base_dir, 'split/label'),  # label
        'split_label_classed': path.join(base_dir, 'split/label_classed'),
        'tfrecord': path.join(base_dir, 'tfrecord'),  # tfrecord
    }
    for dir_item in dir_dict.values():
        if not path.exists(dir_item):
            makedirs(dir_item)
    return dir_dict


def get_net_dirs(base_dir):
    dir_dict = {
        'model': path.join(base_dir, 'model'),  
        'log': path.join(base_dir, 'log'),  
        'prediction': path.join(base_dir, 'prediction'),  
    }
    for dir_item in dir_dict.values():
        if not path.exists(dir_item):
            makedirs(dir_item)
    return dir_dict
