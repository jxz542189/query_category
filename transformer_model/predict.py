import os
import collections
import pandas as pd
import tensorflow as tf
from termcolor import colored
from sklearn.utils import shuffle
from sklearn.externals import joblib
from utils.data_helpers import *
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.contrib import learn


basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
logger = set_logger(colored('train', 'cyan'), False)

label2id = joblib.load(os.path.join(basedir, 'transformer_model', 'model', 'label2id.m'))
# label2id = {'603': 0, '622': 1, '609': 2, '628': 3, '607': 4, '602': 5,
#             '646': 6, '606': 7, '614': 8, '610': 9, '625': 10, '621': 11,
#             '605': 12, '619': 13, '676': 14, '608': 15, '636': 16, '613': 17,
#             '604': 18, '620': 19, '611': 20, '647': 21, '601': 22, '626': 23,
#             '637': 24, '617': 25, '616': 26, '651': 27, '627': 28, '629': 29,
#             '623': 30, '615': 31, '618': 32, '612': 33, '624': 34, '635': 35}
id2label = {}
for key in label2id:
    id2label[label2id[key]] = key

word2id = read_word2id(os.path.join(basedir, 'data', 'word2id.txt'))

model_file = os.path.join(basedir, "transformer_model", "model", "model.pb")
with tf.gfile.GFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

input_names = ['input_x']


def model_fn(features, labels, mode, params):

    predicts = tf.import_graph_def(graph_def,
                                   input_map={k + ":0": features[k] for k in input_names},
                                   return_elements=['transformer-encoding/encoded_outputs:0',
                                                    'output/softmax_scores:0',
                                                    'output/predictions:0'])
    return EstimatorSpec(mode=mode, predictions={
        "encoded_outputs": predicts[0],
        "scores": predicts[1],
        "predictions": predicts[2]
    })


def get_estimator():
    config = tf.ConfigProto(device_count={'GPU': 0})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.log_device_placement = False

    return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))


def input_fn_builder(lines):
    def gen():
        inputs_x = []
        for line in lines:
            inputs_x.append(process(line, word2id, max_sequence_length=10))

        yield {
            'input_x': inputs_x
        }

    def input_fn():
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={'input_x': tf.int32},
            output_shapes={
                'input_x': (None, None)
            }
        ).prefetch(None))

    return input_fn

estimator = get_estimator()


class InputFeatures(object):
    def __init__(self,
                 input_x,
                 input_y):
        self.input_x = input_x
        self.input_y = input_y


def convert_batch_example(batch_examples, batch_label):
    if isinstance(batch_examples, str):
        batch_examples = [batch_examples]

    result = estimator.predict(input_fn_builder(batch_examples), yield_single_examples=False)
    inputs_x = []

    for r in result:
        inputs_x = r['encoded_outputs']
        # inputs_x = [float(a) for input_x in inputs_x for a in input_x]
        break

    res = []

    for input_x, input_y in zip(inputs_x, batch_label):
        res.append(InputFeatures(input_x, input_y))

    return res


def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            output_file,
                                            batch_size=100,
                                            category='601'):
    writer = tf.python_io.TFRecordWriter(output_file)

    num_batches = len(examples) // batch_size + 1
    for i in range(0, num_batches):
        if i % 10000 == 0:
            logger.info(f"{category} current processing : {i}")
        batch_examples = examples[i * batch_size : (i+1) * batch_size]
        batch_labels = label_list[i * batch_size : (i+1) * batch_size]

        if len(batch_examples) == 0:
            continue

        batch_features = convert_batch_example(batch_examples, batch_labels)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

        for current_feature in batch_features:
            features = collections.OrderedDict()
            features["input_x"] = create_float_feature(current_feature.input_x)
            features["input_y"] = create_int_feature([current_feature.input_y])
            logger.info(features)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    lines = ['men,clothing', 'hard bagpack', 'hard wax bean']
    result = estimator.predict(input_fn_builder(lines), yield_single_examples=False)
    for r in result:
        print(r['scores'])
    # for level_3_cat in label2id.keys():
    # for level_3_cat in ['608']:
    #     if level_3_cat in ['601', '602']:
    #         continue
    #     df = pd.read_csv(os.path.join(basedir, 'data', f'train_level_3_{level_3_cat}.csv'), header=None)
    #     df = shuffle(df)
    #     records = df.to_dict("records")
    #     record_len = len(records)
    #     train_records, test_records = records[: int(record_len * 0.99)], records[int(record_len * 0.99):]
    #
    #     train_examples = []
    #     train_label_list = []
    #     eval_examples = []
    #     eval_label_list = []
    #     for record in train_records:
    #         train_examples.append(record[0])
    #         train_label_list.append(record[1])
    #
    #     for record in test_records:
    #         eval_examples.append(record[0])
    #         eval_label_list.append(record[1])
    #
    #     file_based_convert_examples_to_features(train_examples,
    #                                             train_label_list,
    #                                             os.path.join(basedir, 'data', f'train_level_3_{level_3_cat}.tfrecords'),
    #                                             category=level_3_cat)
    #
    #     file_based_convert_examples_to_features(eval_examples,
    #                                             eval_label_list,
    #                                             os.path.join(basedir, 'data', f'eval_level_3_{level_3_cat}.tfrecords'),
    #                                             category=level_3_cat)
    #



