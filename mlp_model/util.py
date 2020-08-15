import time
import datetime
import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from tensorflow.contrib import learn
from utils import optimization
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec



basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(basedir, "transformer_model/model/vocab"))
word2id = vocab_processor.vocabulary_._mapping

model_file = os.path.join(basedir, "transformer_model/model/model.pb")


def model_fn(features, labels, mode, params):
    with tf.gfile.GFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    input_names = ['input_x']
    predicts = tf.import_graph_def(graph_def,
                                   input_map={k + ":0" : features[k] for k in input_names},
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


def process(line, word2id, max_sequence_length=15):
    res = [0] * 15

    words = line.split(' ')[:max_sequence_length]
    for i, word in enumerate(words):
        if word in word2id:
            res[i] = word2id[word]
        else:
            res[i] = 0

    return res


def input_fn_builder(lines):
    def gen():
        inputs_x = []
        for line in lines:
            inputs_x.append(process(line, word2id))

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