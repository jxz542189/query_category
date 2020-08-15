import h5py
import csv
import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from tensorflow.contrib import learn
from simple_bert_model.tokenization import *
from simple_bert_model.bert_modeling import BertModel, BertConfig


logger = set_logger(colored('bert_weights', 'cyan'), False)

tf.flags.DEFINE_string("checkpoint_dir", "../data/uncased_L-12_H-768_A-12", "Checkpoint directory from training run")
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# tokenizer = FullTokenizer(vocab_file=os.path.join(FLAGS.checkpoint_dir, 'vocab.txt'), do_lower_case=True)
# config = BertConfig.from_json_file(os.path.join(FLAGS.checkpoint_dir, 'bert_config.json'))
weight_file = os.path.join(FLAGS.checkpoint_dir, 'dump_weights.hdf5')


class Model(object):
    def __init__(self, config, weight_file):
        self.input_x = tf.placeholder(tf.int32, [None, config.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, config.num_classes], name='input_y')

        l2_loss = tf.constant(0.0)

        def _custom_getter(getter, name, *args, **kwargs):
            kwargs['trainable'] = True
            kwargs['initializer'] = _pretrained_initializer(name, weight_file)

            return getter(name, *args, **kwargs)

        with tf.variable_scope('bert', custom_getter=_custom_getter):
            model = BertModel(config, config.is_training, self.input_x)

        with tf.name_scope('encoded'):
            self.input_encoded = model.all_encoder_layers[0]
            width = self.input_encoded.shape[1] * self.input_encoded.shape[2]
            self.input_encoded = tf.reshape(self.input_encoded,
                                            [-1, width])

        with tf.name_scope("output"):
            W = tf.get_variable("Weights", shape=[width, config.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=[config.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.input_encoded, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + config.l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_x, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def _custom_getter(getter, name, *args, **kwargs):
    kwargs['trainable'] = True
    kwargs['initializer'] = _pretrained_initializer(name, weight_file)

    return getter(name, *args, **kwargs)


def _pretrained_initializer(varname, weight_file):
    varname = varname[5:]
    varname_in_file = re.sub(":0$", "", varname)

    with h5py.File(weight_file, 'r') as fin:
        weights = fin[varname_in_file][...]

    def ret(shape, **kwargs):
        if list(shape) != list(weights.shape):
            raise ValueError(
                "Invalid shape initializing {0}, got {1}, expected {2}".format(
                    varname_in_file, shape, weights.shape)
            )
        return weights

    return ret
