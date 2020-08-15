import tensorflow as tf
from utils.highway import *


class MLP(object):
    def __init__(self,
                 num_classes,
                 input_x,
                 input_y,
                 embedding_size=1920,
                 num_layers=3,
                 l2_reg_lambda=0.1,
                 sequence_length=15,
                 emb_size=128):
        self.input_x = input_x
        self.input_y = input_y
        # self.input_x = tf.placeholder(tf.float32, [None, embedding_size], name='input_x')
        # self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')

        l2_loss = tf.constant(0.0)

        self.input_x = tf.reshape(self.input_x, [-1, sequence_length, emb_size], name='reshape_0')
        self.input_x = highway_network(self.input_x, num_layers, bias=True)
        self.input_x = tf.reshape(self.input_x, [-1, embedding_size], name='reshape_1')

        with tf.variable_scope("output"):
            W = tf.get_variable("Weights",
                                shape=[embedding_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.input_x, W, b, name="scores")
            self.scores = tf.nn.softmax(self.scores, name='softmax_scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")







