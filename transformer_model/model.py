import numpy as np
import tensorflow as tf
from transformer_model.transformer_layer import Transformer
from transformer_model.attention import positional_encoding


class TransformerModel(object):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 num_layers,
                 num_heads,
                 linear_key_dim,
                 linear_value_dim,
                 model_dim,
                 ffn_dim,
                 l2_reg_lambda=0.1):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.encoded_inputs = self.embedded_chars

        with tf.variable_scope("positional-encoding"):
            self.positional_encoded = positional_encoding(embedding_size,
                                                     sequence_length,
                                                     dtype=tf.float32)

        with tf.name_scope("inputs-encoding"):
            num_dims = len(self.input_x.shape.as_list())
            position_broadcast_shape = []
            for _  in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([sequence_length, embedding_size])

            self.position_inputs = tf.reshape(self.positional_encoded,
                                         position_broadcast_shape)

            self.encoded_inputs += self.position_inputs

        with tf.variable_scope("transformer-encoding"):
            self.encoder = Transformer(num_layers=num_layers,
                                       num_heads=num_heads,
                                       linear_key_dim=linear_key_dim,
                                       linear_value_dim=linear_value_dim,
                                       model_dim=model_dim,
                                       ffn_dim=ffn_dim,
                                       dropout=0)
            self.encoded_outputs = self.encoder.build(self.encoded_inputs)
            self.encoded_outputs = tf.reshape(self.encoded_outputs, [-1, sequence_length * model_dim], name="encoded_outputs")

        with tf.variable_scope("output"):
            W = tf.get_variable("Weights",
                                shape=[sequence_length * model_dim, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.encoded_outputs, W, b, name="scores")
            self.scores = tf.nn.softmax(self.scores, name='softmax_scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")






