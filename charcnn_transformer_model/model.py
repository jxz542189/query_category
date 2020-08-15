import numpy as np
import tensorflow as tf
from transformer_model.transformer_layer import Transformer
from transformer_model.attention import positional_encoding


class CharCnnTransformerModel(object):
    def __init__(self,
                 sequence_length,
                 char_length,
                 num_classes,
                 vocab_size,
                 char_size,
                 embedding_size,
                 char_embedding_size,
                 max_chars,
                 projection_dim,
                 n_highway,
                 num_layers,
                 num_heads,
                 linear_key_dim,
                 linear_value_dim,
                 model_dim,
                 ffn_dim,
                 l2_reg_lambda=0.1):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_char_x = tf.placeholder(tf.int32, [None, sequence_length, char_length], name='input_char_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="word_W")
            self.embedded_words = tf.nn.embedding_lookup(self.word_W, self.input_x)
            self.encoded_inputs = self.embedded_words
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

        with tf.variable_scope('char_embedding'):
            self.char_W = tf.Variable(tf.random_uniform([char_size, char_embedding_size], -1.0, 1.0), name="char_W")
            self.embedded_chars = tf.nn.embedding_lookup(self.char_W, self.input_char_x)
            filters = [[1, 32],
                       [2, 32],
                       [3, 64]]
            n_filters = sum(f[1] for f in filters)
            convolutions = []
            for i, (width, num) in enumerate(filters):
                w_init = tf.random_normal_initializer(mean=0.0,
                                                          stddev=np.sqrt(1.0 / (width * char_embedding_size)))
                w = tf.get_variable("W_cnn_%s" % i,
                                    [1, width, char_embedding_size, num],
                                    initializer=w_init,
                                    dtype=tf.float32)
                b = tf.get_variable("b_cnn_%s" % i,
                                    [num],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(self.embedded_chars, w,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID") + b
                conv = tf.nn.max_pool(conv,
                                      [1, 1, max_chars - width + 1, 1],
                                      [1, 1, 1, 1], "VALID")

                conv = tf.nn.relu(conv)
                conv = tf.squeeze(conv, squeeze_dims=[2])
                convolutions.append(conv)
            self.embedded_chars = tf.concat(convolutions, 2)

            batch_size_n_tokens = tf.shape(self.embedded_chars)[0:2]
            self.embedded_chars = tf.reshape(self.embedded_chars, [-1, n_filters])

            with tf.variable_scope('CNN_proj'):
                W_proj_cnn = tf.get_variable("W_proj", [n_filters, model_dim],
                                             initializer=tf.random_normal_initializer(mean=0.0,
                                                                                      stddev=np.sqrt(1.0 / n_filters)),
                                             dtype=tf.float32)
                b_proj_cnn = tf.get_variable("b_proj",
                                             [model_dim],
                                             initializer=tf.constant_initializer(0.0),
                                             dtype=tf.float32)
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i):
                    W_carry = tf.get_variable('W_carry', [highway_dim, highway_dim],
                                              initializer=tf.random_normal_initializer(
                                                  mean=0.0, stddev=np.sqrt(1.0 / highway_dim)
                                              ), dtype=tf.float32)
                    b_carry = tf.get_variable('b_carry',
                                              [highway_dim],
                                              initializer=tf.constant_initializer(-2.0),
                                              dtype=tf.float32)
                    W_transform = tf.get_variable('W_transform',
                                                  [highway_dim, highway_dim],
                                                  initializer=tf.random_normal_initializer(mean=0.0,
                                                                                           stddev=np.sqrt(1.0/ highway_dim)),
                                                  dtype=tf.float32)
                    b_transform = tf.get_variable('b_transform',
                                                  [highway_dim],
                                                  initializer=tf.constant_initializer(0.0),
                                                  dtype=tf.float32)
                self.embedded_chars = high(self.embedded_chars, W_carry, b_carry, W_transform, b_transform)
            self.embedded_chars = tf.matmul(self.embedded_chars, W_proj_cnn) + b_proj_cnn
            shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            self.embedded_chars = tf.reshape(self.embedded_chars, shp)
            self.encoded_inputs = tf.concat([self.encoded_inputs, self.embedded_chars], axis=2)

        with tf.variable_scope("transformer-encoding"):
            self.encoder = Transformer(num_layers=num_layers,
                                       num_heads=num_heads,
                                       linear_key_dim=linear_key_dim,
                                       linear_value_dim=linear_value_dim,
                                       model_dim=2 * model_dim,
                                       ffn_dim=ffn_dim,
                                       dropout=0)
            self.encoded_outputs = self.encoder.build(self.encoded_inputs)
            self.encoded_outputs = tf.reshape(self.encoded_outputs, [-1, sequence_length * 2 * model_dim], name="encoded_outputs")

        with tf.variable_scope("output"):
            W = tf.get_variable("Weights",
                                shape=[sequence_length * 2 * model_dim, num_classes],
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


def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
    carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
    transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
    return carry_gate * transform_gate + (1.0 - carry_gate) * x


