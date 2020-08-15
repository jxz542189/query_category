import numpy as np
import tensorflow as tf


class CharTextCNN(object):
    def __init__(self,
                 sequence_length,
                 char_length,
                 num_classes,
                 vocab_size,
                 char_size,
                 embedding_size,
                 char_embedding_size,
                 filter_sizes,
                 num_filters,
                 n_highway,
                 model_dim,
                 projection_dim,
                 l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_char_x = tf.placeholder(tf.int32, [None, sequence_length, char_length], name='input_char_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        filter_sizes

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="pool")
            pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)

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
                                      [1, 1, char_length - width + 1, 1],
                                      [1, 1, 1, 1], "VALID")

                conv = tf.nn.relu(conv)
                conv = tf.squeeze(conv, squeeze_dims=[2])
                convolutions.append(conv)
            self.embedded_chars = tf.concat(convolutions, 2)

            batch_size_n_tokens = tf.shape(self.embedded_chars)[0:2]
            shp = self.embedded_chars.get_shape().as_list()
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
            # shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            # self.embedded_chars = tf.reshape(self.embedded_chars, shp)
            self.embedded_chars = tf.reshape(self.embedded_chars, [-1, shp[1] * projection_dim])

            self.h_pool = tf.reshape(self.h_pool, [-1, num_filters_total])
            self.h_pool_flat = tf.concat([self.h_pool, self.embedded_chars], axis=1)

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filters_total + shp[1] * projection_dim, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
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


