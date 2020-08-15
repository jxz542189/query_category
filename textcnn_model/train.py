import time
import datetime
import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from textcnn_model.text_cnn import TextCNN
from tensorflow.contrib import learn

logger = set_logger(colored('train', 'cyan'), False)


tf.flags.DEFINE_float("dev_sample_percentage", 0.001, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "", "data file")
tf.flags.DEFINE_string("word2id_file", "./data/word2id.txt", "word2id file")

tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-2, "learning_rate")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer('max_document_length', 15, "max query length")
tf.flags.DEFINE_integer('early_stop_step', 100000, "early_stop_step")


FLAGS = tf.flags.FLAGS

basedir = os.path.abspath(os.path.dirname(__file__))


def preprocess(data_file):
    logger.info("Loading data...")
    x_text, y, label2id, num_classes = load_data_and_labels(data_file, level=3)
    logger.info(f"num_classes: {num_classes}")
    word2id = read_word2id(FLAGS.word2id_file)
    max_document_length = FLAGS.max_document_length
    x = []
    for line in x_text:
        x.append(process(line, word2id, max_sequence_length=max_document_length))
    x = np.array(x)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y = np.array(y)
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    logger.info("Vocabulary Size: {:d}".format(len(word2id)))
    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, word2id, x_dev,  y_dev, label2id


def train(x_train, y_train, word2id, x_dev, y_dev, label2id, leaf_category):
    acc = 0.0
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        num_classes = len(label2id)
        with sess.as_default():
            cnn = TextCNN(sequence_length=x_train.shape[1],
                          num_classes=num_classes,
                          vocab_size=len(word2id) + 1,
                          embedding_size=FLAGS.embedding_dim,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          l2_reg_lambda=FLAGS.l2_reg_lambda)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(basedir, f"runs_{leaf_category}", timestamp))
            logger.info("category {}, Wrting to {}\n".format(leaf_category, out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            label_dir = os.path.join(out_dir, "label2id.m")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            joblib.dump(label2id, label_dir)

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                y = []
                for id in list(y_batch):
                    tmp = [0] * num_classes
                    tmp[id] = 1
                    y.append(tmp)

                y_batch_1 = np.array(y)
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch_1,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy, predictions = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict
                )

                time_str = datetime.datetime.now().isoformat()
                logger.info("category: {}, {}: step {}, loss {:g}, acc {:g}".format(leaf_category, time_str, step, loss, accuracy))                # train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                y = []
                for id in list(y_batch):
                    tmp = [0] * num_classes
                    tmp[id] = 1
                    y.append(tmp)

                y_batch = np.array(y)

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict
                )
                time_str = datetime.datetime.now().isoformat()
                logger.info("category: {}, {}: step {}, loss {:g}, acc {:g}".format(leaf_category, time_str, step, loss, accuracy))
                # if writer:
                #     writer.add_summary(summaries, step)
                return accuracy

            batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            total_batch = 0
            last_improved_batch = 0
            learning_rate_decay_num = 3
            flags = True
            for batch in batches:
                total_batch += 1
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("category: {}, {} Evaluation:".format(leaf_category, current_step))
                    current_acc = dev_step(x_dev, y_dev, writer=None)
                    logger.info("category: {}, {} Evaluation, acc: {:g}".format(leaf_category, current_step, current_acc))
                    if current_acc > acc:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logger.info("category: {}, Saved model checkpoint to {}\n".format(leaf_category, path))
                        acc = current_acc
                        last_improved_batch = total_batch
                if flags and total_batch - last_improved_batch > FLAGS.early_stop_step:
                    logger.info('category: {}, No optimization for a long time, auto-stopping ...'.format(leaf_category))
                    last_improved_batch = total_batch
                    FLAGS.learning_rate = FLAGS.learning_rate / 2
                    logger.info('category: {}, current learning_rate is {}'.format(leaf_category, FLAGS.learning_rate))
                    learning_rate_decay_num -= 1
                    if learning_rate_decay_num == 0:
                        flags = False


def main(argv=None):
    label2id = {'603': 0, '622': 1, '609': 2, '628': 3, '607': 4, '602': 5,
                '646': 6, '606': 7, '614': 8, '610': 9, '625': 10, '621': 11,
                '605': 12, '619': 13, '676': 14, '608': 15, '636': 16, '613': 17,
                '604': 18, '620': 19, '611': 20, '647': 21, '601': 22, '626': 23,
                '637': 24, '617': 25, '616': 26, '651': 27, '627': 28, '629': 29,
                '623': 30, '615': 31, '618': 32, '612': 33, '624': 34, '635': 35}

    for leaf_category in label2id:
        if leaf_category not in ['601']:
            print(leaf_category)
            continue

        try:
            data_file = f"./data/train_level_3_{leaf_category}.csv"
            x_train, y_train, word2id, x_dev, y_dev, label2id = preprocess(data_file)
            train(x_train, y_train, word2id, x_dev, y_dev, label2id, leaf_category)
        except:
            pass


if __name__ == '__main__':
    tf.app.run()
