import time
import datetime
import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from char_textcnn_model.model import CharTextCNN
from utils import optimization

logger = set_logger(colored('train', 'cyan'), False)

tf.flags.DEFINE_float("dev_sample_percentage", 0.00005, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "../data/train_level_3_601.csv", "data file")
tf.flags.DEFINE_string("word2id_file", "../data/word2id.txt", "word2id file")
tf.flags.DEFINE_string("char2id_file", "../data/char2id.txt", "word2id file")
tf.flags.DEFINE_string("id2label_file", "../data/id2label_601.txt", "word2id file")

tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("char_embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("n_highway", 3, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("warmup_proportion", 0.1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 5e-5, "learning_rate")
tf.flags.DEFINE_float("dropout_keep_prob", 0.1, "dropout_keep_prob")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer('sequence_length', 15, "max query length")
tf.flags.DEFINE_integer('max_chars', 15, "max word length")
tf.flags.DEFINE_integer('early_stop_step', 10000, "early_stop_step")
tf.flags.DEFINE_integer('num_layers', 3, "num_layers")
tf.flags.DEFINE_integer('num_heads', 8, "num_heads")
tf.flags.DEFINE_integer('linear_key_dim', 64, "linear_key_dim")
tf.flags.DEFINE_integer('linear_value_dim', 64, "linear_value_dim")
tf.flags.DEFINE_integer('model_dim', 64, "model_dim")
tf.flags.DEFINE_integer('projection_dim', 64, "projection_dim")
tf.flags.DEFINE_integer('ffn_dim', 64, "ffn_dim")
tf.flags.DEFINE_integer('learning_rate_decay_num', 3, "learning_rate_decay_num")


FLAGS = tf.flags.FLAGS

basedir = os.path.abspath(os.path.dirname(__file__))


def preprocess(data_file):
    logger.info("Loading data...")
    label2id = id2label(FLAGS.id2label_file, sep='\u0001')
    logger.info(label2id)
    x_text, y, num_classes = load_data_and_labels_v1(data_file, label2id, level=3)
    word2id = read_word2id(FLAGS.word2id_file)
    char2id = read_word2id(FLAGS.char2id_file)
    logger.info(f"num_classes: {num_classes}")

    x = x_text
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
    logger.info(f"num_char: {len(char2id)}")
    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev, word2id, char2id, label2id


def train(x_train, y_train, x_dev, y_dev, word2id, char2id, label2id, leaf_category):
    acc = 0.0
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

        num_classes = len(label2id)
        with tf.Session(config=session_conf) as sess:
            cnn = CharTextCNN(sequence_length=FLAGS.sequence_length,
                                          char_length=FLAGS.max_chars,
                                          num_classes=num_classes,
                                          vocab_size=len(word2id)+1,
                                          char_size=len(char2id) + 20,
                                          embedding_size=FLAGS.embedding_dim,
                                          char_embedding_size=FLAGS.char_embedding_dim,
                                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                                          num_filters=FLAGS.num_filters,
                                          n_highway=FLAGS.n_highway,
                                          model_dim=FLAGS.model_dim,
                                          projection_dim=FLAGS.projection_dim,
                                          l2_reg_lambda=FLAGS.l2_reg_lambda)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            num_train_steps = int(len(x_train) / FLAGS.batch_size * FLAGS.num_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
            train_op = optimization.create_optimizer(
                cnn.loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, None, global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(basedir, f"runs_{leaf_category}", timestamp))
            logger.info("Wrting to {}\n".format(out_dir))

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
                    tmp[int(id)] = 1
                    y.append(tmp)

                x = []
                x_char = []
                for line in list(x_batch):
                    x.append(process(line, word2id, max_sequence_length=FLAGS.sequence_length))
                    x_char.append(char_process(line, char2id, max_sequence_length=FLAGS.sequence_length,
                                               max_chars=FLAGS.max_chars))
                y_batch_1 = np.array(y)
                feed_dict = {
                    cnn.input_x: np.array(x),
                    cnn.input_char_x: np.array(x_char),
                    cnn.input_y: y_batch_1,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy, predictions = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict
                )
                time_str = datetime.datetime.now().isoformat()
                logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch, writer=None):
                y = []
                for id in list(y_batch):
                    tmp = [0] * num_classes
                    tmp[int(id)] = 1
                    y.append(tmp)

                x = []
                x_char = []
                for line in list(x_batch):
                    x.append(process(line, word2id, max_sequence_length=FLAGS.sequence_length))
                    x_char.append(char_process(line, char2id, max_sequence_length=FLAGS.sequence_length,
                                               max_chars=FLAGS.max_chars))
                y_batch_1 = np.array(y)
                feed_dict = {
                    cnn.input_x: np.array(x),
                    cnn.input_char_x: np.array(x_char),
                    cnn.input_y: y_batch_1,
                    cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict
                )
                return accuracy

            batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            total_batch = 0
            last_improved_batch = 0
            learning_rate_decay_num = FLAGS.learning_rate_decay_num
            flags = True
            for batch in batches:
                total_batch += 1
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("{} Evaluation:".format(current_step))
                    current_acc = dev_step(x_dev, y_dev, writer=None)
                    logger.info("{} Evaluation, acc: {:g}".format(current_step, current_acc))
                    if current_acc > acc:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logger.info("Saved model checkpoint to {}\n".format(path))
                        acc = current_acc
                        last_improved_batch = total_batch
                if flags and total_batch - last_improved_batch > FLAGS.early_stop_step:
                    if learning_rate_decay_num > 0:
                        logger.info('No optimization for a long time, auto-stopping ...')
                        last_improved_batch = total_batch
                        FLAGS.learning_rate = FLAGS.learning_rate / 2
                        logger.info('current learning_rate is {}'.format(FLAGS.learning_rate))
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

