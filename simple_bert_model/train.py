import time
import datetime
from termcolor import colored
from utils.data_helpers import *
from simple_bert_model.tokenization import *
from simple_bert_model.model import Model
from simple_bert_model.model import BertConfig, weight_file
from utils import optimization

logger = set_logger(colored('train', 'cyan'), False)

tf.flags.DEFINE_float("dev_sample_percentage", 0.001, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "../data/train_level_1_20200429.csv", "data file")

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("warmup_proportion", 0.1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 5e-5, "learning_rate")

tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer('max_document_length', 10, "max query length")
tf.flags.DEFINE_integer('early_stop_step', 5000, "early_stop_step")

FLAGS = tf.flags.FLAGS

basedir = os.path.abspath(os.path.dirname(__file__))

tokenizer = FullTokenizer(vocab_file=os.path.join(FLAGS.checkpoint_dir, 'vocab.txt'), do_lower_case=True)
config = BertConfig.from_json_file(os.path.join(FLAGS.checkpoint_dir, 'bert_config.json'))


def convert(x_text, sequence_length):
    res = []
    if not isinstance(x_text, list):
        x_text = [x_text]
    for x in x_text:
        ids = tokenizer.convert_tokens_to_ids(str(x).split(" "))
        ids = ids[:sequence_length] if len(ids) >= sequence_length else ids + (sequence_length - len(ids)) * [0]
        res.append(ids)
    return np.array(res)


def preprocess():
    logger.info("Loading data...")
    x_text, y, label2id, num_classes = load_data_and_labels(FLAGS.data_file, level=1)
    logger.info(f"num_classes: {str(num_classes)}")
    logger.info(f"config.num_classes: {str(config.num_classes)}")
    assert num_classes == config.num_classes

    max_document_length = FLAGS.max_document_length

    x = convert(x_text, max_document_length)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y = np.array(y)
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    logger.info(f"label2id: {label2id}")
    return x_train, y_train, x_dev,  y_dev, label2id


def train(x_train, y_train, x_dev, y_dev, label2id):
    num_train_steps = int(len(x_train) /FLAGS.batch_size * FLAGS.num_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    acc = 0.0
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        num_classes = len(label2id)
        with sess.as_default():
            cnn = Model(config, weight_file)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = optimization.create_optimizer(
                cnn.loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, None, global_step)
            # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            # grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # grad_summaries = []
            # for g, v in grads_and_vars:
            #     if g is not None:
            #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            #         grad_summaries.append(grad_hist_summary)
            #         grad_summaries.append(sparsity_summary)
            # grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            logger.info("Wrting to {}\n".format(out_dir))

            # loss_summary = tf.summary.scalar("loss", cnn.loss)
            # acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            # train_summary_dir = os.path.join(out_dir, "summaries", "train")
            # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            # dev_summary_dir = os.path.join(out_dir, "summaries", "train")
            # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

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
                    cnn.input_y: y_batch_1
                }
                _, step, loss, accuracy, predictions = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict
                )

                time_str = datetime.datetime.now().isoformat()
                logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                y = []
                for id in list(y_batch):
                    tmp = [0] * num_classes
                    tmp[id] = 1
                    y.append(tmp)

                y_batch = np.array(y)

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch
                }
                step, loss, accuracy, predictions = sess.run(
                    [global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict
                )

                time_str = datetime.datetime.now().isoformat()
                logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # if writer:
                #     writer.add_summary(summaries, step)
                return accuracy

            batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)

            total_batch = 0
            last_improved_batch = 0
            learning_rate_decay_num = 5
            flags = True
            for batch in batches:
                total_batch += 1
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("{} Evaluation:".format(current_step))
                    dev_batchs = batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, False)
                    dev_accs = []
                    for dev_batch in dev_batchs:
                        x_batch_dev, y_batch_dev = zip(*dev_batch)
                        tmp_acc = dev_step(x_batch_dev, y_batch_dev, writer=None)
                        dev_accs.append(tmp_acc)
                    current_acc = sum(dev_accs) / len(dev_accs)
                    logger.info("{} Evaluation, acc: {:g}".format(current_step, current_acc))
                    if current_acc > acc:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logger.info("Saved model checkpoint to {}\n".format(path))
                        acc = current_acc
                        # last_improved_batch = total_batch
                # if flags and total_batch - last_improved_batch > FLAGS.early_stop_step:
                #     logger.info('No optimization for a long time, auto-stopping ...')
                #     last_improved_batch = total_batch
                #     FLAGS.learning_rate = FLAGS.learning_rate / 2
                #     logger.info('current learning_rate is {}'.format(FLAGS.learning_rate))
                #     learning_rate_decay_num -= 1
                #     if learning_rate_decay_num == 0:
                #         flags = False


def main(argv=None):
    x_train, y_train, x_dev, y_dev, label2id = preprocess()
    train(x_train, y_train, x_dev, y_dev, label2id)


if __name__ == '__main__':
    tf.app.run()
