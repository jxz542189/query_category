import h5py
import csv
import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from tensorflow.contrib import learn


logger = set_logger(colored('eval', 'cyan'), False)

tf.flags.DEFINE_string("data_file", "../data/train_level_3_20200421.csv", "data file")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "../textcnn_model/runs/1587511511/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

if FLAGS.eval_train:
    x_raw, y_test, label2id, num_classes, label2id = load_data_and_labels(FLAGS.data_file, is_train=False)

vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

logger.info("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        weight_file = os.path.join(FLAGS.checkpoint_dir, "..", "dump_weights.hdf5")
        if os.path.isfile(weight_file):
            os.remove(weight_file)
        with h5py.File(weight_file) as fout:
            for var in tf.trainable_variables():
                var_name = re.sub(":0$", '', var.name)
                logger.info("Saving variable {0} with name {1}".format(
                    var.name, var_name))
                shape = var.get_shape().as_list()
                dset = fout.create_dataset(var_name, shape, dtype='float32')
                values = sess.run([var])[0]
                dset[...] = values

        with h5py.File(weight_file, 'r') as fin:
            for var in tf.trainable_variables():
                var_name = re.sub(":0$", '', var.name)
                weights = fin[var_name]
                logger.info(weights)
                logger.info(weights[...])


if y_test is not None:
    correction_predictions = float(sum(all_predictions))
    logger.info("Total number of test examples: {}".format(len(y_test)))
    logger.info("Accuracy: {:g}".format(correction_predictions / float(len(y_test))))


predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
logger.info("Saving evaluation to {0}".format(out_path))

with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)





