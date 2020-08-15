import h5py
import csv
import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from tensorflow.contrib import learn


logger = set_logger(colored('bert_weights', 'cyan'), False)

tf.flags.DEFINE_string("checkpoint_dir", "./data/uncased_L-12_H-768_A-12", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print(FLAGS.checkpoint_dir)

# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
checkpoint_file = os.path.join(FLAGS.checkpoint_dir, "bert_model.ckpt")
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

        weight_file = os.path.join(FLAGS.checkpoint_dir, "dump_weights.hdf5")
        if os.path.isfile(weight_file):
            os.remove(weight_file)
        with h5py.File(weight_file) as fout:
            for var in tf.trainable_variables():
                var_name = re.sub(":0$", "", var.name)
                logger.info("Saving variable {0} with name {1}".format(
                    var.name, var_name))
                shape = var.get_shape().as_list()
                dset = fout.create_dataset(var_name, shape, dtype='float32')
                values = sess.run([var])[0]
                dset[...] = values