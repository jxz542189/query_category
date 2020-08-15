# from mlp_model.model import MLP
# from mlp_model.util import *
#
#
# logger = set_logger(colored('train', 'cyan'), False)
#
# tf.flags.DEFINE_float("dev_sample_percentage", 0.005, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("data_file", "data/train_level_3_601.csv", "data file")
#
# tf.flags.DEFINE_float("warmup_proportion", 0.1, "L2 regularization lambda (default: 0.0)")
# tf.flags.DEFINE_float("learning_rate", 5e-5, "learning_rate")
#
# tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# tf.flags.DEFINE_integer('early_stop_step', 2000, "early_stop_step")
# tf.flags.DEFINE_integer('num_layers', 1, "num_layers")
# tf.flags.DEFINE_integer('learning_rate_decay_num', 3, "learning_rate_decay_num")
#
#
# FLAGS = tf.flags.FLAGS
#
# level_3_cat = '601'
#
#
# def preprocess():
#     logger.info("Loading data...")
#     x_text, y, label2id, num_classes = load_data_and_labels(os.path.join(basedir, FLAGS.data_file), level=3)
#     print(f"num_classes: {num_classes}")
#     x = []
#     x = np.array(x_text)
#
#     np.random.seed(10)
#     shuffle_indices = np.random.permutation(np.arange(len(y)))
#     x_shuffled = x[shuffle_indices]
#     y = np.array(y)
#     y_shuffled = y[shuffle_indices]
#
#     dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#     x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#     y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
#
#     del x, y, x_shuffled, y_shuffled
#
#     logger.info("Vocabulary Size: {:d}".format(len(word2id)))
#     logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
#     return x_train, y_train,  x_dev,  y_dev, label2id
#
#
# def train(x_train, y_train, x_dev, y_dev, label2id):
#     acc = 0.0
#     with tf.Graph().as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=FLAGS.allow_soft_placement,
#             log_device_placement=FLAGS.log_device_placement
#         )
#
#         num_classes = len(label2id)
#         with tf.Session(config=session_conf) as sess:
#             cnn = MLP(num_classes=num_classes,
#                       embedding_size=1920,
#                       num_layers=FLAGS.num_layers)
#
#             global_step = tf.Variable(0, name="global_step", trainable=False)
#             num_train_steps = int(len(x_train) / FLAGS.batch_size * FLAGS.num_epochs)
#             num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
#             train_op = optimization.create_optimizer(
#                 cnn.loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, None, global_step)
#
#             timestamp = str(int(time.time()))
#             out_dir = os.path.abspath(os.path.join(basedir, "mlp_model", f"runs_{level_3_cat}", timestamp))
#             logger.info("Wrting to {}\n".format(out_dir))
#
#             checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
#             label_dir = os.path.join(out_dir, "label2id.m")
#             checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#             if not os.path.exists(checkpoint_dir):
#                 os.makedirs(checkpoint_dir)
#             saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
#             joblib.dump(label2id, label_dir)
#
#             sess.run(tf.global_variables_initializer())
#
#             def train_step(x_batch, y_batch):
#                 y = []
#                 for id in list(y_batch):
#                     tmp = [0] * num_classes
#                     tmp[int(id)] = 1
#                     y.append(tmp)
#                 res = estimator.predict(input_fn_builder(list(x_batch)), yield_single_examples=False)
#                 x = []
#                 for r in res:
#                     x = r['encoded_outputs']
#                     break
#
#                 x_batch = np.array(x, dtype=np.float32)
#
#                 y_batch_1 = np.array(y)
#
#                 feed_dict = {
#                     cnn.input_x: x_batch,
#                     cnn.input_y: y_batch_1
#                 }
#                 _, step, loss, accuracy, predictions = sess.run(
#                     [train_op, global_step, cnn.loss, cnn.accuracy, cnn.predictions],
#                     feed_dict
#                 )
#                 time_str = datetime.datetime.now().isoformat()
#                 logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#
#             def dev_step(x_batch, y_batch, writer=None):
#                 y = []
#                 for id in list(y_batch):
#                     tmp = [0] * num_classes
#                     tmp[int(id)] = 1
#                     y.append(tmp)
#
#                 res = estimator.predict(input_fn_builder(list(x_batch)), yield_single_examples=False)
#                 x = []
#                 for r in res:
#                     x = r['encoded_outputs']
#                     x_batch = np.array(x, dtype=np.float32)
#                     break
#
#                 y_batch = np.array(y)
#
#                 feed_dict = {
#                     cnn.input_x: x_batch,
#                     cnn.input_y: y_batch
#                 }
#                 step, loss, accuracy = sess.run(
#                     [global_step, cnn.loss, cnn.accuracy],
#                     feed_dict
#                 )
#
#                 return accuracy
#
#             batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
#
#             total_batch = 0
#             last_improved_batch = 0
#             learning_rate_decay_num = FLAGS.learning_rate_decay_num
#             flags = True
#             for batch in batches:
#                 total_batch += 1
#                 x_batch, y_batch = zip(*batch)
#                 train_step(x_batch, y_batch)
#                 current_step = tf.train.global_step(sess, global_step)
#                 if current_step % FLAGS.evaluate_every == 0:
#                     logger.info("{} Evaluation:".format(current_step))
#                     current_acc = dev_step(x_dev, y_dev, writer=None)
#                     logger.info("{} Evaluation, acc: {:g}".format(current_step, current_acc))
#                     if current_acc > acc:
#                         path = saver.save(sess, checkpoint_prefix, global_step=current_step)
#                         logger.info("Saved model checkpoint to {}\n".format(path))
#                         acc = current_acc
#                         last_improved_batch = total_batch
#                 if flags and total_batch - last_improved_batch > FLAGS.early_stop_step:
#                     if learning_rate_decay_num > 0:
#                         logger.info('No optimization for a long time, auto-stopping ...')
#                         last_improved_batch = total_batch
#                         FLAGS.learning_rate = FLAGS.learning_rate / 2
#                         logger.info('current learning_rate is {}'.format(FLAGS.learning_rate))
#                     learning_rate_decay_num -= 1
#                     if learning_rate_decay_num == 0:
#                         flags = False
#
#
# def main(argv=None):
#     x_train, y_train, x_dev, y_dev, label2id = preprocess()
#     train(x_train[:10000], y_train[:10000], x_dev[:3000], y_dev[3000], label2id)
#
#
# if __name__ == '__main__':
#     logger.info(basedir)
#     tf.app.run()
