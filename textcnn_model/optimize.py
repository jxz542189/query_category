import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from utils.optimize_util import convert_variables_to_constants


logger = set_logger(colored('optimize', 'cyan'), False)
tf.flags.DEFINE_string("checkpoint_dir", "../textcnn_model/runs/1587511511/checkpoints", "Checkpoint directory from training run")
FLAGS = tf.flags.FLAGS





graph = tf.Graph()
with graph.as_default():
    config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        tmp_g = graph.as_graph_def()
        input_tensors = []
        output_tensors = []

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        logger.info(input_x)
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        logger.info(dropout_keep_prob)

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        input_tensors.append(input_x)
        input_tensors.append(dropout_keep_prob)
        output_tensors.append(predictions)
        output_tensors.append(scores)
        logger.info('load parameters from checkpoint...')

        dtypes = [n.dtype for n in input_tensors]
        logger.info('optimize...')

        # 可以查看每个节点的信息
        # logger.info(tmp_g)
        # for node in tmp_g.node:
        #     logger.info(node.name)

        tmp_g = optimize_for_inference(
            tmp_g,
            [n.name[:-2] for n in input_tensors],
            [n.name[:-2] for n in output_tensors],
            [dtype.as_datatype_enum for dtype in dtypes],
            False
        )
        logger.info('freeze...')
        tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors],
                                               use_fp16=False)
        tmp_file = os.path.join(FLAGS.checkpoint_dir, "..", "model.pb")
        with tf.gfile.GFile(tmp_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())

