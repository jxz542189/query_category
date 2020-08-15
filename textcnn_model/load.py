import os
import tensorflow as tf
from tensorflow.python.platform import gfile

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.flags.DEFINE_string("checkpoint_dir", "../textcnn_model/runs/1587511511/checkpoints", "Checkpoint directory from training run")
FLAGS = tf.flags.FLAGS

model_file = os.path.join(FLAGS.checkpoint_dir, "..", "model.pb")
with gfile.FastGFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


input_x = sess.graph.get_operation_by_name("input_x").outputs[0]
dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
predictions = sess.graph.get_operation_by_name("output/predictions").outputs[0]
ret = sess.run(predictions, feed_dict={input_x: [[1, 1, 1,1, 1,1, 1,1 ,1, 1,1, 1, 1, 1, 1]], dropout_keep_prob: 1.0})
print(ret)
