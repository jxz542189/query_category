import numpy as np
from utils import tokenization, bert_33
import tensorflow as tf
from tensorflow.python.platform import gfile


def freeze_graph(input_checkpoint, output_graph):
    '''
      :param input_checkpoint:
      :param output_graph: PB模型保存路径
      :return:
      '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "probes/probe_dense_12/bias/Adam_1"
    saver = tf.train.import_meta_graph(
        input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        for var in tf.global_variables():
            print(var)
            print(var.name)
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print(output_graph_def.node)
        print("%d ops in the final graph." %
              len(output_graph_def.node))


def load_pb(file):
    with tf.Session() as sess:
        print("load graph...")
        with tf.gfile.GFile(file, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes = [n for n in graph_def.node]
        for t in graph_nodes:
            print(t.name)


def save2pb(save_model_dir, output_node, pb_path):
    '''
    将pb模型变量版变为pb常量版
    :param save_model_dir:
    :param output_node:
    :param pb_path:
    :return:
    '''
    # save_model_dir = "/home/jixiaozhan/EasyTransfer/scripts/knowledge_distillation/vanilla_teacher_model/1607392874"
    # output_node = ["app/ez_dense/BiasAdd"]
    # pb_path = "/home/jixiaozhan/EasyTransfer/scripts/knowledge_distillation/vanilla_teacher_model/tmp_model/saved_model.pb"
    with tf.Session(graph=tf.Graph()) as sess, tf.device("/cpu:0"):
        meta_graph_def = tf.saved_model.loader.load(sess, ['serve'], save_model_dir)
        nodes = [n.name for n in meta_graph_def.graph_def.node]
        #     for var in nodes:
        #         print(var)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=meta_graph_def.graph_def,
            output_node_names=output_node)
        with tf.gfile.GFile(pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("%d ops in the final graph." % len(output_graph_def.node))
        for node in output_graph_def.node:
            print(node.name)


def predict():
    """
    加载pb变量模型
    :return:
    """
    VOCAB_PATH_HZ = '/home/recsys/jixiaozhan/sansu_detect_bert/modelParams/chinese_L-12_H-768_A-12/vocab.txt'
    title = "hide new secretions from the parental units"
    MODEL_V2 = "/home/jixiaozhan/EasyTransfer/scripts/knowledge_distillation/vanilla_teacher_model/1607392874"
    tokenizer_hz = tokenization.FullTokenizer(vocab_file=VOCAB_PATH_HZ, do_lower_case=True)
    example = bert_33.get_input_features(title, tokenizer_hz)
    label_id = example.pop('label_ids')
    example['label_id'] = label_id

    predict_fn_hz_v2 = tf.contrib.predictor.from_saved_model(MODEL_V2)
    predict_pro_list2 = predict_fn_hz_v2(example)

    # print(softmax(predict_pro_list2['logits']))


def predict_v2():
    """
    加载pb常量模型
    :return:
    """
    VOCAB_PATH_HZ = '/home/recsys/jixiaozhan/sansu_detect_bert/modelParams/chinese_L-12_H-768_A-12/vocab.txt'
    title = "hide new secretions from the parental units"
    model_file = "/home/jixiaozhan/EasyTransfer/scripts/knowledge_distillation/vanilla_teacher_model/tmp_model/saved_model.pb"
    tokenizer_hz = tokenization.FullTokenizer(vocab_file=VOCAB_PATH_HZ, do_lower_case=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    input_ids = sess.graph.get_tensor_by_name("input_ids:0")
    input_mask = sess.graph.get_tensor_by_name("input_mask:0")
    segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")
    predictions = sess.graph.get_tensor_by_name('app/ez_dense/BiasAdd:0')[0]
    example = bert_33.get_input_features(title, tokenizer_hz)
    ret = sess.run(predictions, feed_dict={
        input_ids:np.array(example['input_ids']),
        input_mask:np.array(example['input_mask']),
        segment_ids: np.array(example['segment_ids'])
    })