import tensorflow as tf
from termcolor import colored
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from utils.data_helpers import *
from tensorflow.contrib import learn

checkpoint_dir = "../textcnn_model/model_pb"

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

logger = set_logger(colored('eval', 'cyan'), False)
for category_id in ["646", "647", "651"]:
    print(category_id)
    FLAGS = tf.flags.FLAGS

    model_file = os.path.join(FLAGS.checkpoint_dir, f"model_{category_id}.pb")
    id2label = {}
    with open(os.path.join(FLAGS.checkpoint_dir, f"id2label_{category_id}.txt")) as f:
        for line in f:
            id, label = line.split("\u0001")
            id2label[int(id)] = re.sub("\n", "", label)
    print(f"类目数: {len(id2label)}")
    word2id = read_word2id(os.path.join("../", 'data', 'word2id.txt'))
    test_path = os.path.join('test', f'{category_id}.csv')

    with tf.gfile.GFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        input_names = ['input_x', 'dropout_keep_prob']


    def model_fn(features, labels, mode, params):
        predicts = tf.import_graph_def(graph_def,
                                     input_map={k + ":0": features[k] for k in input_names},
                                     return_elements=['output/predictions:0', 'output/scores:0'])
        return EstimatorSpec(mode=mode, predictions={
            "predict": predicts[0],
            "scores": predicts[1]
        })


    def get_estimator():
        config = tf.ConfigProto(device_count={'GPU': 0})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.log_device_placement = False

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config),
                         params=None)


    def input_fn_builder(lines):
        def gen():
            inputs_x = []
            for line in lines:
                inputs_x.append(process(line, word2id, max_sequence_length=10))

            yield {
                'input_x': inputs_x,
                'dropout_keep_prob': [1.0 for _ in range(len(inputs_x))]
            }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_x': tf.int32,
                              'dropout_keep_prob': tf.float32},
                output_shapes={
                    'input_x': (None, None),
                    'dropout_keep_prob':(None)
                }
            ).prefetch(tf.data.experimental.AUTOTUNE))

        return input_fn


    estimator = get_estimator()

    df = pd.read_csv(f"./data/train_level_3_{category_id}_part.csv", header=None)
    df = df.head(1000)
    df.columns = ['query', 'label']
    records = df.to_dict('records')
    new_records = []
    i = 0
    for record in records:
        if i % 100 == 0:
            print(i)
        result = estimator.predict(input_fn_builder([record['query']]), yield_single_examples=False)
        for r in result:
            record['predict'] = int(id2label[int(r['predict'][0])])
            new_records.append(record)
            # print(r['predict'])
            # print(r['scores'])
        i += 1

    df = pd.DataFrame(new_records)
    print(df.head())
    df.to_csv(test_path, index=None)


