import os
import shutil
from datetime import datetime
import tensorflow as tf
from termcolor import colored
from utils.data_helpers import *
from mlp_model.model import *
from utils import optimization

RESUME_TRAINING = False
logger = set_logger(colored('train', 'cyan'), False)
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def model_fn_builder(num_train_steps, num_warmup_steps, global_step):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_x = features["input_x"]
        input_y = features["input_y"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = MLP(num_classes=params, input_x=input_x, input_y=input_y)

        if mode == tf.estimator.ModeKeys.PREDICT:

            predictions = {
                'scores': model.scores
            }

            export_outputs = {
                'scores' : tf.estimator.export.PredictOutput(predictions)
            }

            return tf.estimator.EstimatorSpec(mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                model.loss, params.learning_rate, num_train_steps, num_warmup_steps, None, global_step)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=model.loss,
                                              train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            labels_one_hot = tf.one_hot(
                labels,
                depth=params.num_classes,
                on_value=True,
                off_value=False,
                dtype=tf.bool
            )

            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels, model.predictions),
                'auroc': tf.metrics.auc(labels_one_hot, model.scores)
            }

            return tf.estimator.EstimatorSpec(mode,
                                              loss=model.loss,
                                              eval_metric_ops=eval_metric_ops)


    return model_fn


def creater_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model_fn_builder(hparams.num_train_steps,
                                                                 hparams.num_warmup_steps,
                                                                 global_step=global_step))
    logger.info("")
    logger.info("Estimator Type: {}".format(type(estimator)))
    logger.info("")

    return estimator


def serving_input_fn():
    receiver_tensor = {
        "input_x": tf.placeholder(tf.float32, [None, 1920])
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


def file_based_input_fn_builder(input_file, embedding_size, is_training, batch_size, drop_remainder=False):
    name_to_features = {
        "input_x": tf.FixedLenFeature([embedding_size], tf.float32),
        "input_y": tf.FixedLenFeature([], tf.int32)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        return example

    def input_fn():
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder
            )
        )
        return d
    return input_fn


def input_fn_builder(features, embedding_size, is_training, batch_size, drop_remainder):
    all_input_x = []
    all_input_y = []

    for feature in features:
        all_input_x.append(feature.input_x)
        all_input_y.append(feature.input_y)

    def input_fn():
        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_x":
                tf.constant(
                    all_input_x, shape=[num_examples, embedding_size],
                    dtype=tf.float32
                ),
            "input_y":
                tf.constant(
                    all_input_y, shape=[num_examples], dtype=tf.int32
                )
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d
    return input_fn


if __name__ == '__main__':
    cate_level_3 = '601'
    df = pd.read_csv(os.path.join(basedir, 'data', f'train_level_3_{cate_level_3}.csv'), header=None)
    records = df.to_dict('records')
    labels = []
    for record in records:
        labels.append(record[1])
    num_classes = len(set(labels))
    num_epochs = 20
    batch_size = 32
    num_train_steps = int(len(labels) / batch_size * num_epochs)

    hparams = tf.contrib.training.HParams(num_epochs=num_epochs,
                                          batch_size=batch_size,
                                          max_steps=num_train_steps,
                                          learning_rate=0.01,
                                          num_classes=num_classes,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=0.1 * num_train_steps,
                                          embedding_size=1920)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    model_dir = os.path.join(basedir, 'mlp_model', f'run_{cate_level_3}')

    run_config = tf.estimator.RunConfig(log_step_count_steps=5000,
                                        tf_random_seed=19830610,
                                        model_dir=model_dir)

    estimator = creater_estimator(run_config, hparams)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda : file_based_input_fn_builder(
            input_file=os.path.join(basedir, 'data', f'train_level_3_{cate_level_3}.tfrecords'),
            embedding_size=hparams.embedding_size,
            is_training=True,
            batch_size=hparams.batch_size
        ),
        max_steps=hparams.max_steps,
        hooks=None
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: file_based_input_fn_builder(
            input_file=os.path.join(basedir, 'data', f'train_level_3_{cate_level_3}.tfrecords'),
            embedding_size=hparams.embedding_size,
            is_training=False,
            batch_size=batch_size
        ),
        exporters=[tf.estimator.LatestExporter(
            name='predict',
            serving_input_receiver_fn=serving_input_fn,
            exports_to_keep=3,
            as_text=True
        )],
        steps=None,
        throttle_secs=60
    )

    if not RESUME_TRAINING:
        logger.info("Removing previous artifacts...")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        logger.info("Resuming training...")

    time_start = datetime.utcnow()
    logger.info("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))

    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)
    time_end = datetime.utcnow()
    logger.info(".......................................")
    logger.info("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    logger.info("")
    time_elapsed = time_end - time_start
    logger.info("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
