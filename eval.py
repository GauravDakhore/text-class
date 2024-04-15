import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.keras import layers
import csv
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score
from absl import flags

# Parameters
# ==================================================

flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = tf.estimator.DNNClassifier.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        all_predictions = []
        all_probabilities = []

        for x_test_batch in batches:
            batch_predictions, batch_probabilities = sess.run([predictions, "output/probabilities:0"], {
                input_x: x_test_batch,
                dropout_keep_prob: 1.0
            })
            
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_probabilities = np.concatenate([all_probabilities, batch_probabilities])

# Print evaluation metrics
if y_test is not None:
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate F1 score
    f1 = f1_score(y_test, all_predictions)
    print("F1 Score:", f1)

    # Calculate AUC
    auc = roc_auc_score(y_test, all_probabilities[:, 1])  # Probabilities for positive class
    print("AUC:", auc)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, all_predictions)
    print("Accuracy:", accuracy)

# Save the evaluation to a csv file
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
