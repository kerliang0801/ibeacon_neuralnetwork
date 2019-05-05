
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets
IBEACON_TRAINING = "ibeacon_training.csv"
IBEACON_TEST = "ibeacon_testing.csv"

def main():
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=IBEACON_TRAINING,
      target_dtype=np.int,
      features_dtype=np.int)
  
  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=IBEACON_TEST,
      target_dtype=np.int,
      features_dtype=np.int)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 4 layer DNN
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[32,128,64,32],
                                              n_classes=4)
  # Define the training inputs

  print ("Loaded dataset\n")

  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y
  print ("Got train inputs\n")
  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=8000)
  print ("Fitted model\n")

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y
  print ("Got test inputs\n")

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify new flower
  #def new_samples():
   # return np.array([[64, 27, 56, 21]], dtype=np.int)

  #predictions = list(classifier.predict(input_fn=new_samples))

  #print("Predicted class: {}\n".format(predictions))

if __name__ == "__main__":
    main()