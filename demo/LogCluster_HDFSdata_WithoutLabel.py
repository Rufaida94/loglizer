#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LogClustering
from loglizer import dataloader, preprocessing


struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = None # The anomaly label file
epsilon = 0.5 # threshold for estimating invariant space
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection

if __name__ == '__main__':
    # Load structured log without label info
    (x_train, _), (x_test, _) = dataloader.load_HDFS(struct_log,
                                                     window='session', 
                                                     train_ratio=0.5,
                                                     split_type='sequential')
    # Feature extraction
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')

    # Model initialization and training
    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(x_train) # Use only normal samples for training

    # Predict anomalies on the training set offline, and manually check for correctness
    y_train = model.predict(x_train)

    # Predict anomalies on the test set to simulate the online mode
    # x_test may be loaded from another log file
    x_test = feature_extractor.transform(x_test)
    y_test = model.predict(x_test)

    # If you have labeled data, you can evaluate the accuracy of the model as well.
    # Load structured log with label info

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))
