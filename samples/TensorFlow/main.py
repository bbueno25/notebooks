from os import environ as os_environ
from google.cloud import bigquery
from google.oauth2 import service_account
from matplotlib import pyplot as plt
from numpy import log as np_log
from pandas import DataFrame as pd_DataFrame
from pandas import concat as pd_concat
from pandas.plotting import autocorrelation_plot, scatter_matrix
from tensorflow import Session as tf_Session
from tensorflow import Variable as tf_Variable
from tensorflow import argmax as tf_argmax
from tensorflow import cast as tf_cast
from tensorflow import equal as tf_equal
from tensorflow import initialize_all_variables as tf_initialize_all_variables
from tensorflow import log as tf_log
from tensorflow import logical_and as tf_logical_and
from tensorflow import matmul as tf_matmul
from tensorflow import nn as tf_nn
from tensorflow import ones as tf_ones
from tensorflow import ones_like as tf_ones_like
from tensorflow import placeholder as tf_placeholder
from tensorflow import reduce_mean as tf_reduce_mean
from tensorflow import reduce_sum as tf_reduce_sum
from tensorflow import train as tf_train
from tensorflow import truncated_normal as tf_truncated_normal
from tensorflow import zeros_like as tf_zeros_like


def collect_data(json_path, project_id):
    credentials = service_account.Credentials.from_service_account_file(json_path)
    bq_client = bigquery.Client(credentials=credentials, project=project_id)
    aord = bq_client.query("SELECT ['Date', 'Close'] FROM bigquery.Table('bingo-ml-1.market_data.aord')").execute().result().to_dataframe().set_index('Date')
    dax = bigquery.QueryJob.from_table(bigquery.Table('bingo-ml-1.market_data.dax'),
                              ['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
    djia = bigquery.Query.from_table(bigquery.Table('bingo-ml-1.market_data.djia'),
                               ['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
    ftse = bigquery.Query.from_table(bigquery.Table('bingo-ml-1.market_data.ftse'),
                               ['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
    hangseng = bigquery.Query.from_table(bigquery.Table('bingo-ml-1.market_data.hangseng'),
                                   ['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
    nikkei = bigquery.Query.from_table(bigquery.Table('bingo-ml-1.market_data.nikkei'),
                                 ['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
    nyse = bigquery.Query.from_table(bigquery.Table('bingo-ml-1.market_data.nyse'),
                               ['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
    snp = bigquery.Query.from_table(bigquery.Table('bingo-ml-1.market_data.snp'),
                              ['Date', 'Close']).execute().result().to_dataframe().set_index('Date')
    return aord, dax, djia, ftse, hangseng, nikkei, nyse, snp


def confusion_metrics_1L(actual_classes, feature_data, model, sess, test_classes_tf, test_predictors_tf):
    """
    CONFUSION MATRIX - SINGLE LAYER

    The metrics for this most simple of TensorFlow models are unimpressive,
    an F1 Score of 0.36 is not going to blow any light bulbs in the room.
    That's partly because of its simplicity and partly because It hasn't been tuned;
    selection of hyperparameters is very important in machine learning modelling.
    """
    feed_dict = {feature_data: test_predictors_tf.values,
                 actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)}
    tf_confusion_metrics(model, actual_classes, sess, feed_dict)
    #
    # Precision = 0.905660377358
    # Recall = 0.780487804878
    # F1 Score = 0.838427947598
    # Accuracy = 0.871527777778


def confusion_metrics_2HL(actual_classes, feature_data, model, sess1, test_classes_tf, test_predictors_tf):
    """
    CONFUSION METRICS - FEED-FORWARD NEURAL NETWORK WITH TWO HIDDEN LAYERS

    Looking at precision, recall, and accuracy,
    you can see a measurable improvement in performance, but certainly not a step function.
    This indicates that we're likely reaching the limits of this relatively simple feature set.
    """
    feed_dict = {feature_data: test_predictors_tf.values,
                 actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)}
    tf_confusion_metrics(model, actual_classes, sess1, feed_dict)
    #
    # Precision = 0.921052631579
    # Recall = 0.853658536585
    # F1 Score = 0.886075949367
    # Accuracy = 0.90625


def correlation_plot(closing_data):
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(15)
    _ = autocorrelation_plot(closing_data['aord_close'], label='aord_close')
    _ = autocorrelation_plot(closing_data['dax_close'], label='dax_close')
    _ = autocorrelation_plot(closing_data['djia_close'], label='djia_close')
    _ = autocorrelation_plot(closing_data['ftse_close'], label='ftse_close')
    _ = autocorrelation_plot(closing_data['hangseng_close'], label='hangseng_close')
    _ = autocorrelation_plot(closing_data['nikkei_close'], label='nikkei_close')
    _ = autocorrelation_plot(closing_data['nyse_close'], label='nyse_close')
    _ = autocorrelation_plot(closing_data['snp_close'], label='snp_close')
    _ = plt.legend(loc='upper right')
    plt.show()


def create_model_1L(training_classes_tf, training_predictors_tf):
    """ CREATE MODEL - SINGLE LAYER """
    sess = tf_Session()
    # Define variables for the number of predictors and number of classes to remove magic numbers from our code.
    num_predictors = len(training_predictors_tf.columns)    # 24 in the default case
    num_classes = len(training_classes_tf.columns)    # 2 in the default case
    # Define placeholders for the data we feed into the process - feature data and actual classes.
    feature_data = tf_placeholder("float", [None, num_predictors])
    actual_classes = tf_placeholder("float", [None, num_classes])
    # Define a matrix of weights and initialize it with some small random values.
    weights = tf_Variable(tf_truncated_normal([num_predictors, num_classes], stddev=0.0001))
    biases = tf_Variable(tf_ones([num_classes]))
    # Define the model
    # Here we take a softmax regression of the product of our feature data and weights.
    model = tf_nn.softmax(tf_matmul(feature_data, weights) + biases)
    # Define a cost function (we're using the cross entropy).
    cost = -tf_reduce_sum(actual_classes * tf_log(model))
    # Define a training step
    # Here we use gradient descent with a learning rate of 0.01 using the cost function we just defined.
    training_step = tf_train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    init = tf_initialize_all_variables()
    sess.run(init)
    return actual_classes, feature_data, model, sess, training_step


def create_model_2HL(training_classes_tf, training_predictors_tf):
    """
    CREATE MODEL - FEED-FORWARD NEURAL NETWORK WITH TWO HIDDEN LAYERS

    Again, you'll train the model over 30,000 iterations
    using the full dataset each time.
    Every thousandth iteration, you'll assess the accuracy
    of the model on the training data to assess progress.
    """
    sess1 = tf_Session()
    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)
    feature_data = tf_placeholder("float", [None, num_predictors])
    actual_classes = tf_placeholder("float", [None, 2])
    weights1 = tf_Variable(tf_truncated_normal([24, 50], stddev=0.0001))
    biases1 = tf_Variable(tf_ones([50]))
    weights2 = tf_Variable(tf_truncated_normal([50, 25], stddev=0.0001))
    biases2 = tf_Variable(tf_ones([25]))
    weights3 = tf_Variable(tf_truncated_normal([25, 2], stddev=0.0001))
    biases3 = tf_Variable(tf_ones([2]))
    hidden_layer_1 = tf_nn.relu(tf_matmul(feature_data, weights1) + biases1)
    hidden_layer_2 = tf_nn.relu(tf_matmul(hidden_layer_1, weights2) + biases2)
    model = tf_nn.softmax(tf_matmul(hidden_layer_2, weights3) + biases3)
    cost = -tf_reduce_sum(actual_classes * tf_log(model))
    train_op1 = tf_train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    init = tf_initialize_all_variables()
    sess1.run(init)
    return actual_classes, feature_data, model, sess1, train_op1


def describe_data(aord, dax, djia, ftse, hangseng, nikkei, nyse, snp):
    closing_data = pd_DataFrame()
    closing_data['aord_close'] = aord['Close']
    closing_data['dax_close'] = dax['Close']
    closing_data['djia_close'] = djia['Close']
    closing_data['ftse_close'] = ftse['Close']
    closing_data['hangseng_close'] = hangseng['Close']
    closing_data['nikkei_close'] = nikkei['Close']
    closing_data['nyse_close'] = nyse['Close']
    closing_data['snp_close'] = snp['Close']
    # Pandas includes a very convenient function for filling gaps in the data.
    closing_data = closing_data.fillna(method='ffill')
    closing_data.describe()
    return closing_data


def log_correlations_Dmin0(log_return_data):
    """
    LOG CORRELATIONS - SAME DAY

    Here, we are directly working with the premise.
    We're correlating the close of the S&P 500 with signals available before the close of the S&P 500.
    And you can see that the S&P 500 close is correlated with European indices at around 0.65 for the FTSE and DAX,
    which is a strong correlation, and Asian/Oceanian indices at around 0.15-0.22,
    which is a significant correlation, but not with US indices.
    """
    tmp = pd_DataFrame()
    tmp['aord_0'] = log_return_data['aord_log_return']
    tmp['dax_0'] = log_return_data['dax_log_return']
    tmp['djia_1'] = log_return_data['djia_log_return'].shift()
    tmp['ftse_0'] = log_return_data['ftse_log_return']
    tmp['hangseng_0'] = log_return_data['hangseng_log_return']
    tmp['nikkei_0'] = log_return_data['nikkei_log_return']
    tmp['nyse_1'] = log_return_data['nyse_log_return'].shift()
    tmp['snp_0'] = log_return_data['snp_log_return']
    tmp.corr().iloc[:, 0]
    #
    # snp_0         1.000000
    # nyse_1       -0.496712
    # djia_1       -0.511219
    # ftse_0        0.910895
    # dax_0         0.949037
    # hangseng_0    0.573571
    # nikkei_0      0.847618
    # aord_0        0.792358
    # Name: snp_0, dtype: float64


def log_correlations_Dmin1(log_return_data):
    """
    LOG CORRELATIONS - PREVIOUS DAY

    Now look at how the log returns for the S&P closing values correlate with index values
    from the previous day to see if the previous closing is predictive.
    Following from the premise that financial markets are Markov processes,
    there should be little or no value in historical values.
    You should see little to no correlation in this data,
    meaning that yesterday's values are no practical help in predicting today's close.
    """
    tmp = pd_DataFrame()
    tmp['aord_0'] = log_return_data['aord_log_return'].shift()
    tmp['dax_0'] = log_return_data['dax_log_return'].shift()
    tmp['djia_1'] = log_return_data['djia_log_return'].shift(2)
    tmp['ftse_0'] = log_return_data['ftse_log_return'].shift()
    tmp['hangseng_0'] = log_return_data['hangseng_log_return'].shift()
    tmp['nikkei_0'] = log_return_data['nikkei_log_return'].shift()
    tmp['nyse_1'] = log_return_data['nyse_log_return'].shift(2)
    tmp['snp_0'] = log_return_data['snp_log_return']
    tmp.corr().iloc[:, 0]
    #
    # snp_0         1.000000
    # nyse_1        0.059031
    # djia_1        0.071371
    # ftse_0       -0.451623
    # dax_0        -0.481275
    # hangseng_0   -0.302121
    # nikkei_0     -0.434308
    # aord_0       -0.382160
    # Name: snp_0, dtype: float64


def log_correlations_Dmin2(log_return_data):
    """
    LOG CORRELATIONS - DAY BEFORE YESTERDAY

    Let's go one step further and look at correlations between today and the the day before yesterday.
    Again, there are little to no correlations.
    """
    tmp = pd_DataFrame()
    tmp['aord_0'] = log_return_data['aord_log_return'].shift(2)
    tmp['dax_0'] = log_return_data['dax_log_return'].shift(2)
    tmp['djia_1'] = log_return_data['djia_log_return'].shift(3)
    tmp['ftse_0'] = log_return_data['ftse_log_return'].shift(2)
    tmp['hangseng_0'] = log_return_data['hangseng_log_return'].shift(2)
    tmp['nikkei_0'] = log_return_data['nikkei_log_return'].shift(2)
    tmp['nyse_1'] = log_return_data['nyse_log_return'].shift(3)
    tmp['snp_0'] = log_return_data['snp_log_return']
    tmp.corr().iloc[:, 0]
    #
    # snp_0         1.000000
    # nyse_1       -0.064236
    # djia_1       -0.069342
    # ftse_0        0.047785
    # dax_0         0.051619
    # hangseng_0    0.035143
    # nikkei_0      0.050108
    # aord_0        0.023883
    # Name: snp_0, dtype: float64


def log_returns(closing_data):
    """
    Looking at the log returns, you should see that the mean, min, max are all similar.
    You could go further and center the series on zero, scale them, and normalize the standard deviation,
    but there's no need to do that at this point.
    """
    log_return_data = pd_DataFrame()
    log_return_data['aord_log_return'] = np_log(closing_data['aord_close'] / closing_data['aord_close'].shift())
    log_return_data['dax_log_return'] = np_log(closing_data['dax_close'] / closing_data['dax_close'].shift())
    log_return_data['djia_log_return'] = np_log(closing_data['djia_close'] / closing_data['djia_close'].shift())
    log_return_data['ftse_log_return'] = np_log(closing_data['ftse_close'] / closing_data['ftse_close'].shift())
    log_return_data['hangseng_log_return'] = np_log(closing_data['hangseng_close'] / closing_data['hangseng_close'].shift())
    log_return_data['nikkei_log_return'] = np_log(closing_data['nikkei_close'] / closing_data['nikkei_close'].shift())
    log_return_data['nyse_log_return'] = np_log(closing_data['nyse_close'] / closing_data['nyse_close'].shift())
    log_return_data['snp_log_return'] = np_log(closing_data['snp_close'] / closing_data['snp_close'].shift())
    log_return_data.describe()
    return log_return_data


def log_returns_correlation_plot(log_return_data):
    """
    No autocorrelations are visible in the plot, which is what we're looking for.
    Individual financial markets are Markov processes - knowledge of history doesn't allow you to predict the future.
    You now have time series for the indices, stationary in the mean, similarly centered and scaled.
    Now start to look for signals to try to predict the close of the S&P 500.
    """
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(15)
    _ = autocorrelation_plot(log_return_data['aord_log_return'], label='aord_log_return')
    _ = autocorrelation_plot(log_return_data['dax_log_return'], label='dax_log_return')
    _ = autocorrelation_plot(log_return_data['djia_log_return'], label='djia_log_return')
    _ = autocorrelation_plot(log_return_data['ftse_log_return'], label='ftse_log_return')
    _ = autocorrelation_plot(log_return_data['hangseng_log_return'], label='hangseng_log_return')
    _ = autocorrelation_plot(log_return_data['nikkei_log_return'], label='nikkei_log_return')
    _ = autocorrelation_plot(log_return_data['nyse_log_return'], label='nyse_log_return')
    _ = autocorrelation_plot(log_return_data['snp_log_return'], label='snp_log_return')
    _ = plt.legend(loc='upper right')
    plt.show()


def log_returns_plot(log_return_data):
    _ = pd_concat([log_return_data['aord_log_return'],
                   log_return_data['dax_log_return'],
                   log_return_data['djia_log_return'],
                   log_return_data['ftse_log_return'],
                   log_return_data['hangseng_log_return'],
                   log_return_data['nikkei_log_return'],
                   log_return_data['nyse_log_return'],
                   log_return_data['snp_log_return']],
                   axis=1).plot(figsize=(20, 15))
    plt.show()


def log_returns_scatter_plot(log_return_data):
    """
    The story with the previous scatter plot for log returns is more subtle and more interesting.
    The US indices are strongly correlated, as expected.
    The other indices, less so, which is also expected, but there is structure and signal there.
    """
    _ = scatter_matrix(log_return_data, figsize=(20, 20), diagonal='kde')


def plot_data(closing_data):
    _ = pd_concat([closing_data['aord_close'],
                   closing_data['dax_close '],
                   closing_data['djia_close'],
                   closing_data['ftse_close'],
                   closing_data['hangseng_close'],
                   closing_data['nikkei_close'],
                   closing_data['nyse_close'],
                   closing_data['snp_close']],
                   axis=1).plot(figsize=(20, 15))
    plt.show()


def plot_scaled_data(closing_data):
    _ = pd_concat([closing_data['aord_close_scaled'],
                   closing_data['dax_close_scaled'],
                   closing_data['djia_close_scaled'],
                   closing_data['ftse_close_scaled'],
                   closing_data['hangseng_close_scaled'],
                   closing_data['nikkei_close_scaled'],
                   closing_data['nyse_close_scaled'],
                   closing_data['snp_close_scaled']],
                   axis=1).plot(figsize=(20, 15))
    plt.show()


def scale_data(closing_data):
    closing_data['aord_close_scaled'] = closing_data['aord_close'] / max(closing_data['aord_close'])
    closing_data['dax_close_scaled'] = closing_data['dax_close'] / max(closing_data['dax_close'])
    closing_data['djia_close_scaled'] = closing_data['djia_close'] / max(closing_data['djia_close'])
    closing_data['ftse_close_scaled'] = closing_data['ftse_close'] / max(closing_data['ftse_close'])
    closing_data['hangseng_close_scaled'] = closing_data['hangseng_close'] / max(closing_data['hangseng_close'])
    closing_data['nikkei_close_scaled'] = closing_data['nikkei_close'] / max(closing_data['nikkei_close'])
    closing_data['nyse_close_scaled'] = closing_data['nyse_close'] / max(closing_data['nyse_close'])
    closing_data['snp_close_scaled'] = closing_data['snp_close'] / max(closing_data['snp_close'])


def scatter_plot(closing_data):
    _ = scatter_matrix(pd_concat([closing_data['aord_close_scaled'],
                                  closing_data['dax_close_scaled'],
                                  closing_data['djia_close_scaled'],
                                  closing_data['ftse_close_scaled'],
                                  closing_data['hangseng_close_scaled'],
                                  closing_data['nikkei_close_scaled'],
                                  closing_data['nyse_close_scaled'],
                                  closing_data['snp_close_scaled']],
                                  axis=1), figsize=(20, 20), diagonal='kde')
    plt.show()


def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    predictions = tf_argmax(model, 1)
    actuals = tf_argmax(actual_classes, 1)
    ones_like_actuals = tf_ones_like(actuals)
    zeros_like_actuals = tf_zeros_like(actuals)
    ones_like_predictions = tf_ones_like(predictions)
    zeros_like_predictions = tf_zeros_like(predictions)
    tp_op = tf_reduce_sum(tf_cast(tf_logical_and(tf_equal(actuals, ones_like_actuals),
                                                 tf_equal(predictions, ones_like_predictions)), "float"))
    tn_op = tf_reduce_sum(tf_cast(tf_logical_and(tf_equal(actuals, zeros_like_actuals),
                                                 tf_equal(predictions, zeros_like_predictions)), "float"))
    fp_op = tf_reduce_sum(tf_cast(tf_logical_and(tf_equal(actuals, zeros_like_actuals),
                                                 tf_equal(predictions, ones_like_predictions)), "float"))
    fn_op = tf_reduce_sum(tf_cast(tf_logical_and(tf_equal(actuals, ones_like_actuals),
                                                 tf_equal(predictions, zeros_like_predictions)), "float"))
    tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
    tpfn = float(tp) + float(fn)
    tpr = 0 if tpfn == 0 else float(tp) / tpfn
    fpr = 0 if tpfn == 0 else float(fp) / tpfn
    total = float(tp) + float(fp) + float(fn) + float(tn)
    accuracy = 0 if total == 0 else (float(tp) + float(tn)) / total
    recall = tpr
    tpfp = float(tp) + float(fp)
    precision = 0 if tpfp == 0 else float(tp) / tpfp
    f1_score = 0 if recall == 0 else (2 * (precision * recall)) / (precision + recall)
    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1 Score = ', f1_score)
    print('Accuracy = ', accuracy)


def train_model_1L(actual_classes, feature_data, model, sess, training_classes_tf, training_predictors_tf, training_step):
    """
    TRAIN MODEL - SINGLE LAYER

    You'll train the model over 30,000 iterations using the full dataset each time.
    Every thousandth iteration we'll assess the accuracy of the model on the training data to assess progress.
    An accuracy of 65% on the training data is fine, certainly better than random.
    """
    correct_prediction = tf_equal(tf_argmax(model, 1), tf_argmax(actual_classes, 1))
    accuracy = tf_reduce_mean(tf_cast(correct_prediction, "float"))
    for i in range(1, 30001):
        sess.run(training_step,
                 feed_dict={feature_data: training_predictors_tf.values,
                            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)})
        if i % 5000 == 0:
            acc = sess.run(accuracy,
                           feed_dict={feature_data: training_predictors_tf.values,
                                      actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)})
            print(i, acc)
    #
    # 5000 0.809896
    # 10000 0.859375
    # 15000 0.881076
    # 20000 0.891493
    # 25000 0.896701
    # 30000 0.904514


def train_model_2HL(actual_classes, feature_data, model, sess1, train_op1, training_classes_tf, training_predictors_tf):
    """
    TRAIN MODEL - FEED-FORWARD NEURAL NETWORK WITH TWO HIDDEN LAYERS

    A significant improvement in accuracy  with the training data shows that the
    hidden layers are adding additional capacity for learning to the model.
    """
    correct_prediction = tf_equal(tf_argmax(model, 1), tf_argmax(actual_classes, 1))
    accuracy = tf_reduce_mean(tf_cast(correct_prediction, "float"))
    for i in range(1, 30001):
        sess1.run(train_op1,
                  feed_dict={feature_data: training_predictors_tf.values,
                             actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)})
        if i % 5000 == 0:
            acc = sess1.run(accuracy,
                            feed_dict={feature_data: training_predictors_tf.values,
                                       actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)})
            print(i, acc)
    #
    # 5000 0.931424
    # 10000 0.934028
    # 15000 0.934028
    # 20000 0.934028
    # 25000 0.934028
    # 30000 0.934028


def training_data(log_return_data):
    log_return_data['snp_log_return_positive'] = 0
    log_return_data.ix[log_return_data['snp_log_return'] >= 0, 'snp_log_return_positive'] = 1
    log_return_data['snp_log_return_negative'] = 0
    log_return_data.ix[log_return_data['snp_log_return'] < 0, 'snp_log_return_negative'] = 1
    training_test_data = pd_DataFrame(columns=['snp_log_return_positive', 'snp_log_return_negative',
                                               'snp_log_return_1', 'snp_log_return_2', 'snp_log_return_3',
                                               'nyse_log_return_1', 'nyse_log_return_2', 'nyse_log_return_3',
                                               'djia_log_return_1', 'djia_log_return_2', 'djia_log_return_3',
                                               'nikkei_log_return_0', 'nikkei_log_return_1', 'nikkei_log_return_2',
                                               'hangseng_log_return_0', 'hangseng_log_return_1', 'hangseng_log_return_2',
                                               'ftse_log_return_0', 'ftse_log_return_1', 'ftse_log_return_2',
                                               'dax_log_return_0', 'dax_log_return_1', 'dax_log_return_2',
                                               'aord_log_return_0', 'aord_log_return_1', 'aord_log_return_2'])
    for i in range(7, len(log_return_data)):
        snp_log_return_positive = log_return_data['snp_log_return_positive'].ix[i]
        snp_log_return_negative = log_return_data['snp_log_return_negative'].ix[i]
        snp_log_return_1 = log_return_data['snp_log_return'].ix[i - 1]
        snp_log_return_2 = log_return_data['snp_log_return'].ix[i - 2]
        snp_log_return_3 = log_return_data['snp_log_return'].ix[i - 3]
        nyse_log_return_1 = log_return_data['nyse_log_return'].ix[i - 1]
        nyse_log_return_2 = log_return_data['nyse_log_return'].ix[i - 2]
        nyse_log_return_3 = log_return_data['nyse_log_return'].ix[i - 3]
        djia_log_return_1 = log_return_data['djia_log_return'].ix[i - 1]
        djia_log_return_2 = log_return_data['djia_log_return'].ix[i - 2]
        djia_log_return_3 = log_return_data['djia_log_return'].ix[i - 3]
        nikkei_log_return_0 = log_return_data['nikkei_log_return'].ix[i]
        nikkei_log_return_1 = log_return_data['nikkei_log_return'].ix[i - 1]
        nikkei_log_return_2 = log_return_data['nikkei_log_return'].ix[i - 2]
        hangseng_log_return_0 = log_return_data['hangseng_log_return'].ix[i]
        hangseng_log_return_1 = log_return_data['hangseng_log_return'].ix[i - 1]
        hangseng_log_return_2 = log_return_data['hangseng_log_return'].ix[i - 2]
        ftse_log_return_0 = log_return_data['ftse_log_return'].ix[i]
        ftse_log_return_1 = log_return_data['ftse_log_return'].ix[i - 1]
        ftse_log_return_2 = log_return_data['ftse_log_return'].ix[i - 2]
        dax_log_return_0 = log_return_data['dax_log_return'].ix[i]
        dax_log_return_1 = log_return_data['dax_log_return'].ix[i - 1]
        dax_log_return_2 = log_return_data['dax_log_return'].ix[i - 2]
        aord_log_return_0 = log_return_data['aord_log_return'].ix[i]
        aord_log_return_1 = log_return_data['aord_log_return'].ix[i - 1]
        aord_log_return_2 = log_return_data['aord_log_return'].ix[i - 2]
        training_test_data = training_test_data.append({'snp_log_return_positive':snp_log_return_positive,
                                                        'snp_log_return_negative':snp_log_return_negative,
                                                        'snp_log_return_1':snp_log_return_1,
                                                        'snp_log_return_2':snp_log_return_2,
                                                        'snp_log_return_3':snp_log_return_3,
                                                        'nyse_log_return_1':nyse_log_return_1,
                                                        'nyse_log_return_2':nyse_log_return_2,
                                                        'nyse_log_return_3':nyse_log_return_3,
                                                        'djia_log_return_1':djia_log_return_1,
                                                        'djia_log_return_2':djia_log_return_2,
                                                        'djia_log_return_3':djia_log_return_3,
                                                        'nikkei_log_return_0':nikkei_log_return_0,
                                                        'nikkei_log_return_1':nikkei_log_return_1,
                                                        'nikkei_log_return_2':nikkei_log_return_2,
                                                        'hangseng_log_return_0':hangseng_log_return_0,
                                                        'hangseng_log_return_1':hangseng_log_return_1,
                                                        'hangseng_log_return_2':hangseng_log_return_2,
                                                        'ftse_log_return_0':ftse_log_return_0,
                                                        'ftse_log_return_1':ftse_log_return_1,
                                                        'ftse_log_return_2':ftse_log_return_2,
                                                        'dax_log_return_0':dax_log_return_0,
                                                        'dax_log_return_1':dax_log_return_1,
                                                        'dax_log_return_2':dax_log_return_2,
                                                        'aord_log_return_0':aord_log_return_0,
                                                        'aord_log_return_1':aord_log_return_1,
                                                        'aord_log_return_2':aord_log_return_2},
                                                        ignore_index=True)
    training_test_data.describe()
    predictors_tf = training_test_data[training_test_data.columns[2:]]
    classes_tf = training_test_data[training_test_data.columns[:2]]
    training_set_size = int(len(training_test_data) * 0.8)
    test_set_size = len(training_test_data) - training_set_size
    training_predictors_tf = predictors_tf[:training_set_size]
    training_classes_tf = classes_tf[:training_set_size]
    test_predictors_tf = predictors_tf[training_set_size:]
    test_classes_tf = classes_tf[training_set_size:]
    training_predictors_tf.describe()
    test_predictors_tf.describe()
    return test_classes_tf, test_predictors_tf, training_classes_tf, training_predictors_tf


if __name__ == "__main__":
    CREDENTIALS = "C:\\Users\\ben\\Downloads\\Time-Series_Prediction-291369b3a56d.json"
    PROJECT_ID = "ltsm-201717"
    aord, dax, djia, ftse, hangseng, nikkei, nyse, snp = collect_data()
    # TIME-SERIES ANALYSIS - exploratory data analysis
    # closing_data = describe_data(aord, dax, djia, ftse, hangseng, nikkei, nyse, snp)
    # plot_data(closing_data)
    # scale_data(closing_data)
    # plot_scaled_data(closing_data)
    # correlation_plot(closing_data)
    # scatter_plot(closing_data)
    # log_return_data = log_returns(closing_data)
    # log_returns_plot(log_return_data)
    # log_returns_correlation_plot(log_return_data)
    # log_correlations_Dmin0(log_return_data)
    # log_correlations_Dmin1(log_return_data)
    # log_correlations_Dmin2(log_return_data)
    # test_classes_tf, test_predictors_tf, training_classes_tf, training_predictors_tf = training_data(log_return_data)
    # # TIME-SERIES PREDICTION - binary classification with tensorflow
    # actual_classes, feature_data, model, sess, training_step = create_model_1L(training_classes_tf, training_predictors_tf)
    # train_model_1L(actual_classes, feature_data, model, sess, training_classes_tf, training_predictors_tf, training_step)
    # confusion_metrics_1L(actual_classes, feature_data, model, sess, test_classes_tf, test_predictors_tf)
    # actual_classes, feature_data, model, sess, train_op1 = create_model_2HL(training_classes_tf, training_predictors_tf)
    # train_model_2HL(actual_classes, feature_data, model, sess, train_op1, training_classes_tf, training_predictors_tf)
    # confusion_metrics_2HL(actual_classes, feature_data, model, sess, test_classes_tf, test_predictors_tf)
    print('working.')
