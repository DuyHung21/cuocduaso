import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
import cv2
import random
import math
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

def get_data(path="final"):
    X = np.load("X.npy")
    y = np.load("y.npy")

    return np.asarray(X), np.asarray(y)

X_origin, y_origin = get_data()
x_train = X_origin
y_train = y_origin
n_classes = len(np.unique(y_origin))

def normalize(x):
    return (x - x.mean())/x.std()

#x_train = np.array([grayAndEqualizeHist(img) for img in X_origin])

x_train = normalize(x_train)

#x_train = np.expand_dims(x_train, axis=3)

### Generate the validation set from the training set
from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1)

# Number of validation examples.
n_validation = len(y_validation)
# Number of training examples
n_train = len(y_train)
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Num classes: ", n_classes)
def ConvNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma), name="W1")
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x, b1)

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = x
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="W2")
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x = tf.nn.bias_add(x, b2)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # TODO: Layer 3: Convolutional. Output = 1x1x400.
    W3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma), name="W3")
    x = tf.nn.conv2d(x, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x, b3)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    
    
    # TODO: Flatten. Input = 14x14x6. Output = 1176.
    layer1flat = flatten(layer1)
    
    # Flatten x. Input = 1x1x400. Output = 400.
    xflat = flatten(x)
    
    # Concat layer1flat and x. Input = 1176 + 400. Output = 1576
    x = tf.concat([xflat, layer1flat], 1)
    
    # Dropout
    x = tf.cond(is_training,
                             lambda: tf.nn.dropout(x, keep_prob),
                             lambda: x)
    
    # TODO: Layer 4: Fully Connected. Input = 1576. Output = 43.
    W4 = tf.Variable(tf.truncated_normal(shape=(1576, n_classes), mean = mu, stddev = sigma), name="W4")
    b4 = tf.Variable(tf.zeros(n_classes), name="b4")    
    logits = tf.add(tf.matmul(x, W4), b4)
    
    return logits

### Train my model.
from sklearn.utils import shuffle

rate = 1.0e-3
EPOCHS = 250
BATCH_SIZE = 128
keep_prob = 0.5
with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='x')
    y = tf.placeholder(tf.int64, (None), name='y')
    is_training = tf.placeholder(tf.bool, name='is_training')
    
 


logits = ConvNet(x)
#logits = LeNet(x)

with tf.variable_scope('evaluation'):
    one_hot_y = tf.one_hot(y, n_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy, name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

with tf.variable_scope('prediction'):
    prediction = tf.argmax(logits, 1, name='prediction')
    probability = tf.nn.softmax(logits, name='probability')
correct_prediction = tf.equal(prediction, tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, is_training: False, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# Training and evaluation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)
    
    print("Training...")
    print()
    i = 0
    while True:
    #for i in range(EPOCHS):
        begin = datetime.now()
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, is_training: True, y: batch_y})
            
        validation_accuracy = evaluate(x_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Time: ", datetime.now() - begin)
        print()
        if (i == EPOCHS):
            break
        i+= 1
        if i%50 == 0:
            saver.save(sess, 'checkpoint/lenet_' + str(i) + '.ckpt')
    saver.save(sess, 'checkpoint/lenet.ckpt')
    print("Model saved")
