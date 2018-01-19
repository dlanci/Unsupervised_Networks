import numpy as np
import pandas as pd

from sklearn.utils import shuffle
import tensorflow as tf

def relu(x):
    return x * (x > 0)


def error_rate(p, t):
    return np.mean(p != t)

def evaluation(Y_pred, Y):
    correct = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return accuracy

def init_weights(shape):
    w = np.random.randn(*shape) / np.sqrt(sum(shape))
    return w


def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def random_mini_batches(X, mini_batch_size, seed):

    """
    Creates a list of random mini_batches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[0]        #number of examples in set
    mini_batches=[]
    
    #shuffle (X)
    permutation = list(np.random.permutation(m))
    
    shuffled_X = X[permutation,:]

    #partition of (shuffled_X, shuffled_Y) except the last mini_batch
    
    num_complete_mini_batches = math.floor(m/mini_batch_size)
    for k in range(num_complete_mini_batches):
        
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batches.append(mini_batch_X)
        
    # handling the case of last mini_batch < mini_batch_size    
    if m % mini_batch_size !=0:
        
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_mini_batches:m,:]
        mini_batches.append(mini_batch_X)
    
    return mini_batches