"""
Utilities function file - DICES copy (self-contained, no Mutlipico dependency)

@author DisCo Authors
"""
import tensorflow as tf
import numpy as np
import pickle

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

def save_object(model, fname):
    fd = open(fname, 'wb')
    pickle.dump(model, fd)
    fd.close()

def load_object(fname):
    fd = open(fname, 'rb')
    model = pickle.load(fd)
    fd.close()
    return model

def calc_mode(value_list):
    if len(value_list) == 0:
        return -1, 0
    freq_map = {}
    for i in range(len(value_list)):
        v_i = value_list[i]
        cnt = freq_map.get(v_i)
        if cnt is not None:
            cnt = cnt + 1
            freq_map.update({v_i: cnt})
        else:
            freq_map.update({v_i: 1})
    most_freq_v = None
    max_cnt = 0
    for v, cnt in freq_map.items():
        if cnt > max_cnt:
            max_cnt = cnt
            most_freq_v = v
    return most_freq_v, max_cnt

def D_KL(px, qx, keep_batch=False):
    eps = 1e-6
    px_ = tf.clip_by_value(px, eps, 1.0-eps)
    log_px = tf.math.log(px_)
    qx_ = tf.clip_by_value(qx, eps, 1.0-eps)
    log_qx = tf.math.log(qx_)
    term1 = tf.reduce_sum(-(px_ * log_px), axis=1, keepdims=True)
    term2 = tf.reduce_sum(px_ * log_qx, axis=1, keepdims=True)
    KL = -(term1 + term2)
    loss = KL
    if not keep_batch:
        loss = tf.math.reduce_mean(KL)
    return loss

def drop_out(input, rate=0.0, seed=69):
    if rate >= 1.0:
        mask = tf.zeros_like(input, dtype=tf.float32)
        output = tf.zeros_like(input, dtype=tf.float32)
        return output, mask
    mask = tf.math.less_equal(
        tf.random.uniform(shape=(input.shape[0], input.shape[1]), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed),
        (1.0 - rate))
    mask = tf.cast(mask, tf.float32) * (1.0 / (1.0 - rate))
    output = input * mask
    return output, mask

def softmax(x, tau=0.0):
    if tau > 0.0:
        x = x / tau
    max_x = tf.expand_dims(tf.reduce_max(x, axis=1), axis=1)
    exp_x = tf.exp(tf.subtract(x, max_x))
    return exp_x / tf.expand_dims(tf.reduce_sum(exp_x, axis=1), axis=1)

def calc_catNLL(target, prob, keep_batch=False):
    eps = 1e-7
    py = tf.clip_by_value(prob, eps, 1.0-eps)
    Ly = -tf.reduce_sum(tf.math.log(py) * target, axis=1, keepdims=True)
    if keep_batch is False:
        return tf.reduce_mean(Ly)
    return Ly

def init_weights(init_type, shape, seed, stddev=1.0):
    if init_type == "he_uniform":
        initializer = tf.compat.v1.keras.initializers.he_uniform()
        params = initializer(shape)
    elif init_type == "he_normal":
        initializer = tf.compat.v1.keras.initializers.he_normal()
        params = initializer(shape)
    elif init_type == "truncated_normal":
        params = tf.random.truncated_normal(shape, stddev=stddev, seed=seed)
    elif init_type == "normal" or init_type == "gaussian":
        params = tf.random.normal(shape, stddev=stddev, seed=seed)
    else:
        k = 1.0 / (shape[0] * 1.0)
        bound = np.sqrt(k)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)
    return params

def l1_l2_norm_calculation(theta_y, norm_type, mini_bath_size):
    norm_value = 0.0
    for var in theta_y:
        w_norm = tf.norm(var, ord=norm_type)
        norm_value += (w_norm * w_norm)
    norm_value = norm_value * 1 / mini_bath_size
    return norm_value
