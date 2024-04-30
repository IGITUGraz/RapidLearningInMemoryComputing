from collections import namedtuple

import tensorflow as tf

from functional.threshold import threshold, threshold_lsnn
from module.pcm import PCMNoise

LIFParameters = namedtuple('LIFParameters', (
    'thr',
    'tau',
    'tau_o',
    'damp',
    'n_ref'
))


@tf.custom_gradient
def tf_bi_bij_bj_noise(a, b):
    c = tf.einsum("bi,bij->bj", a, b + PCMNoise.read_noise(b))

    def grad(dy):
        return tf.einsum("bj,bij->bi", dy, b), tf.einsum("bj,bi->bij", dy, a)

    return c, grad


@tf.custom_gradient
def tf_bi_ij_bj_noise(a, b):
    c = tf.einsum("bi,ij->bj", a, b + PCMNoise.read_noise(b))

    def grad(dy):
        return tf.matmul(dy, tf.transpose(b)), tf.matmul(tf.transpose(a), dy)

    return c, grad


@tf.function
def lif_step(inputs, v, z, r, v_out, w_in, w_rec, w_out, thr, decay, kappa, n_refractory, dampening_factor):
    new_v = decay * v + (tf.matmul(inputs, w_in) +
                         tf.matmul(z, (1.0 - tf.eye(w_rec.shape[0], w_rec.shape[0])) * w_rec)) - z * thr

    new_z = tf.where(tf.greater(r, 0.1), tf.zeros_like(z), threshold(new_v, thr, dampening_factor))
    new_r = tf.clip_by_value(r + n_refractory * new_z - 1, 0.0, float(n_refractory))

    new_out = kappa * v_out + tf.matmul(new_z, w_out)

    return new_v, new_z, new_r, new_out


@tf.function
def lif_step_noise(inputs, v, z, r, v_out, w_in, w_rec, w_out, thr, decay, kappa, n_refractory, dampening_factor):
    new_v = decay * v + (tf_bi_ij_bj_noise(inputs, w_in) +
                         tf_bi_ij_bj_noise(z, (1.0 - tf.eye(w_rec.shape[0], w_rec.shape[0])) * w_rec)) - z * thr

    new_z = tf.where(tf.greater(r, 0.1), tf.zeros_like(z), threshold(new_v, thr, dampening_factor))
    new_r = tf.clip_by_value(r + n_refractory * new_z - 1, 0.0, float(n_refractory))

    new_out = kappa * v_out + tf_bi_ij_bj_noise(new_z, w_out)

    return new_v, new_z, new_r, new_out


@tf.function
def lif_step_batch(inputs, v, z, r, out, w_in, w_rec, w_out, thr, decay, kappa, n_refractory, dampening_factor):
    new_v = decay * v + (
            tf.einsum("bi,bir-> br", inputs, w_in) +
            tf.einsum("br, brj -> bj", z, (1.0 - tf.eye(w_rec.shape[1], w_rec.shape[1])) * w_rec)) - z * thr

    new_z = tf.where(tf.greater(r, 0.1), tf.zeros_like(z), threshold(new_v, thr, dampening_factor))
    new_r = tf.clip_by_value(r + n_refractory * new_z - 1, 0.0, float(n_refractory))

    new_out = kappa * out + tf.einsum("bi,bir-> br", new_z, w_out)

    return new_v, new_z, new_r, new_out


@tf.function
def lif_step_batch_noise(inputs, v, z, r, out, w_in, w_rec, w_out, thr, decay, kappa, n_refractory, dampening_factor):
    new_v = decay * v + (tf_bi_bij_bj_noise(inputs, w_in) +
                         tf_bi_bij_bj_noise(z, (1.0 - tf.eye(w_rec.shape[1], w_rec.shape[1])) * w_rec)) - z * thr

    new_z = tf.where(tf.greater(r, 0.1), tf.zeros_like(z), threshold(new_v, thr, dampening_factor))
    new_r = tf.clip_by_value(r + n_refractory * new_z - 1, 0.0, float(n_refractory))

    new_out = kappa * out + tf_bi_bij_bj_noise(new_z, w_out)

    return new_v, new_z, new_r, new_out


@tf.function
def lif_step_lsnn(inputs, v, z, r, v_out, w_in, w_rec, w_out, thr, thr_b, decay, kappa, n_refractory, dampening_factor):
    new_v = decay * v + (tf.matmul(inputs, w_in) +
                         tf.matmul(z, (1.0 - tf.eye(w_rec.shape[0], w_rec.shape[0])) * w_rec)) - z * thr

    new_z = tf.where(tf.greater(r, 0.1), tf.zeros_like(z), threshold_lsnn(new_v, thr, thr_b, dampening_factor))
    new_r = tf.clip_by_value(r + n_refractory * new_z - 1, 0.0, float(n_refractory))

    new_out = kappa * v_out + tf.matmul(new_z, w_out)

    return new_v, new_z, new_r, new_out
