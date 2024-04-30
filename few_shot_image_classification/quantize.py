import tensorflow_probability as tfp
import tensorflow as tf


@tf.custom_gradient
def quantize(w, nb, a):
    non_sign_bits = nb - 1
    m = pow(2, non_sign_bits)
    n = m / a
    w_quan = tf.clip_by_value(tf.math.round(w * n), -m, m - 1) / n

    def grad(dy):
        return dy, None, None

    return w_quan, grad

@tf.custom_gradient
def quantize_stochastic_tunable(w, nb, a):
    non_sign_bits = nb - 1
    m = pow(2, non_sign_bits)
    n = m / a
    noise = tf.random.uniform(w.shape, minval=-0.5, maxval=0.5, dtype=w.dtype)
    w_quan = tf.clip_by_value(tf.math.round(w * n + noise), -m, m - 1) / n

    def grad(dy):
        return dy, None, None

    return w_quan, grad


@tf.custom_gradient
def quantize_nofix(w):
    w_quan = tf.cast(tf.cast(tf.math.minimum(w, 255.), tf.int8), tf.float32)

    def grad(dy):
        return dy

    return w_quan, grad


@tf.custom_gradient
def quantize_u(w, nb, a):
    m = pow(2, nb)
    n = m / a
    w_quan = tf.clip_by_value(tf.math.round(w * n), 0, m - 1) / n

    def grad(dy):
        return dy, None, None

    return w_quan, grad


@tf.custom_gradient
def quantize_10(w):
    nb = 1 / 16
    w_quan = tf.math.round(w / nb) * nb

    def grad(dy):
        return dy

    return w_quan, grad

@tf.custom_gradient
def no_quantization(w, nb, a):
    def grad(dy):
        return dy, None, None
    return w, grad

@tf.custom_gradient
def quantize_stochastic(w):
    nb = 1 / 16
    w_quan = tf.cast(tf.cast((w / nb), tf.int64), tf.float32) * nb

    dif_w = (w - w_quan) / nb
    update_pos_i = tf.cast(
        tfp.distributions.Bernoulli(probs=tf.clip_by_value(dif_w, clip_value_min=0., clip_value_max=1.)).sample(),
        tf.float32)
    update_neg_i = tf.cast(
        tfp.distributions.Bernoulli(probs=tf.clip_by_value(-dif_w, clip_value_min=0., clip_value_max=1.)).sample(),
        tf.float32)

    update_stoch = (update_pos_i - update_neg_i) * (1 / 16)

    final_w = w_quan + update_stoch

    def grad(dy):
        return dy

    return final_w, grad
