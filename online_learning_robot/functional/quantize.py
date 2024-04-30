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
