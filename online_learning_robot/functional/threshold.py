import tensorflow as tf


@tf.custom_gradient
def threshold(v, thr, dampening_factor):
    z_ = tf.greater((v - thr) / thr, 0.0)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        return dy * dampening_factor * tf.maximum(1.0 - tf.abs((v - thr) / thr), 0) / thr, None, None

    return z_, grad


@tf.custom_gradient
def threshold_lsnn(v, thr, thr_b, dampening_factor):
    z_ = tf.greater((v - thr) / thr_b, 0.0)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        return dy * dampening_factor * tf.maximum(1.0 - tf.abs((v - thr) / thr_b), 0) / thr_b, None, None, None

    return z_, grad
