import tensorflow as tf


@tf.function
def exp_convolve(tensor, decay, damp):
    with tf.name_scope('ExpConvolve'):
        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + damp * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])

    return filtered_tensor
