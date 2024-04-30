import tensorflow as tf

from pcm import PCMNoise


@tf.custom_gradient
def tf_bix_bxr_bir_noise_feed(a, b, f):
    c = tf.einsum("bix,bxr->bir", a, b + PCMNoise.read_noise(b))

    def grad(dy):
        return tf.einsum("bir,bxr->bix", dy, f + PCMNoise.read_noise(f)), tf.einsum("bir,bix->bxr", dy, a), None

    return c, grad


@tf.custom_gradient
def tf_bix_bxr_bir_feed(a, b, f):
    c = tf.einsum("bix,bxr->bir", a, b)

    def grad(dy):
        return tf.einsum("bir,bxr->bix", dy, f), tf.einsum("bir,bix->bxr", dy, a), None

    return c, grad


@tf.custom_gradient
def tf_bix_bxr_bir_noise(a, b):
    c = tf.einsum("bix,bxr->bir", a, b + PCMNoise.read_noise(b))

    def grad(dy):
        return tf.einsum("bir,bxr->bix", dy, b), tf.einsum("bir,bix->bxr", dy, a)

    return c, grad


@tf.custom_gradient
def tf_bix_bxr_bir(a, b):
    c = tf.einsum("bix,bxr->bir", a, b)

    def grad(dy):
        return tf.einsum("bir,bxr->bix", dy, b), tf.einsum("bir,bix->bxr", dy, a)

    return c, grad


# d => axis along conv (dot products) are applied... flatten output image dims (w_out*h_out)
# f => output channels
# k => kernel_dim*kernel_dim*in_channels, also dot product vector size
# b => batch_dim
@tf.custom_gradient
def tf_bidk_bkf_bdf_noise_factorout(image_shaped, kernel_flat):
    sigmas = PCMNoise.read_noise_sigma(kernel_flat)

    sigma_feature_sq = tf.square(
        tf.expand_dims(image_shaped, axis=-1) * tf.expand_dims(tf.expand_dims(sigmas, axis=1), axis=1))

    sigma_output = tf.sqrt(tf.reduce_sum(sigma_feature_sq, axis=3))  # bdf

    conv_clean = tf.einsum("bkf, bidk->bidf", kernel_flat, image_shaped)

    output = conv_clean + tf.random.normal(conv_clean.shape, mean=0.0, stddev=sigma_output)

    def grad(dy):
        return tf.einsum("bidf,bkf->bidk", dy, kernel_flat), tf.einsum("bidf,bidk->bkf", dy, image_shaped)

    return output, grad


@tf.custom_gradient
def tf_beo_boh_beh_noise(a, b):
    c = tf.einsum("beo,boh->beh", a, b + PCMNoise.read_noise(b))

    def grad(dy):
        return tf.einsum("beh,boh->beo", dy, b), tf.einsum("beh,beo->boh", dy, a)

    return c, grad


@tf.custom_gradient
def tf_beo_boh_beh(a, b):
    c = tf.einsum("beo,boh->beh", a, b)

    def grad(dy):
        return tf.einsum("beh,boh->beo", dy, b), tf.einsum("beh,beo->boh", dy, a)

    return c, grad
