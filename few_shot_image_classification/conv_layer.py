import numpy as np
import tensorflow as tf

from einsums import tf_bidk_bkf_bdf_noise_factorout


@tf.function
def tf_bikf_bdk_bdf(image_shaped, kernel_flat):
    return tf.einsum("bkf, bidk->bidf", kernel_flat, image_shaped)


class ConvLayer(tf.keras.layers.Layer):
    """2D convolution layer for MAML and PCM devices.

    2 batch sizes: outer and inner on input, outer on kernel
    Perform pcm read noise per dot product.
    tf_bidk_bkf_bdf_noise_factorout factors out the noise part in samples the noise afterwards
    """
    def __init__(self, image_h, image_w, in_channels, kernel_dims, out_channels, padding=1, stride=1, noise=True,
                 **kwargs):
        super(ConvLayer, self).__init__(**kwargs)

        self.image_h = image_h
        self.image_w = image_w
        self.channels = in_channels
        self.kernel_dims = kernel_dims
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.noise = noise

        self.paddings = tf.constant([[0, 0], [0, 0], [padding, padding], [padding, padding], [0, 0]])

        self.out_height = int((image_h + 2 * padding - kernel_dims) / stride + 1)
        self.out_width = int((image_w + 2 * padding - kernel_dims) / stride + 1)

        if self.noise:
            self.mult_function = tf_bidk_bkf_bdf_noise_factorout
        else:
            self.mult_function = tf_bikf_bdk_bdf

        # Get the indices of the convolutions
        all_in = []
        for center_index_h in range(padding, self.image_h - 1 + padding * 2, self.stride):
            for center_index_w in range(padding, self.image_w - 1 + padding * 2, self.stride):
                indices_kernel = []
                for k_h in range(self.kernel_dims):
                    for k_w in range(self.kernel_dims):
                        for k_c in range(self.channels):
                            indices_kernel.append((center_index_h + k_h - 1, center_index_w + k_w - 1, k_c))
                all_in.append(indices_kernel)

        self.all_in = tf.constant(np.asarray(all_in))

    def call(self, x, kernel):
        # Pad image with zeros
        x_pad = tf.pad(x, self.paddings, "CONSTANT")

        batch_dim = x.shape[0] * x.shape[1]

        # Reshape indices in list
        indices = tf.reshape(self.all_in, (-1, 3))  # 3 because conv 2d  h,w,c

        batch_inner_outer = x_pad.shape[:2]

        # Extract features in the order they get used by applying the convolution
        image_as = tf.gather_nd(tf.reshape(x_pad, (-1, * x_pad.shape[2:])),
                                tf.tile(indices[tf.newaxis, ...], [batch_dim, 1, 1]), batch_dims=1)

        # Reshape image to (batch, flat_features) for dot product, width*height (depends on padding and how many
        # times the dot product is performed)
        image_ready_for_conv = tf.reshape(image_as, (*batch_inner_outer, self.all_in.shape[0], self.all_in.shape[1]))

        # Collapse kernel into (height*width*in_channels, out_channels)
        kernel_reshape = tf.reshape(kernel, (
            kernel.shape[0], self.kernel_dims * self.kernel_dims * self.channels, self.out_channels))

        conv_output_factor_out = self.mult_function(image_ready_for_conv, kernel_reshape)

        return tf.reshape(conv_output_factor_out,
                          (*batch_inner_outer, self.out_height, self.out_width, self.out_channels))
