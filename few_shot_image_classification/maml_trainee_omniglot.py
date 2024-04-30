import numpy as np
import tensorflow as tf

from einsums import tf_bix_bxr_bir_noise_feed, tf_bix_bxr_bir_noise, tf_bix_bxr_bir_feed, tf_bix_bxr_bir, \
    tf_beo_boh_beh_noise, tf_beo_boh_beh
from quantize import quantize_stochastic, quantize_stochastic_tunable, no_quantization
from conv_layer import ConvLayer
from pcm import PCMNoise, PCM


@tf.function
def identity(a, training=False):  # noqa
    return tf.identity(a)


class MAMLTraineeOmniglot(tf.keras.layers.Layer):
    def __init__(self, n_in, n_out, num_neurons_per_layer, use_biases=True, use_feedback_align=False,
                 use_batch_norm=True, update_only_readout=False, noise=False, **kwargs):
        self.n_in = n_in
        self.n_out = n_out
        self.num_neurons_per_layer = num_neurons_per_layer
        self.use_biases = use_biases
        self.use_feedback_align = use_feedback_align
        self.use_batch_norm = use_batch_norm
        self.update_only_readout = update_only_readout
        self.noise = noise

        if noise and use_feedback_align:
            self.w_module = PCMNoise(quantize_func=quantize_stochastic)
            self.tf_bix_bxr_bir = tf_bix_bxr_bir_noise_feed
        elif noise:
            self.w_module = PCMNoise(quantize_func=quantize_stochastic)
            self.tf_bix_bxr_bir = tf_bix_bxr_bir_noise
        elif use_feedback_align:
            self.tf_bix_bxr_bir = tf_bix_bxr_bir_feed
            self.w_module = PCM()
        else:
            self.w_module = PCM()
            self.tf_bix_bxr_bir = tf_bix_bxr_bir

        self.f_weights = []
        self.bias_terms = []
        self.norms = []

        self.work_weights = None
        self.work_biases = None
        self.variables_norm = None

        self.activation_fun = tf.nn.relu

        h_b = n_in
        for i, layer in enumerate(num_neurons_per_layer):
            self.f_weights.append(tf.Variable(name="weight_" + str(i),
                                              initial_value=tf.keras.initializers.GlorotUniform()((h_b, layer)),
                                              dtype='float32', trainable=True))
            self.bias_terms.append(tf.Variable(name="bias_" + str(i),
                                               initial_value=np.zeros(layer).astype(np.float32),
                                               dtype='float32', trainable=self.use_biases))
            if use_batch_norm:
                self.norms.append(tf.keras.layers.BatchNormalization(center=False, scale=False))
                self.norms[-1].build((None, None, layer))
            else:
                self.norms.append(identity)
            h_b = layer

        self.f_weights.append(tf.Variable(name="weight_out",
                                          initial_value=tf.keras.initializers.GlorotUniform()(
                                              (num_neurons_per_layer[-1], n_out)),
                                          dtype='float32', trainable=True))
        self.bias_terms.append(tf.Variable(name="bias_out", initial_value=np.zeros(n_out).astype(np.float32),
                                           dtype='float32', trainable=self.use_biases))

        super(MAMLTraineeOmniglot, self).__init__(**kwargs)

    def set_weights(self, batch_size):
        # work weights store noise weights
        # work final is either work weights if no feedback or work weights plus noise feedback weights
        self.work_weights = []
        self.work_biases = []

        for w in self.f_weights:
            forward_weights = self.w_module.init_weights(tf.tile(w[tf.newaxis, ...], [batch_size, 1, 1]))

            if self.feedback:
                self.work_weights.append(
                    [forward_weights, self.w_module.init_weights(tf.tile(w[tf.newaxis, ...], [batch_size, 1, 1]))])
            else:
                self.work_weights.append([forward_weights])

        for b in self.bias_terms:
            self.work_biases.append(tf.tile(b[tf.newaxis, tf.newaxis, ...], [batch_size, 1, 1]))

    def get_weights_to_watch(self):
        if self.update_only_readout:
            return [self.work_weights[-1]]

        weights_to_update_list = []
        for w in self.work_weights:
            weights_to_update_list.append(w[0])

        if self.bias:
            return weights_to_update_list + self.work_biases
        else:
            return weights_to_update_list

    def reprogram_weights(self, d_w):
        if self.update_only_readout:
            self.work_weights[-1][0] = self.w_module.re_program(self.work_weights[-1][0], d_w[0])
        else:
            num_of_weights = len(self.work_weights)

            for i, (w, d) in enumerate(zip(self.work_weights, d_w[:num_of_weights])):
                self.work_weights[i][0] = self.w_module.re_program(w[0], d)
            if self.bias:
                for i, (b, d) in enumerate(zip(self.work_biases, d_w[num_of_weights:])):
                    self.work_biases[i] = b + d

    def get_variables(self):
        if self.use_batch_norm:
            self.variables_norm = []
            for norm in self.norms:
                self.variables_norm = self.variables_norm + norm.trainable_variables

        if self.bias and self.use_batch_norm:
            return self.f_weights + self.bias_terms + self.variables_norm
        elif self.bias:
            return self.f_weights + self.bias_terms
        elif self.use_batch_norm:
            return self.f_weights + self.variables_norm
        else:
            return self.f_weights

    def call(self, inputs, train=True):
        data = tf.reshape(inputs, (*inputs.shape[:2], -1))  # b, inner_b, w, h

        x = data
        for w, b, norm in zip(self.work_weights[:-1], self.work_biases[:-1], self.norms):
            x = self.activation_fun(norm(self.tf_bix_bxr_bir(x, *w) + b, training=train))

        x_out = self.tf_bix_bxr_bir(x, *self.work_weights[-1]) + self.work_biases[-1]

        return x_out


class MAMLTraineeOmniglotConv(tf.keras.layers.Layer):
    def __init__(self, dim_in, n_out, hidden_channels=64, kernel_dims=3, noise=False, quantize=None, num_bits=None, **kwargs):
        assert dim_in == 28, "This conv net only accepts 28x28x1 images"

        self.dim_in = dim_in
        self.n_out = n_out
        self.hidden_channels = hidden_channels
        self.kernel_dims = kernel_dims
        self.noise = noise
        self.quantize = quantize
        self.num_bits = num_bits

        self.conv1 = ConvLayer(self.dim_in, self.dim_in, 1, 3, hidden_channels, padding=1, stride=2, noise=noise)
        self.conv2 = ConvLayer(self.conv1.out_height, self.conv1.out_width, hidden_channels, self.kernel_dims,
                               hidden_channels, padding=1, stride=2, noise=noise)
        self.conv3 = ConvLayer(self.conv2.out_height, self.conv2.out_width, hidden_channels, self.kernel_dims,
                               hidden_channels, padding=1, stride=2, noise=noise)
        self.conv4 = ConvLayer(self.conv3.out_height, self.conv3.out_width, hidden_channels, self.kernel_dims,
                               hidden_channels, padding=1, stride=2, noise=noise)

        self.kernel_1 = tf.Variable(name="kernel1", initial_value=tf.keras.initializers.GlorotUniform()(
            (self.kernel_dims, self.kernel_dims, 1, hidden_channels)), dtype='float32', trainable=True)
        self.kernel_2 = tf.Variable(name="kernel2", initial_value=tf.keras.initializers.GlorotUniform()(
            (self.kernel_dims, self.kernel_dims, hidden_channels, hidden_channels)), dtype='float32', trainable=True)
        self.kernel_3 = tf.Variable(name="kernel3", initial_value=tf.keras.initializers.GlorotUniform()(
            (self.kernel_dims, self.kernel_dims, hidden_channels, hidden_channels)), dtype='float32', trainable=True)
        self.kernel_4 = tf.Variable(name="kernel4", initial_value=tf.keras.initializers.GlorotUniform()(
            (self.kernel_dims, self.kernel_dims, hidden_channels, hidden_channels)), dtype='float32', trainable=True)

        self.w_readout = tf.Variable(name="kernel5",
                                     initial_value=tf.keras.initializers.GlorotUniform()((hidden_channels, self.n_out)),
                                     dtype='float32', trainable=True)

        self.kernels = [self.kernel_1, self.kernel_2, self.kernel_3, self.kernel_4]

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]

        self.norm_1 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=1.)
        self.norm_1.build((None, None, self.conv1.out_height, self.conv1.out_width, hidden_channels))

        self.norm_2 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=1.)
        self.norm_2.build((None, None, self.conv2.out_height, self.conv2.out_width, hidden_channels))

        self.norm_3 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=1.)
        self.norm_3.build((None, None, self.conv3.out_height, self.conv3.out_width, hidden_channels))

        self.norm_4 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=1.)
        self.norm_4.build((None, None, self.conv4.out_height, self.conv4.out_width, hidden_channels))

        self.norms = [self.norm_1, self.norm_2, self.norm_3, self.norm_4]

        self.max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(self.conv4.out_height, self.conv4.out_width),
                                                        strides=(1, 1), padding='valid')

        self.mat_mul_readout = tf_beo_boh_beh_noise if noise else tf_beo_boh_beh

        if noise:
            self.weight_setter = PCMNoise(quantize_func=quantize_stochastic, nb=self.num_bits, a=1.)
        else:
            quantization_func = quantize_stochastic_tunable if quantize else no_quantization
            self.weight_setter = PCM(quantization_func=quantization_func, nb=self.num_bits, a=1.)

        self.readout_weight = None
        self.kernels_forward = None

        super(MAMLTraineeOmniglotConv, self).__init__(**kwargs)

    def set_weights(self, batch_size):

        self.kernels_forward = []

        for k in self.kernels:
            self.kernels_forward.append(
                self.weight_setter.init_weights(tf.tile(k[tf.newaxis, ...], (batch_size, 1, 1, 1, 1))))

        self.readout_weight = self.weight_setter.init_weights(tf.tile(self.w_readout[tf.newaxis, ...],
                                                                      (batch_size, 1, 1)))

    def get_weights_to_watch(self):
        return [self.readout_weight]

    def reprogram_weights(self, d_w):
        self.readout_weight = self.weight_setter.re_program(self.readout_weight, d_w[0])

    def get_variables(self):
        return self.kernels + [self.w_readout]

    def call(self, inputs, training=True):
        x = inputs
        # print("Input: ", np.abs(inputs.numpy()).sum())
        # orig
        for conv, norm, kern_f in zip(self.conv_layers, self.norms, self.kernels_forward):
            x = norm(tf.nn.relu(conv(x, kern_f)), training=training)  # noqa
            # print("Conv: ", np.abs(x.numpy()).sum())

        # for conv, norm, kern_f in zip(self.conv_layers, self.norms, self.kernels_forward):
        #     x = tf.nn.relu(conv(x, kern_f))  # noqa
        batch_dims = x.shape[:2]
        pre_out = tf.squeeze(self.max_pool_2d(tf.reshape(x, (-1, *x.shape[2:]))))
        pre_out = tf.reshape(pre_out, (*batch_dims, *pre_out.shape[1:]))
        # print("Max Pool: ", np.abs(pre_out.numpy()).sum())

        out = self.mat_mul_readout(pre_out, self.readout_weight)
        # print("Final: ", np.abs(out.numpy()).sum())
        # import pudb
        # pu.db

        return out
