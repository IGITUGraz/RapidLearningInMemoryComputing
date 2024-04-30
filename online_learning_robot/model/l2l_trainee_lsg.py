from collections import namedtuple

import numpy as np
import tensorflow as tf

from functional.lif import LIFParameters
from module.lsnn_cell import LSNNCell
from module.rsnn_cell import RSNNCell
from module.scorbot import Scorbot

RegularizationParameters = namedtuple('RegularizationParameters', (
    'target_trainee_f',
    'target_lsg_f',
    'lambda_trainee',
    'lambda_lsg'
))


class L2LTraineeLSG:
    def __init__(
            self,
            dataset,
            clock_input_size,
            coordinate_input_size,
            lsg_size,
            trainee_size,
            output_size,
            lsg_output_size=None,
            noise_trainee=False,
            quantize_trainee=False,
            learning_rate_outer=1.5e-3,
            learning_rate_inner=0.001,
            decay_learning_rate_outer=0.98,
            alif_fraction_lsg=0.3,
            trainee_w_out_scaling_factor=1.0,
            smooth_output=False,
            smooth_output_window_length=30,
            output_angles=False,
            use_scorbot=False,
            scorbot_joints=(0, 1, 2, 3),
            lsg_params=LIFParameters(thr=0.4, tau=20.0, tau_o=20.0, damp=0.3, n_ref=0),
            trainee_params=LIFParameters(thr=0.4, tau=20.0, tau_o=20.0, damp=0.3, n_ref=0),
            reg_params=RegularizationParameters(target_trainee_f=10.0, target_lsg_f=10.0, lambda_trainee=0.0,
                                                lambda_lsg=0.0)
    ):

        self.dataset = dataset
        self.clock_input_size = clock_input_size
        self.coordinate_input_size = coordinate_input_size
        self.lsg_size = lsg_size
        self.trainee_size = trainee_size
        self.output_size = output_size
        self.lsg_output_size = trainee_size if lsg_output_size is None else lsg_output_size
        self.noise_trainee = noise_trainee
        self.quantize_trainee = quantize_trainee
        self.learning_rate_outer = learning_rate_outer
        self.learning_rate_inner = learning_rate_inner
        self.decay_learning_rate_outer = decay_learning_rate_outer
        self.alif_fraction_lsg = alif_fraction_lsg
        self.trainee_w_out_scaling_factor = trainee_w_out_scaling_factor
        self.smooth_output = smooth_output
        self.smooth_output_window_length = smooth_output_window_length
        self.output_angles = output_angles
        self.use_scorbot = use_scorbot
        self.scorbot_joints = scorbot_joints
        self.lsg_params = lsg_params
        self.trainee_params = trainee_params
        self.reg_params = reg_params

        self.init_position, self.clock_signal_trainee, self.clock_signal_lsg = self.dataset.get_static_data()

        cartesian_output_dim = 3 if use_scorbot else output_size
        self.lsg_input_size = clock_input_size + (cartesian_output_dim * coordinate_input_size)
        self.use_lsg_scale_weights = lsg_output_size is not None

        self.W_in_lsg = None
        self.W_rec_lsg = None
        self.W_out_lsg = None
        self.W_scale_lsg = None
        self.W_in_trainee = None
        self.W_rec_trainee = None
        self.W_out_trainee = None

        self.init_weights(self.use_lsg_scale_weights)

        self.variables = [
            self.W_in_lsg,
            self.W_rec_lsg,
            self.W_out_lsg,
            self.W_in_trainee,
            self.W_rec_trainee,
            self.W_out_trainee]

        if self.W_scale_lsg is not None:
            self.variables.append(self.W_scale_lsg)

        if alif_fraction_lsg is not None:
            n_lsg_lif = lsg_size - int(alif_fraction_lsg * lsg_size)
            n_lsg_alif = lsg_size - n_lsg_lif

            self.lsg_cell = LSNNCell(
                n_lif=n_lsg_lif,
                n_alif=n_lsg_alif,
                n_out=self.lsg_output_size,
                beta=1.6,
                tau_adaptation=600,
                params=lsg_params
            )
        else:
            self.lsg_cell = RSNNCell(
                n_in=self.lsg_input_size,
                units=lsg_size,
                n_out=self.lsg_output_size,
                noise=False,
                eprop=False,
                scaling_factor=1.0,
                batch_weight=False,
                params=lsg_params
            )

        self.trainee_cell = RSNNCell(
            n_in=clock_input_size,
            units=trainee_size,
            n_out=output_size,
            noise=noise_trainee,
            quantize=quantize_trainee,
            eprop=True,
            scaling_factor=trainee_w_out_scaling_factor,
            batch_weight=True,
            params=trainee_params
        )

        self.lsg = tf.keras.layers.RNN(self.lsg_cell, return_sequences=True)
        self.trainee = tf.keras.layers.RNN(self.trainee_cell, return_sequences=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_outer)

        if self.use_scorbot:
            self.scorbot = Scorbot()

    def init_weights(self, use_lsg_scale_weights=False):
        self.W_in_lsg = tf.Variable(
            name="input_weights_lsg",
            initial_value=(np.random.randn(self.lsg_input_size, self.lsg_size) / np.sqrt(self.lsg_input_size))
            .astype(np.float32),
            dtype='float32',
            trainable=True
        )
        self.W_rec_lsg = tf.Variable(
            name="recurrent_weights_lsg",
            initial_value=(np.random.randn(self.lsg_size, self.lsg_size) / np.sqrt(self.lsg_size))
            .astype(np.float32),
            dtype='float32',
            trainable=True
        )
        self.W_out_lsg = tf.Variable(
            name="output_weights_lsg",
            initial_value=tf.keras.initializers.GlorotUniform()((self.lsg_size, self.lsg_output_size)),
            dtype='float32',
            trainable=True
        )
        if use_lsg_scale_weights:
            self.W_scale_lsg = tf.Variable(
                name="scale_weights_lsg",
                initial_value=(np.random.randn(self.lsg_output_size, self.trainee_size) / np.sqrt(self.lsg_output_size))
                .astype(np.float32),
                dtype='float32',
                trainable=True
            )
        self.W_in_trainee = tf.Variable(
            name="input_weights_trainee",
            initial_value=(np.random.randn(self.clock_input_size, self.trainee_size) / np.sqrt(self.clock_input_size))
            .astype(np.float32),
            dtype='float32',
            trainable=True
        )
        self.W_rec_trainee = tf.Variable(
            name="recurrent_weights_trainee",
            initial_value=(np.random.randn(self.trainee_size, self.trainee_size) / np.sqrt(self.trainee_size))
            .astype(np.float32),
            dtype='float32',
            trainable=True
        )
        self.W_out_trainee = tf.Variable(
            name="output_weights_trainee",
            initial_value=(np.random.randn(self.trainee_size, self.output_size) / np.sqrt(self.trainee_size))
            .astype(np.float32) * (1. - np.exp(-1 / 20)),
            dtype='float32',
            trainable=True
        )

    def set_weights(self, variables):
        self.W_in_lsg.assign(variables['W_in_lsg'])
        self.W_rec_lsg.assign(variables['W_rec_lsg'])
        self.W_out_lsg.assign(variables['W_out_lsg'])
        self.W_in_trainee.assign(variables['W_in_trainee'])
        self.W_rec_trainee.assign(variables['W_rec_trainee'])
        self.W_out_trainee.assign(variables['W_out_trainee'])
        if len(variables) == 7:
            self.W_scale_lsg.assign(variables['W_scale_lsg'])

    def get_variables(self):
        keys = ['W_in_lsg', 'W_rec_lsg', 'W_out_lsg', 'W_in_trainee', 'W_rec_trainee', 'W_out_trainee']
        if len(self.variables) == 7:
            keys.append('W_scale_lsg')

        result = {}
        for key, value in zip(keys, self.variables):
            result[key] = value

        return result

    def decrease_learning_rate(self):
        self.optimizer.lr = self.optimizer.lr * self.decay_learning_rate_outer

    # @tf.function
    def loss(self, output, omega_t, angles_t, coordinates_t):

        if self.output_angles:
            position = self.init_position[:, None, :] + output
        else:
            # step_size = 0.01 (see trajectories.py)
            position = self.init_position[:, None, :] + 0.01 * tf.cumsum(output, axis=1)

        if self.use_scorbot:
            # TODO does not work in the general case
            args = {'q{0}'.format(joint + 1): position[:, :, joint] for joint in self.scorbot_joints}
            cartesian = self.scorbot.direct_kinematics(**args)
            cartesian = tf.stack((cartesian[0], cartesian[1], cartesian[2]), -1)
        else:
            phi0 = position[..., 0]
            phi1 = position[..., 1] + phi0

            cartesian_x = (tf.cos(phi0) + tf.cos(phi1)) * .5
            cartesian_y = (tf.sin(phi0) + tf.sin(phi1)) * .5
            cartesian = tf.stack((cartesian_x, cartesian_y), -1)

        mse_cartesian = tf.reduce_mean(tf.square(coordinates_t - cartesian))
        if self.output_angles:
            mse_angles = tf.reduce_mean(tf.square(angles_t - position))

            return 0.5 * mse_angles + 0.5 * mse_cartesian, cartesian
        else:
            mse_angular_velocities = tf.reduce_mean(tf.square(omega_t - output))

            return 0.5 * mse_angular_velocities + 0.5 * mse_cartesian, cartesian

    @tf.function
    def reg_loss(self, spikes, target_freq, lambda_reg):
        return tf.reduce_sum(tf.square(tf.reduce_mean(spikes, axis=(0, 1)) - target_freq / 1000.0)) * lambda_reg

    def inner_loop(self, target_coordinates_spike, coordinates_t, omega_t, angles_t, mask):
        batch_size = target_coordinates_spike.shape[0]

        w_in_tr = tf.tile(tf.expand_dims(self.W_in_trainee, axis=0), [batch_size, 1, 1])
        w_rec_tr = tf.tile(tf.expand_dims(self.W_rec_trainee, axis=0), [batch_size, 1, 1])
        w_out_tr = tf.tile(tf.expand_dims(self.W_out_trainee, axis=0), [batch_size, 1, 1])

        self.trainee_cell.set_init_weights(w_in_tr, w_rec_tr, w_out_tr)

        trainee_output_training, trainee_spikes_training, trainee_potential_training, trainee_trace_in_training, \
            trainee_trace_rec_training, dz_dh = self.trainee(self.clock_signal_trainee)

        _, cartesian_training = self.loss(trainee_output_training, omega_t, angles_t, coordinates_t)

        self.lsg_cell.set_init_weights(self.W_in_lsg, self.W_rec_lsg, self.W_out_lsg)

        lsg_input = tf.concat((self.clock_signal_lsg, target_coordinates_spike), axis=-1)

        lsg_learning_signals, lsg_spikes = self.lsg(lsg_input)

        if self.W_scale_lsg is not None:
            lsg_learning_signals = tf.einsum("bto, ol->btl", tf.nn.relu(lsg_learning_signals), self.W_scale_lsg)

        d_w_in = self.trainee_cell.calc_d_l_trace(dz_dh, lsg_learning_signals, trainee_trace_in_training)
        d_w_rec = self.trainee_cell.calc_d_l_trace(dz_dh, lsg_learning_signals, trainee_trace_rec_training)

        self.trainee_cell.reprogram_weights(-self.learning_rate_inner * d_w_in, -self.learning_rate_inner * d_w_rec)

        trainee_output_test, trainee_spikes_test, trainee_potential_test, trainee_trace_in_test, \
            trainee_trace_rec_test, _ = self.trainee(self.clock_signal_trainee)

        if self.smooth_output:
            window = tf.signal.hann_window(self.smooth_output_window_length)

            window = window / tf.reduce_sum(window)

            first = tf.stack((window, tf.zeros_like(window)), axis=-1)
            second = tf.stack((tf.zeros_like(window), window), axis=-1)
            smooth_kernel = tf.stack((first, second), axis=-1)

            # trainee_output_test = tf.nn.conv1d(trainee_output_test, smooth_kernel, 1, 'SAME')

            # TODO test this
            padded_angular_vel = tf.concat((tf.reverse(trainee_output_test[:, 0:self.smooth_output_window_length - 1],
                                                       axis=[1]), trainee_output_test,
                                            tf.reverse(trainee_output_test[:, -self.smooth_output_window_length + 1:],
                                                       axis=[1])), axis=1)

            trainee_output_test = tf.nn.conv1d(padded_angular_vel, smooth_kernel, 1, 'SAME')[:, int(
                self.smooth_output_window_length // 2):(int(self.smooth_output_window_length // 2) +
                                                        trainee_output_test.shape[1])]

        mean_loss, cartesian_test = self.loss(trainee_output_test, omega_t, angles_t, coordinates_t)

        reg_loss = self.reg_loss(tf.concat((trainee_spikes_training, trainee_spikes_test), axis=1),
                                 self.reg_params.target_trainee_f, self.reg_params.lambda_trainee) \
            + self.reg_loss(lsg_spikes, self.reg_params.target_lsg_f, self.reg_params.lambda_lsg)

        loss = mean_loss + reg_loss

        if self.output_angles:
            trainee_output_test = trainee_output_test + self.init_position[:, None, :]
            trainee_output_training = trainee_output_training + self.init_position[:, None, :]

        return loss, trainee_output_test, trainee_output_training, cartesian_training, cartesian_test, \
            (lsg_spikes, trainee_spikes_training, trainee_spikes_test), \
            (trainee_potential_training, trainee_potential_test), (trainee_trace_in_training, trainee_trace_in_test), \
            (trainee_trace_rec_training, trainee_trace_rec_test), dz_dh, lsg_learning_signals, mean_loss

    def train(self):
        target_coordinates_spike, coordinates_t, omega_t, angles_t, mask = self.dataset.get_data()

        with tf.GradientTape(persistent=False) as g:
            loss, trainee_output_test, trainee_output_training, cartesian_training, cartesian_test, spikes, potentials,\
                trace_in, trace_rec, dz_dh, learning_signals, loss_plot \
                = self.inner_loop(target_coordinates_spike, coordinates_t, omega_t, angles_t, mask)

        grads = g.gradient(loss, self.variables)

        self.optimizer.apply_gradients(zip(grads, self.variables))

        return loss.numpy(), trainee_output_test.numpy(), trainee_output_training.numpy(), cartesian_training, \
            cartesian_test, (omega_t, angles_t, coordinates_t), spikes, loss_plot

    def test(self, data=None):

        if data is None:
            target_coordinates_spike, coordinates_t, omega_t, angles_t, mask = self.dataset.get_data()
        else:
            target_coordinates_spike, coordinates_t, omega_t, angles_t, mask = data

        loss, trainee_output_test, trainee_output_training, cartesian_training, cartesian_test, spikes, potentials, \
            trace_in, trace_rec, dz_dh, learning_signals, loss_plot = \
            self.inner_loop(target_coordinates_spike, coordinates_t, omega_t, angles_t, mask)

        return loss.numpy(), trainee_output_test.numpy(), trainee_output_training.numpy(), cartesian_training, \
            cartesian_test, (omega_t, angles_t, coordinates_t), spikes, potentials, trace_in, trace_rec, dz_dh, \
            learning_signals, loss_plot
