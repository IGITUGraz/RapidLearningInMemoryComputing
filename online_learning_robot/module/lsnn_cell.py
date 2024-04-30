from collections import namedtuple

import tensorflow as tf
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell  # noqa

from functional.lif import lif_step_lsnn, LIFParameters

LSNNOutputState = namedtuple('LSNNOutputState', (
    'v',
    'z',
    'r',
    'out',
    'b'
))


class LSNNCell(tf.keras.layers.Layer):
    def __init__(
            self,
            n_lif,
            n_alif,
            n_out,
            beta=1.6,
            tau_adaptation=600,
            params=LIFParameters(thr=0.4, tau=20.0, tau_o=20.0, damp=0.3, n_ref=5),
            **kwargs
    ):
        super(LSNNCell, self).__init__(**kwargs)

        self.n_lif = n_lif
        self.n_alif = n_alif
        self.n_out = n_out
        self.beta = beta
        self.tau_adaptation = tau_adaptation
        self.params = params

        self.units = n_lif + n_alif

        self.dampening_factor = params.damp
        self.decay = tf.exp(-1. / params.tau)
        self.kappa = tf.exp(-1. / params.tau_o)
        self.n_refractory = params.n_ref
        self.thr_base = params.thr

        self.decay_b = tf.exp(-1. / tau_adaptation)
        self.beta = tf.concat((tf.zeros(n_lif), tf.fill(n_alif, beta)), axis=-1)[tf.newaxis, ...]

        self.state_size = [self.units, self.units, self.units, n_out, self.units]
        self.output_size = [n_out, self.units]

        self.w_in = None
        self.w_rec = None
        self.w_out = None

    def set_init_weights(self, w_in, w_rec, w_out):
        self.w_in = w_in
        self.w_rec = w_rec
        self.w_out = w_out

    def call(self, inputs, states):
        state = LSNNOutputState(v=states[0], z=states[1], r=states[2], out=states[3], b=states[4])

        new_b = self.decay_b * state.b + state.z
        thr_ac = self.thr_base + new_b * self.beta

        new_v, new_z, new_r, new_out = lif_step_lsnn(
            inputs,
            state.v,
            state.z,
            state.r,
            state.out,
            self.w_in,
            self.w_rec,
            self.w_out,
            thr_ac,
            self.thr_base,
            self.decay,
            self.kappa,
            self.n_refractory,
            self.dampening_factor
        )

        return [new_out, new_z], [new_v, new_z, new_r, new_out, new_b]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))
