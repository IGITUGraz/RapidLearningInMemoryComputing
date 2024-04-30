from collections import namedtuple

import tensorflow as tf
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell  # noqa

from functional.lif import LIFParameters, lif_step_batch_noise, lif_step_noise, lif_step_batch, lif_step
from module.pcm import PCMNoise, PCM

RSNNState = namedtuple('RSNNState', (
    'z',
    'v',
    'r'
))

RSNNOutputState = namedtuple('RSNNOutputState', (
    'v',
    'z',
    'r',
    'out'
))

RSNNOutputStateEprop = namedtuple('RSNNOutputStateEprop', (
    "v",
    "z",
    "r",
    "out",
    "trace_in",
    "trace_rec"
))


class RSNNCell(tf.keras.layers.Layer):
    def __init__(
            self,
            n_in,
            units,
            n_out,
            noise=False,
            quantize=False,
            eprop=False,
            scaling_factor=1.0,
            batch_weight=False,
            params=LIFParameters(thr=0.4, tau=20.0, tau_o=20.0, damp=0.3, n_ref=5),
            **kwargs
    ):
        super(RSNNCell, self).__init__(**kwargs)

        self.n_in = n_in
        self.units = units
        self.n_out = n_out
        self.noise = noise
        self.quantize = quantize
        self.eprop = eprop
        self.scaling_factor = scaling_factor
        self.batch_weight = batch_weight
        self.params = params

        self.dampening_factor = params.damp
        self.decay = tf.exp(-1. / params.tau)
        self.kappa = tf.exp(-1. / params.tau_o)
        self.n_refractory = params.n_ref
        self.thr = params.thr

        if not eprop:
            self.state_size = [units, units, units, n_out]
            self.output_size = [n_out, units]
            self.call_f = self.call_simple
        else:
            self.state_size = [units, units, units, n_out, n_in, units]
            self.output_size = [n_out, units, n_in, units, units]
            self.call_f = self.call_eprop

        if noise:
            self.w_module = PCMNoise(quantize=quantize)
        else:
            self.w_module = PCM(quantize=quantize)

        if noise and batch_weight:
            self.lif_step = lif_step_batch_noise
        elif noise:
            self.lif_step = lif_step_noise
        elif batch_weight:
            self.lif_step = lif_step_batch
        else:
            self.lif_step = lif_step

        self.w_in = None
        self.w_rec = None
        self.w_out = None

    def set_init_weights(self, w_in, w_rec, w_out):
        self.w_in = self.w_module.init_weights(w_in)
        self.w_rec = self.w_module.init_weights(w_rec)
        self.w_out = self.w_module.init_weights(w_out)

    def reprogram_weights(self, delta_w_in, delta_w_rec):
        self.w_in = self.w_module.re_program(self.w_in, delta_w_in)
        self.w_rec = self.w_module.re_program(self.w_rec, delta_w_rec)

    @staticmethod
    def calc_d_l_trace(dz_dh, l, trace):
        return tf.einsum("btr,bti->bir", dz_dh * l, trace)

    def call(self, inputs, states):
        new_out, new_states = self.call_f(inputs, states)
        return new_out, new_states

    def call_eprop(self, inputs, states):
        state = RSNNOutputStateEprop(
            v=states[0],
            z=states[1],
            r=states[2],
            out=states[3],
            trace_in=states[4],
            trace_rec=states[5]
        )

        new_v, new_z, new_r, new_out = self.lif_step(
            inputs,
            state.v,
            state.z,
            state.r,
            state.out,
            self.w_in,
            self.w_rec,
            self.w_out * self.scaling_factor,
            self.thr,
            self.decay,
            self.kappa,
            self.n_refractory,
            self.dampening_factor
        )

        dz_dh = tf.maximum(self.dampening_factor * (1.0 - tf.abs((new_v - self.thr) / self.thr)), 0.) / self.thr
        dz_dh = tf.where(tf.greater(state.r, .1), tf.zeros_like(dz_dh), dz_dh)

        new_trace_in = state.trace_in * self.decay + inputs
        new_trace_rec = state.trace_rec * self.decay + state.z

        return [new_out, new_z, new_v, new_trace_in, new_trace_rec, dz_dh], \
               [new_v, new_z, new_r, new_out, new_trace_in, new_trace_rec]

    def call_simple(self, inputs, states):
        state = RSNNOutputState(v=states[0], z=states[1], r=states[2], out=states[3])

        new_v, new_z, new_r, new_out = self.lif_step(
            inputs,
            state.v,
            state.z,
            state.r,
            state.out,
            self.w_in,
            self.w_rec,
            self.w_out * self.scaling_factor,
            self.thr,
            self.decay,
            self.kappa,
            self.n_refractory,
            self.dampening_factor
        )

        return [new_out, new_z], [new_v, new_z, new_r, new_out]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))
