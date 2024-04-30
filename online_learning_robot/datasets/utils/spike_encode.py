import tensorflow as tf
import tensorflow_probability as tfp


def spike_encode(input_component, min_val, max_val, n_input_code=100, max_rate_hz=200, dt=1, n_dt_per_step=None):
    if 110 < n_input_code < 210:  # 100
        factor = 20
    elif 90 < n_input_code < 110:  # 100
        factor = 10
    elif 15 < n_input_code < 25:  # 20
        factor = 4
    else:
        factor = 2

    sigma_tuning = (max_val - min_val) / n_input_code * factor
    mean_line = tf.cast(tf.linspace(min_val - 2. * sigma_tuning, max_val + 2. * sigma_tuning, n_input_code), tf.float32)
    max_rate = max_rate_hz / 1000
    max_prob = max_rate * dt

    step_neuron_firing_prob = max_prob * tf.exp(-(mean_line[None, None, :] - input_component[..., None]) ** 2 /
                                                (2 * sigma_tuning ** 2))

    if n_dt_per_step is not None:
        spike_code = tfp.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample(n_dt_per_step)
        dims = len(spike_code.get_shape())
        r = list(range(dims))
        spike_code = tf.transpose(spike_code, r[1:-1] + [0, r[-1]])
    else:
        spike_code = tf.compat.v1.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample()

    spike_code = tf.cast(spike_code, tf.float32)

    return spike_code
