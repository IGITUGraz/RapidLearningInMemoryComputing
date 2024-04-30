import tensorflow as tf

from functional.quantize import quantize_10


@tf.custom_gradient
def clip_with_grad(x, v_min, v_max):
    x_clipped = tf.clip_by_value(x, v_min, v_max)

    def grad(dy):
        return dy, None, None

    return x_clipped, grad


class PCM:
    def __init__(self, quantize):
        self.quantize = quantize

    @staticmethod
    def init_weights(w):
        return w

    @staticmethod
    def program(w):
        return w

    def re_program(self, w, dw):
        if self.quantize:
            dw = quantize_10(dw)

        return w + dw

    @staticmethod
    def read(w):
        return w


class PCMNoise:
    def __init__(self, quantize, w_max=1.0, g_max=25.0, t_start=32):
        self.quantize = quantize
        self.w_max = w_max
        self.g_max = g_max
        self.t_start = t_start
        self.nb = 5.0
        self.a = 1.0
        self.delta_w_reprogram = 1 / 16

    def drift_noise(self, w, mask=None):
        g_target = w * self.g_max
        g_relative = tf.maximum(tf.math.abs(g_target / self.g_max), 0)

        # gt should be normalized wrt g_max
        mu_drift = tf.clip_by_value((-0.0155 * tf.math.log(g_relative) + 0.0244), 0.049, 0.1)
        sig_drift = tf.clip_by_value((-0.0125 * tf.math.log(g_relative) - 0.0059), 0.008, 0.045)

        if mask is not None:
            nu_drift = tf.maximum(tf.math.abs(mu_drift + sig_drift * tf.random.normal(g_relative.shape)), 0) * mask
        else:
            nu_drift = tf.maximum(tf.math.abs(mu_drift + sig_drift * tf.random.normal(g_relative.shape)), 0)
        w_noise = g_target * ((32 / 1) ** (- tf.stop_gradient(nu_drift))) / self.g_max
        w_clipped_noise = clip_with_grad(w_noise, -1, 1)

        return w_clipped_noise - w

    def program_noise(self, w, mask=None):
        g_t = w * self.g_max

        positive_part = tf.maximum(g_t, 0.0)
        negative_part = tf.minimum(g_t, 0.0)

        mask_pos = tf.cast(tf.cast(positive_part, tf.bool), tf.float32)
        mask_neg = tf.cast(tf.cast(negative_part, tf.bool), tf.float32)

        sigma_prog = tf.maximum(-1.1731 * tf.square(g_t / self.g_max) + 1.9650 * tf.abs(g_t / self.g_max) + 0.2635, 0.)

        if mask is not None:
            noise = mask * tf.random.normal(g_t.shape, mean=0.0, stddev=sigma_prog)
        else:
            noise = tf.random.normal(g_t.shape, mean=0.0, stddev=sigma_prog)

        positive_part_noise = clip_with_grad(positive_part + noise * mask_pos, 0.0, self.g_max)
        negative_part_noise = clip_with_grad(negative_part + noise * mask_neg, -self.g_max, 0.0)

        g_t_noise = positive_part_noise + negative_part_noise
        w_noise = g_t_noise / self.g_max

        return w_noise - w

    @staticmethod
    def read_noise(w):
        g_max = 25.0
        t_start = 32.0
        w_max = 1.0
        number_of_synapses = 2
        w = clip_with_grad(w, -w_max, w_max)
        g_t = w * g_max
        t_read = 250.0e-9

        positive_part = tf.maximum(g_t, 0.0)
        negative_part = tf.minimum(g_t, 0.0)

        mask_pos = tf.cast(tf.cast(positive_part, tf.bool), tf.float32)
        mask_neg = tf.cast(tf.cast(negative_part, tf.bool), tf.float32)

        q_s = tf.minimum(0.0088 / tf.math.pow(tf.math.abs(g_t), 0.65), 0.2)

        sigma = tf.math.abs(g_t) * q_s * tf.math.sqrt(tf.math.log((t_start + t_read) / t_read))

        noise = tf.random.normal(g_t.shape, mean=0.0,
                                 stddev=sigma * tf.sqrt(number_of_synapses * ((1 / number_of_synapses) ** 2)))

        positive_part_noise = clip_with_grad(positive_part + noise * mask_pos, 0.0, g_max)
        negative_part_noise = clip_with_grad(negative_part + noise * mask_neg, -g_max, 0.0)
        g_t_noise = positive_part_noise + negative_part_noise

        w_noise = g_t_noise / g_max
        w_clipped_noise = clip_with_grad(w_noise, -w_max, w_max)

        return tf.stop_gradient(w_clipped_noise - w)

    def init_weights(self, w):
        return clip_with_grad(
            clip_with_grad(w, -self.w_max, self.w_max)
            + tf.random.uniform(w.shape, minval=-0.01, maxval=0.01)
            + self.drift_noise(w, None), -self.w_max, self.w_max)

    def program(self, w, mask=None):
        return clip_with_grad(
            clip_with_grad(w, -self.w_max, self.w_max)
            + tf.stop_gradient(self.program_noise(w, mask))
            + self.drift_noise(w, mask),
            -self.w_max, self.w_max)

    def re_program(self, w, delta_w):
        if self.quantize:
            delta_w = quantize_10(delta_w)

        return self.program(w + delta_w, mask=tf.cast(tf.where(delta_w == 0.0, False, True), tf.float32))

    def read(self):
        pass
