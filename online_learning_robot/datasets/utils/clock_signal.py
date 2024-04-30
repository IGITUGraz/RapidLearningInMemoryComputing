import numpy as np


def clock_signal(seq_len, n_in, step_len, step_group, spike_every):
    step_len = step_len * spike_every
    input_spikes = np.zeros((int(seq_len), n_in))
    n_of_steps = seq_len//step_len
    if n_of_steps == 0:
        n_of_steps = 1

    neuron_pointer = 0
    step_pointer = 0

    for i in range(0, seq_len, n_of_steps):
        for j in range(step_group):
            local_step_pointer = step_pointer
            for k in range(step_len):
                if k % spike_every == 0:
                    input_spikes[local_step_pointer][neuron_pointer] = 1
                local_step_pointer += 1
                if local_step_pointer >= seq_len:
                    if j+1 == step_group:
                        return input_spikes
                    else:
                        break
            neuron_pointer += 1
            if neuron_pointer >= n_in:
                neuron_pointer = 0
            if j+1 == step_group:
                step_pointer = local_step_pointer

    return input_spikes
