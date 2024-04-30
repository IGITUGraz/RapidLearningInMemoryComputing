import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_norms(norms, fig_size=(20, 20)):
    fig, ax = plt.subplots(len(norms), figsize=fig_size)

    for a, norm in zip(ax, norms):
        a.plot(norm)

    plt.tight_layout()
    plt.close(fig)

    return fig


def raster_plot(ax, spikes, line_width=1.2, **kwargs):

    n_t, n_n = spikes.shape
    event_times, event_ids = spikes.numpy().nonzero()

    max_spike = 10000
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]

    for n, t in zip(event_ids, event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=line_width, **kwargs)

    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([0, n_t])
    ax.set_yticks([0, n_n])
    del event_times
    del event_ids


def plot_spikes(spikes, fig_size=(10, 10)):
    fig, ax = plt.subplots(3, figsize=fig_size)

    ax[0].pcolormesh(spikes[0][0], cmap='binary')
    ax[0].title.set_text('LSG Training Spikes')
    ax[1].pcolormesh(spikes[1][0], cmap='binary')
    ax[1].title.set_text('Trainee Training Spikes')
    ax[2].pcolormesh(spikes[2][0], cmap='binary')
    ax[2].title.set_text('Trainee Testing Spikes')

    plt.tight_layout()
    plt.close(fig)

    return fig


def plot_learning_signals(learning_signals, fig_size=(10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.matshow(learning_signals[0].numpy().T)
    ax.title.set_text('LSG Learning Signals')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.close(fig)

    return fig


def plot_membrane_potentials(potentials, fig_size=(10, 10)):
    fig, ax = plt.subplots(2, figsize=fig_size)

    ax[0].plot(potentials[0][0])
    ax[0].title.set_text('Trainee Training Membrane Potentials')
    ax[1].plot(potentials[1][0])
    ax[1].title.set_text('Trainee Testing Membrane Potentials')

    plt.tight_layout()
    plt.close(fig)

    return fig


def plot_traces(traces, prefix, fig_size=(10, 10)):
    fig, ax = plt.subplots(2, figsize=fig_size)

    ax[0].plot(traces[0][0])
    ax[0].title.set_text(f'Trainee Training {prefix} Traces')
    ax[1].plot(traces[1][0])
    ax[1].title.set_text(f'Trainee Testing {prefix} Traces')

    plt.tight_layout()
    plt.close(fig)

    return fig


def plot_dz_dh(dz_dh, fig_size=(10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.matshow(dz_dh[0].numpy().T)
    ax.title.set_text('Trainee Training dz/dh')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.close(fig)

    return fig


def plot_coordinates(outputs, targets, num_timesteps, model_outputs_angles, fig_size=(10, 10), show=False):
    output_train = outputs['train']
    output_test = outputs['test']
    cartesian_train = outputs['cartesian_train']
    cartesian_test = outputs['cartesian_test']

    omega_targets = targets['omega']
    angle_targets = targets['angles']
    cartesian_targets = targets['cartesian']

    fig, ax = plt.subplots(1 + (output_train.shape[-1]), 2, figsize=fig_size)

    if cartesian_train.shape[-1] == 3:
        ax[0][0].remove()
        axis = fig.add_subplot(1 + (output_train.shape[-1]), 2, 1, projection='3d')
        axis.set_title('Training coordinates 1')
        axis.plot(np.asarray(cartesian_train)[0, :, 0], np.asarray(cartesian_train)[0, :, 1],
                  np.asarray(cartesian_train)[0, :, 2], label='model')
        axis.plot(np.asarray(cartesian_targets)[0, :, 0], np.asarray(cartesian_targets)[0, :, 1],
                  np.asarray(cartesian_targets)[0, :, 2], label='target')
        axis.legend(bbox_to_anchor=(1.3, 1.06))
    else:
        ax[0][0].set_title('Training coordinates 1')
        ax[0][0].plot(np.asarray(cartesian_train)[0, :, 0], np.asarray(cartesian_train)[0, :, 1], label='model')
        ax[0][0].plot(np.asarray(cartesian_targets)[0, :, 0], np.asarray(cartesian_targets)[0, :, 1], label='target')

        ax[0][0].set_xlabel('x')
        ax[0][0].set_ylabel('y')
        ax[0][0].legend()

    if cartesian_test.shape[-1] == 3:
        ax[0][1].remove()
        axis = fig.add_subplot(1 + (output_train.shape[-1]), 2, 2, projection='3d')
        axis.set_title('Test coordinates 1')
        axis.plot(np.asarray(cartesian_test)[0, :, 0], np.asarray(cartesian_test)[0, :, 1],
                  np.asarray(cartesian_test)[0, :, 2], label='model')
        axis.plot(np.asarray(cartesian_targets)[0, :, 0], np.asarray(cartesian_targets)[0, :, 1],
                  np.asarray(cartesian_targets)[0, :, 2], label='target')
        axis.legend(bbox_to_anchor=(1.3, 1.06))
    else:
        ax[0][1].set_title('Test coordinates 1')
        ax[0][1].plot(np.asarray(cartesian_test)[0, :, 0], np.asarray(cartesian_test)[0, :, 1], label='model')
        ax[0][1].plot(np.asarray(cartesian_targets)[0, :, 0], np.asarray(cartesian_targets)[0, :, 1], label='target')
        ax[0][1].set_xlabel('x')
        ax[0][1].set_ylabel('y')
        ax[0][1].legend()

    t = np.arange(0, num_timesteps)
    for i in range(output_train.shape[-1]):
        ax[1+i][0].set_title('Training trial Angles {0}'.format(i+1) if model_outputs_angles
                             else 'Training trial Angular velocities {0}'.format(i+1))
        ax[1+i][0].plot(t, np.asarray(output_train)[0, t, i], label='model')
        ax[1+i][0].plot(t, np.asarray(angle_targets if model_outputs_angles else omega_targets)[0, t, i],
                        label='target')
        ax[1+i][0].set_xlabel('Time in ms')
        ax[1+i][0].set_ylabel('Angles [rad]' if model_outputs_angles else 'Angular velocity [rad/s]')
        ax[1+i][0].legend()

    t = np.arange(0, num_timesteps)
    for i in range(output_test.shape[-1]):
        ax[1+i][1].set_title('Test trial Angles {0}'.format(i+1) if model_outputs_angles
                             else 'Test trial Angular velocities {0}'.format(i+1))
        ax[1+i][1].plot(t, np.asarray(output_test)[0, t, i], label='model')
        ax[1+i][1].plot(t, np.asarray(angle_targets if model_outputs_angles else omega_targets)[0, t, i],
                        label='target')
        ax[1+i][1].set_xlabel('Time in ms')
        ax[1+i][1].set_ylabel('Angles [rad]' if model_outputs_angles else 'Angular velocity [rad/s]')
        ax[1+i][1].legend()

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
