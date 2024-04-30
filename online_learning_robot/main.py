import argparse
import os
import pickle
import random
import sys

import matplotlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from datasets.trajectories import BrownianTrajectories, BrownianTrajectoriesScorbot
from functional.lif import LIFParameters
from model.l2l_trainee_lsg import RegularizationParameters, L2LTraineeLSG
from utils.utils import str2bool
from utils.plots import plot_spikes, plot_coordinates, plot_learning_signals, plot_membrane_potentials, plot_traces, \
    plot_dz_dh

matplotlib.use("Agg")

parser = argparse.ArgumentParser(description='RobotArm-L2L')
parser.add_argument('--experiment', default='robot_arm_l2l', type=str,
                    help='Name of the experiment for logging (default: robot_arm_l2l)')
# Input
parser.add_argument('--trajectory_length', default=249, type=int,
                    help='Length of trajectories in ms (default: 249)')
parser.add_argument('--lsg_timesteps_per_trainee_timestep', default=1, type=int,
                    help='Number of learning signal generator timesteps per trainee timestep (default: 1)')
parser.add_argument('--clock_input_size', default=5, type=int,
                    help='Number of neurons for the clock-like input signal (default: 5)')
parser.add_argument('--coordinate_input_size', default=16, type=int,
                    help='Number of encoding neurons per joint for coordinate input signal (default: 16)')
parser.add_argument('--num_joints', default=2, type=int,
                    help='The number of joints of the robot (default: 2)')
parser.add_argument('--joints', default=[0, 1], type=int,
                    help='The joints of Scorbot to use (default: [0, 1])')
parser.add_argument('--joint_length', default=0.5, type=float,
                    help='The length of the robot joints (default: 0.5)')
parser.add_argument('--window_length', default=120, type=int,  # 50 for angles
                    help='Length of the Hann window for smoothing the Brownian motion (default: 50)')
parser.add_argument('--step_size_brownian', default=0.01, type=float,
                    help='Time step used for generating the Brownian motion (default: 0.01)')
parser.add_argument('--use_scorbot', default=True, type=str2bool, nargs='?', const=False,
                    help='Include Scorbot`s kinematics into the simulation (default: False)')

# Model
parser.add_argument('--checkpoint', default='', type=str,
                    help='Load model weights and evaluate model from given checkpoint (default: '')')
parser.add_argument('--lsg_size', default=800, type=int,
                    help='Number of neurons in the learning signal generator (default: 800)')
parser.add_argument('--trainee_size', default=250, type=int,
                    help='Number of neurons in the trainee model (default: 250)')
parser.add_argument('--noise_trainee', default=True, type=str2bool, nargs='?', const=True,
                    help='Model noise in the trainee model (default: True)')
parser.add_argument('--quantize_trainee', default=True, type=str2bool, nargs='?', const=True,
                    help='Use quantized weight updates in the trainee model (default: True)')
parser.add_argument('--trainee_w_out_scaling_factor', default=1.0, type=float,
                    help='Scaling factor for the output weights of the trainee (default: 1.0)')
parser.add_argument('--smooth_output', default=False, type=str2bool, nargs='?', const=False,
                    help='Smooth model output by using a Hann window (default: False)')
parser.add_argument('--smooth_output_window_length', default=20, type=int,
                    help='Length of the Hann window for smoothing the models output (default: 30)')
parser.add_argument('--output_angles', default=False, type=str2bool, nargs='?', const=False,
                    help='Train the network to output angles instead of angular velocities (default: False)')
parser.add_argument('--lsg_output_size', default=None, type=int,
                    help='Project learning signals using a layer of size `lsg_output_size` with ReLU activation;'
                         'if None, use learning signals directly (default: None)')
parser.add_argument('--alif_fraction_lsg', default=0.3, type=float,
                    help='Fraction of neurons with adaptive threshold in the learning signal generator (default: 0.3)')

# Neurons
parser.add_argument('--lsg_thr', default=1.3, type=float,
                    help='Threshold of the learning signal generator neurons (default: 1.3)')
parser.add_argument('--lsg_tau', default=20.0, type=float,
                    help='Membrane time constant of the learning signal generator neurons (default: 20.0)')
parser.add_argument('--lsg_tau_o', default=20.0, type=float,
                    help='Decay time constant of the learning signal generator output neurons (default: 20.0)')
parser.add_argument('--lsg_dampening_factor', default=0.3, type=float,
                    help='Dampening factor for the spike-function in the learning signal generator (default: 0.3)')
parser.add_argument('--lsg_refractory_time', default=5, type=int,
                    help='Refractory time steps of the learning signal generator neurons (default: 5)')
parser.add_argument('--trainee_thr', default=0.6, type=float,
                    help='Threshold of the trainee neurons (default: 0.6)')
parser.add_argument('--trainee_tau', default=20.0, type=float,
                    help='Membrane time constant of the trainee neurons (default: 20.0)')
parser.add_argument('--trainee_tau_o', default=20.0, type=float,
                    help='Decay time constant of the trainee output neurons (default: 20.0)')
parser.add_argument('--trainee_dampening_factor', default=0.3, type=float,
                    help='Dampening factor for the spike-function in the trainee (default: 0.3)')
parser.add_argument('--trainee_refractory_time', default=5, type=int,
                    help='Refractory time steps of the trainee neurons (default: 5)')

# Optimization
parser.add_argument('--iterations', default=100000, type=int,
                    help='The number of iterations to train for (default: 100000)')
parser.add_argument('--validate_every', default=10000, type=int,
                    help='Period of the validation runs in iterations (default: 10000)')
parser.add_argument('--batch_size', default=90, type=int,
                    help='Mini-batch size (default: 90)')
parser.add_argument('--learning_rate_outer', default=1.5e-3, type=float,
                    help='Outer-loop learning rate (default: 1.5e-3)')
parser.add_argument('--learning_rate_inner', default=0.0001, type=float,
                    help='Inner-loop learning rate (default: 0.0001)')
parser.add_argument('--decay_learning_rate_outer', default=0.99, type=float,
                    help='Outer-loop learning rate decay factor (default: 0.99)')
parser.add_argument('--decay_learning_rate_outer_every', default=500, type=int,
                    help='Period of the outer-loop learning rate decay in iterations (default: 500)')
parser.add_argument('--regularization_factor_lsg', default=0.25, type=float,
                    help='Spike rate regularization factor for the learning signal generator (default: 0.0)')
parser.add_argument('--regularization_factor_trainee', default=0.25, type=float,
                    help='Spike rate regularization factor for the trainee (default: 0.0)')
parser.add_argument('--target_rate_lsg', default=10.0, type=float,
                    help='Target firing rate of the learning signal generator neurons in Hz (default: 10.0)')
parser.add_argument('--target_rate_trainee', default=20.0, type=float,
                    help='Target firing rate of trainee neurons in Hz (default: 10.0)')

# Miscellaneous
parser.add_argument('--seed', default=None, type=int,
                    help='Seed for initializing training (default: None)')
parser.add_argument('--dataset_seed', default=42, type=int,
                    help='Seed for creating the dataset (default: 42)')
parser.add_argument('--plot_every', default=250, type=int,
                    help='Period of plot creation in iterations; set to None to not create plots (default: 250)')
parser.add_argument('--store_every', default=250, type=int,
                    help='Period of storing model checkpoints in iterations; set to None to not store (default: 250)')

# TODO get last step from saved run


def main():
    args = parser.parse_args()
    print("ARGS")
    print(args)

    # Set up the run
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print("Checkpoint not found. Exiting...")
            sys.exit()
        else:
            run = dict(run_hash=os.path.basename(os.path.dirname(args.checkpoint)))
    else:
        run = dict(experiment=args.experiment)
        run['hparams'] = {k: v for k, v in vars(args).items() if k not in ['experiment', 'checkpoint']}

    checkpoint_folder = os.path.join('checkpoints', args.experiment, run['run_hash'])
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # Set the random seed for reproducible results
    if args.seed is not None:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    # Create the dataset
    if args.use_scorbot:
        dataset = BrownianTrajectoriesScorbot(
            batch_size=args.batch_size,
            trajectory_length=args.trajectory_length,
            clock_input_size=args.clock_input_size,
            coordinate_input_size=args.coordinate_input_size,
            num_joints=args.num_joints,
            joints=args.joints,
            lsg_timesteps_per_trainee_timestep=args.lsg_timesteps_per_trainee_timestep,
            window_length=args.window_length,
            step_size=args.step_size_brownian,
            seed=args.dataset_seed
        )
    else:
        dataset = BrownianTrajectories(
            batch_size=args.batch_size,
            trajectory_length=args.trajectory_length,
            clock_input_size=args.clock_input_size,
            coordinate_input_size=args.coordinate_input_size,
            num_joints=args.num_joints,
            joint_length=args.joint_length,
            lsg_timesteps_per_trainee_timestep=args.lsg_timesteps_per_trainee_timestep,
            window_length=args.window_length,
            step_size=args.step_size_brownian,
            seed=args.dataset_seed
        )

    # Create the model
    lsg_params = LIFParameters(
        thr=args.lsg_thr,
        tau=args.lsg_tau,
        tau_o=args.lsg_tau_o,
        damp=args.lsg_dampening_factor,
        n_ref=args.lsg_refractory_time
    )
    trainee_params = LIFParameters(
        thr=args.trainee_thr,
        tau=args.trainee_tau,
        tau_o=args.trainee_tau_o,
        damp=args.trainee_dampening_factor,
        n_ref=args.trainee_refractory_time
    )
    reg_params = RegularizationParameters(
        target_trainee_f=args.target_rate_trainee,
        target_lsg_f=args.target_rate_lsg,
        lambda_trainee=args.regularization_factor_trainee,
        lambda_lsg=args.regularization_factor_lsg
    )

    model = L2LTraineeLSG(
        dataset=dataset,
        clock_input_size=args.clock_input_size,
        coordinate_input_size=args.coordinate_input_size,
        lsg_size=args.lsg_size,
        trainee_size=args.trainee_size,
        output_size=args.num_joints,
        lsg_output_size=args.lsg_output_size,
        noise_trainee=args.noise_trainee,
        quantize_trainee=args.quantize_trainee,
        learning_rate_outer=args.learning_rate_outer,
        learning_rate_inner=args.learning_rate_inner,
        decay_learning_rate_outer=args.decay_learning_rate_outer,
        alif_fraction_lsg=args.alif_fraction_lsg,
        trainee_w_out_scaling_factor=args.trainee_w_out_scaling_factor,
        smooth_output=args.smooth_output,
        smooth_output_window_length=args.smooth_output_window_length,
        output_angles=args.output_angles,
        use_scorbot=args.use_scorbot,
        scorbot_joints=args.joints,
        lsg_params=lsg_params,
        trainee_params=trainee_params,
        reg_params=reg_params
    )

    # Optionally load checkpoint
    if args.checkpoint:
        weights = pickle.load(open(args.checkpoint, 'rb'))

        model.set_weights(weights)

        test(model=model, run=run, iterations=30, output_angles=args.output_angles, plot_every=2,
             checkpoint_folder=checkpoint_folder)
    else:
        model = train(model=model, run=run, iterations=args.iterations, output_angles=args.output_angles,
                      validate_every=args.validate_every, decay_lr_every=args.decay_learning_rate_outer_every,
                      plot_every=args.plot_every, store_every=args.store_every, checkpoint_folder=checkpoint_folder)

        with open(os.path.join(checkpoint_folder, run['run_hash'] + '-weights.pickle'), 'wb') as handle:
            pickle.dump(model.get_variables(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        test(model=model, run=run, iterations=30, output_angles=args.output_angles, plot_every=2,
             checkpoint_folder=checkpoint_folder, global_step=args.iterations - 1)


def train(model, run, iterations, output_angles, validate_every, decay_lr_every, plot_every, store_every,
          checkpoint_folder):
    progress = tqdm(range(iterations), desc='Training')

    for i in progress:

        if i != 0 and i % decay_lr_every == 0:
            model.decrease_learning_rate()

        _, output_test, output_train, cartesian_training, cartesian_test, targets, spikes, loss = model.train()

        progress.set_postfix({'loss': loss.numpy()})

        if store_every is not None and (i % store_every == 0 or i == iterations - 1):
            with open(os.path.join(checkpoint_folder, run.hash + '-weights-it{0}.pickle').format(i), 'wb') as handle:
                pickle.dump(model.get_variables(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        if plot_every is not None and (i % plot_every == 0 or i == iterations - 1):
            spike_fig = plot_spikes(spikes)
            coordinate_fig = plot_coordinates(
                outputs={'train': output_train, 'test': output_test,  # are either angles or angular velocities
                         'cartesian_train': cartesian_training, 'cartesian_test': cartesian_test},
                targets={'omega': targets[0], 'angles': targets[1], 'cartesian': targets[2]},
                model_outputs_angles=output_angles,
                num_timesteps=cartesian_training.shape[1])


        if validate_every is not None and i != 0 and (i % validate_every == 0 or i == iterations - 1):
            validate(model=model, run=run, iterations=100, output_angles=output_angles, global_step=i, plot_every=10)

    return model


def validate(model, run, iterations, output_angles, global_step, plot_every):
    progress = tqdm(range(iterations), desc='Validating', leave=False)

    losses = []
    spike_figs = []
    coordinate_figs = []
    for i in progress:
        _, output_test, output_train, cartesian_training, cartesian_test, targets, spikes, _, _, _, _, _, loss = \
            model.test()

        losses.append(loss)
        progress.set_postfix({'loss': np.mean(losses)})

        if plot_every is not None and (i % plot_every == 0 or i == iterations - 1):
            spike_fig = plot_spikes(spikes)
            coordinate_fig = plot_coordinates(
                outputs={'train': output_train, 'test': output_test,  # are either angles or angular velocities
                         'cartesian_train': cartesian_training, 'cartesian_test': cartesian_test},
                targets={'omega': targets[0], 'angles': targets[1], 'cartesian': targets[2]},
                model_outputs_angles=output_angles,
                num_timesteps=cartesian_training.shape[1])




def test(model, run, iterations, output_angles, plot_every, checkpoint_folder, global_step=None):
    progress = tqdm(range(iterations), desc='Testing')

    losses = []
    dataset = []
    spike_list = []
    spike_figs = []
    coordinate_figs = []
    coordinate_list = []
    learning_signal_figs = []
    learning_signal_list = []
    membrane_potential_figs = []
    membrane_potential_list = []
    input_trace_figs = []
    input_trace_list = []
    recurrent_trace_figs = []
    recurrent_trace_list = []
    dz_dh_figs = []
    dz_dh_list = []
    for i in progress:
        data = model.dataset.get_data()

        coordinates_encoded, coordinates, angular_velocities, angles, mask = data

        dataset.append(
            [coordinates_encoded[0][tf.newaxis, ...], coordinates[0][tf.newaxis, ...],
             angular_velocities[0][tf.newaxis, ...], angles[0][tf.newaxis, ...],
             model.clock_signal_trainee[0][tf.newaxis, ...]])

        _, output_test, output_train, cartesian_training, cartesian_test, targets, spikes, potentials, input_traces, \
            recurrent_traces, dz_dh, learning_signals, loss = model.test(data=data)

        learning_signal_list.append(learning_signals)
        spike_list.append({'lsg': spikes[0], 'trainee_train': spikes[1], 'trainee_test': spikes[2]})
        membrane_potential_list.append({'trainee_train': potentials[0], 'trainee_test': potentials[1]})
        input_trace_list.append({'trainee_train': input_traces[0], 'trainee_test': input_traces[1]})
        recurrent_trace_list.append({'trainee_train': recurrent_traces[0], 'trainee_test': recurrent_traces[1]})
        dz_dh_list.append(dz_dh)
        coordinate_list.append({
            'outputs': {
                'train': output_train,
                'test': output_test,
                'cartesian_train': cartesian_training,
                'cartesian_test': cartesian_test
            },
            'targets': {
                'omega': targets[0],
                'angles': targets[1],
                'cartesian': targets[2]
            }
        })

        losses.append(loss)
        progress.set_postfix({'loss': np.mean(losses)})

        if plot_every is not None and (i % plot_every == 0 or i == iterations - 1):
            spike_fig = plot_spikes(spikes)
            dz_dh_fig = plot_dz_dh(dz_dh)
            input_trace_fig = plot_traces(input_traces, prefix='Input')
            recurrent_trace_fig = plot_traces(recurrent_traces, prefix='Recurrent')
            learning_signal_fig = plot_learning_signals(learning_signals)
            membrane_potential_fig = plot_membrane_potentials(potentials)
            coordinate_fig = plot_coordinates(
                outputs={'train': output_train, 'test': output_test,  # are either angles or angular velocities
                         'cartesian_train': cartesian_training, 'cartesian_test': cartesian_test},
                targets={'omega': targets[0], 'angles': targets[1], 'cartesian': targets[2]},
                model_outputs_angles=output_angles,
                num_timesteps=cartesian_training.shape[1])


    print('Test loss: ', np.mean(losses))

    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-dataset-test.pickle'), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-spikes-test.pickle'), 'wb') as handle:
        pickle.dump(spike_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-coordinates-test.pickle'), 'wb') as handle:
        pickle.dump(coordinate_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-learning_signals-test.pickle'), 'wb') as handle:
        pickle.dump(learning_signal_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-membrane_potentials-test.pickle'), 'wb') as handle:
        pickle.dump(membrane_potential_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-input_traces-test.pickle'), 'wb') as handle:
        pickle.dump(input_trace_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-recurrent_traces-test.pickle'), 'wb') as handle:
        pickle.dump(recurrent_trace_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_folder, run['run_hash'] + '-dz_dh-test.pickle'), 'wb') as handle:
        pickle.dump(dz_dh_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
