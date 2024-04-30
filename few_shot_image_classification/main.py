import argparse
import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# from omniglot import Omniglot, OmniglotEfficientTesting
from maml_trainee_omniglot import MAMLTraineeOmniglot, MAMLTraineeOmniglotConv
from maml_trainer import MAMLTrainer
from utils import str2bool, isodate
from collections import defaultdict
from checkpoint_loader import load_checkpoint_conv
from omniglot_v2 import OmniglotV2 as Omniglot

# TODO:
#   - check number of parameters (does CNN use biases?)


parser = argparse.ArgumentParser(description='Omniglot-MAML')
parser.add_argument('--experiment', default='omniglot_maml', type=str,
                    help='Name of the experiment for logging (default: omniglot_maml)')
# Input
parser.add_argument('--num_train_examples', type=int, default=5,
                    help='Number of training examples (default: 5)')
parser.add_argument('--num_test_examples', type=int, default=5,
                    help='Number of test examples (default: 5)')
parser.add_argument('--inner_classes', type=int, default=5,
                    help='Number of classes for each task (default: 5)')
parser.add_argument('--input_dim', type=int, default=28,
                    help='Width and height of the Omniglot images (default: 28')
# Model
parser.add_argument('--checkpoint', default='', type=str,
                    help='Load model weights and evaluate model from given checkpoint (default: '')')
parser.add_argument('--network_type', type=str, choices=["ANN", "CNN"], default="CNN",
                    help="Define the network model to use (default: CNN")
parser.add_argument('--num_neurons_per_layer', type=int, nargs='+', default=[200, 128, 64, 64],
                    help='Defines the number of neurons per layer if `network_type=ANN` (default: [200, 128, 64, 64]')
parser.add_argument('--hidden_channels', default=64, type=int,
                    help='The number of channels in the CNN layers if `network_type=CNN` (default: 64)')
parser.add_argument('--kernel_size', default=3, type=int,
                    help='The kernel size in the CNN layers if `network_type=CNN` (default: 3)')
parser.add_argument('--use_biases', default=True, type=str2bool, nargs='?', const=True,
                    help='Use bias terms in the network if `network_type=ANN` (default: True)')
parser.add_argument('--use_batch_norm', default=True, type=str2bool, nargs='?', const=True,
                    help='Use batch norm in the network if `network_type=ANN` (default: True)')
parser.add_argument('--use_feedback_align', default=False, type=str2bool, nargs='?', const=False,
                    help='Use feedback alignment in the network if `network_type=ANN` (default: False)')
parser.add_argument('--update_only_readout', default=True, type=str2bool, nargs='?', const=True,
                    help='In the inner-loop, update only the readout layer (default: True)')
parser.add_argument('--noise', default=True, type=str2bool, nargs='?', const=True,
                    help='Model noise in the model (default: True)')
# Optimization
parser.add_argument('--iterations', default=30000, type=int,
                    help='The number of iterations to train for (default: 30000)')
parser.add_argument('--test_iterations', type=int, default=500,
                    help='The number of test iterations (default: 500)')
parser.add_argument('--val_iterations', type=int, default=5,
                    help='The number of val iterations (default: 5)')
parser.add_argument('--validate_every', default=500, type=int,
                    help='Period of the validation runs in iterations (default: 500)')
parser.add_argument('--batch_size', default=40, type=int,
                    help='Mini-batch size (default: 40)')
parser.add_argument('--learning_rate_outer', default=0.001, type=float,
                    help='Outer-loop learning rate (default: 0.001)')
parser.add_argument('--learning_rate_inner', default=0.1, type=float,
                    help='Inner-loop learning rate (default: 0.1)')
parser.add_argument('--decay_learning_rate_outer', default=1.0, type=float,
                    help='Outer-loop learning rate decay factor (default: 1.0)')
parser.add_argument('--decay_learning_rate_outer_every', default=1000, type=int,
                    help='Period of the outer-loop learning rate decay in iterations (default: 1000)')
parser.add_argument('--inner_updates', type=int, default=4,
                    help='Number of gradient steps in the inner-loop (default: 4')
# Miscellaneous
parser.add_argument('--seed', default=None, type=int,
                    help='Seed for initializing training (default: None)')
parser.add_argument('--dataset_seed', default=42, type=int,
                    help='Seed for creating the dataset (default: 42)')
parser.add_argument('--store_every', default=250, type=int,
                    help='Period of storing model checkpoints in iterations; set to None to not store (default: 250)')

parser.add_argument('--quantize', default=False, type=str2bool, nargs='?', const=False)
parser.add_argument('--num_bits', default=8, type=int)
parser.add_argument('--stochastic_quantization', default=True, type=str2bool, nargs='?', const=True)


def main():
    args = parser.parse_args()

    # Set up the run
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print("Checkpoint not found. Exiting...")
            sys.exit()
        else:
            # run = Run(run_hash=os.path.basename(os.path.dirname(args.checkpoint)))

            run = dict()
    else:
        # run = Run(experiment=args.experiment)
        run = dict()
        run['hparams'] = {k: v for k, v in vars(args).items() if k not in ['experiment', 'checkpoint']}

    checkpoint_folder = os.path.join('checkpoints', args.experiment, isodate())
    run['checkpoint_folder'] = checkpoint_folder
    run['metrics'] = defaultdict(list)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    print(run)

    # Set the random seed for reproducible results
    if args.seed is not None:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    # Create the dataset
    dataset = Omniglot(
        num_test=args.num_test_examples,
        num_train=args.num_train_examples,
        inner_classes=args.inner_classes,
        seed=args.dataset_seed
    )

    # Create the model

    if args.network_type == "ANN":
        trainee = MAMLTraineeOmniglot(
            n_in=args.input_dim * args.input_dim,
            n_out=dataset.n_out,
            num_neurons_per_layer=args.num_neurons_per_layer,
            use_biases=args.use_biases,
            use_feedback_align=args.use_feedback_align,
            use_batch_norm=args.use_batch_norm,
            update_only_readout=args.update_only_readout,
            noise=args.noise,
        )
    elif args.network_type == "CNN":
        trainee = MAMLTraineeOmniglotConv(
            dim_in=args.input_dim,
            n_out=dataset.n_out,
            hidden_channels=args.hidden_channels,
            kernel_dims=args.kernel_size,
            noise=args.noise,
            quantize=args.quantize,
            num_bits=args.num_bits,
        )
    else:
        assert False, f"Unrecognised network type: {args.network_type}"

    trainer = MAMLTrainer(
        trainee=trainee,
        dataset=dataset,
        lr=args.learning_rate_outer,
        lr_inner=args.learning_rate_inner,
        inner_updates=args.inner_updates,
        lr_decay=args.decay_learning_rate_outer
    )

    # Optionally load checkpoint
    if args.checkpoint:
        load_checkpoint_conv(args.checkpoint, trainee)
        test(run=run, trainer=trainer, batch_size=args.batch_size, iterations=args.test_iterations,
             global_step=args.iterations - 1)
        with open(os.path.join(checkpoint_folder, 'metrics.pkl'), 'wb') as f:
            pickle.dump(run, f)
    else:
        trainee = train(run=run, trainer=trainer, batch_size=args.batch_size, iterations=args.iterations,
                        validate_every=args.validate_every, val_iterations=args.val_iterations,
                        decay_lr_every=args.decay_learning_rate_outer_every, store_every=args.store_every,
                        checkpoint_folder=checkpoint_folder)

        # pickle.dump(trainee.get_variables(), open(os.path.join(checkpoint_folder, run.hash + '-weights.pickle'), 'wb'))
        pickle.dump(trainee.get_variables(), open(os.path.join(checkpoint_folder, 'weights.pickle'), "wb"))
        with open(os.path.join(checkpoint_folder, 'metrics.pkl'), 'wb') as f:
            pickle.dump(run, f)

        test(run=run, trainer=trainer, batch_size=args.batch_size, iterations=args.test_iterations,
             global_step=args.iterations - 1)
        with open(os.path.join(checkpoint_folder, 'metrics.pkl'), 'wb') as f:
            pickle.dump(run, f)


def train(run, trainer, batch_size, iterations, validate_every, val_iterations, decay_lr_every, store_every,
          checkpoint_folder):
    progress = tqdm(range(iterations), desc='Training')

    for i in progress:

        if i != 0 and i % decay_lr_every == 0:
            trainer.decrease_learning_rate()

        loss, acc = trainer.do_stuff(batch_size, training=True)

        progress.set_postfix({'loss': loss})

        string_norm_data = ""
        for norm in trainer.trainee.norms:
            for var in norm.variables:
                string_norm_data = string_norm_data + f"{norm.name} ({var.name}): {tf.norm(var)} - "
            string_norm_data = string_norm_data + "\n"

        run['metrics']['train/acc'].append(acc)
        run['metrics']['train/loss'].append(loss)
        run['metrics']['train/string_norm_data'].append(string_norm_data)

        if store_every is not None and (i % store_every == 0 or i == iterations - 1):
            pickle.dump(trainer.trainee.get_variables(), open(os.path.join(checkpoint_folder, 'weights-it{0}.pickle').format(i), "wb"))

        if validate_every is not None and i != 0 and (i % validate_every == 0 or i == iterations - 1):
            validate(run=run, trainer=trainer, batch_size=batch_size, iterations=val_iterations, global_step=i,
                     checkpoint_folder=checkpoint_folder)

    return trainer.trainee


def validate(run, trainer, batch_size, iterations, global_step, checkpoint_folder):
    progress = tqdm(range(iterations), desc='Validating', leave=False)

    losses = []
    accuracies = []
    for _ in progress:
        loss, acc = trainer.do_stuff(batch_size, training=False)

        losses.append(loss)
        accuracies.append(acc)
        progress.set_postfix({'loss': np.mean(losses)})

    for norm in trainer.trainee.norms:
        for var in norm.variables:
            pickle.dump(var.numpy(), open(
                os.path.join(checkpoint_folder,
                             f'-{norm.name}-{var.name.split(":")[0]}-it{global_step}.pickle'), "wb"))

    run['metrics']['valid/loss'].append(np.mean(losses))
    run['metrics']['valid/acc'].append(np.mean(accuracies))


def test(run, trainer, batch_size, iterations, global_step):
    progress = tqdm(range(iterations), desc='Testing')

    losses = []
    accuracies = []
    for _ in progress:
        loss, acc = trainer.do_stuff(batch_size, training=False)
        losses.append(loss)
        accuracies.append(acc)
        progress.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accuracies)})

    run['metrics']['test/loss'].append(np.mean(losses))
    run['metrics']['test/acc'].append(np.mean(accuracies))
    print(run)


if __name__ == '__main__':
    main()
