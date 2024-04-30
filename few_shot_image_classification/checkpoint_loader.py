import pickle as pkl
from maml_trainee_omniglot import MAMLTraineeOmniglotConv
import numpy as np

def load_checkpoint_conv(path, trainee: MAMLTraineeOmniglotConv):
    with open(path, 'rb') as f:
        weights = pkl.load(f)

    trainee.kernel_1.load(weights[0])
    trainee.kernel_2.load(weights[1])
    trainee.kernel_3.load(weights[2])
    trainee.kernel_4.load(weights[3])
    trainee.w_readout.load(weights[4])

    for i, w in enumerate(weights):
        print(f'Weight {i}', np.abs(w).sum())



def load_checkpoint_conv_only(path, trainee: MAMLTraineeOmniglotConv):
    with open(path, 'rb') as f:
        weights = pkl.load(f)

    trainee.kernel_1.load(weights[0])
    trainee.kernel_2.load(weights[1])
    trainee.kernel_3.load(weights[2])
    trainee.kernel_4.load(weights[3])

    # The readout weights are not loaded!
    # trainee.w_readout.load(weights[4])