# Rapid learning with phase-change memory-based in-memory computing through learning-to-learn

This is the code repository for the paper

Rapid learning with phase-change memory-based in-memory computing through learning-to-learn  
*Thomas Ortner, Horst Petschenig, Athanasios Vasilopoulos, Roland Renner, Spela Brglez, Thomas Limbacher, Enrique Pinero, Alejandro Linares Barranco, Angeliki Pantazi, Robert Legenstein*  
[ArXiv Link](https://arxiv.org/abs/2405.05141)

## Setup
You need [Tensorflow](https://www.tensorflow.org/) to run this code. We used Python 3.9 and TensorFlow 2.5. See the corresponding [Conda](https://docs.conda.io/en/latest/) `environment.yml` file to install all necessary dependencies: 

    conda env create --file=environment.yml --name RapidLearningInMemoryComputing
    conda activate RapidLearningInMemoryComputing
    pip install --no-warn-conflicts -r requirements.txt

## Usage

### Few-shot image classification with PCM-based neuromorphic hardware
To start training on the few-shot image classification task, run 

    cd few_shot_image_classification
    python main.py --seed 1234 --batch_size=32 --hidden_channels=56 --noise=False

To load an existing checkpoint, run

    cd few_shot_image_classification
    python main_omniglot.py --checkpoint checkpoints/pretrained-weights.pickle --seed 42  --batch_size=1 --hidden_channels=56 --noise=False --dataset_seed=128

### Rapid online learning of robot arm trajectories in biologically-inspired neural networks

To start training on the robotic arm online learning task, run 

    cd online_learning_robot
    python main.py

To load an existing checkpoint, run 
    
    cd online_learning_robot
    python main.py --checkpoint checkpoints/pretrained-weights.pickle

## Acknowledgements
This work was funded in part by the CHIST-ERA grant CHIST-ERA-18-ACAI-004, by the Austrian Science Fund (FWF) [10.55776/I4670-N], by grant PCI2019-111841-2 funded by MCIN/AEI/ 10.13039/501100011033, by SNSF under the project number 20CH21_186999 / 1 and by the European Union. For the purpose of open access, the author has applied a CC BY public copyright licence to any Author Accepted Manuscript version arising from this submission. This work was supported by NSF EFRI grant #2318152.
E. P.-F. work was supported by a "Formación de Profesorado Universitario" Scholarship, with reference number FPU19/04597 from the Spanish Ministry of Education, Culture and Sports. Furthermore, we thank the In-Memory Computing team at IBM for their technical support with the PCM-based NMHW as well as the IBM Research AI Hardware Center. Moreover, we thank Joris Gentinetta for his help with the setup for the robotic arm experiments.
