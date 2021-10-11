# AO-MARL
Implementation of Adaptive Optics control with Multi-Agent Model-Free Reinforcement Learning.

This work has been a collaboration of Barcelona Supercomputing Center, Paris Observatory and Universitat Politècnica de Catalunya for the RisingSTARS project.

<p align="middle">
  <img src="https://github.com/Tomeu7/AO-MARL/blob/main/img/Image1.png" width="150" />
  <img src="https://github.com/Tomeu7/AO-MARL/blob/main/img/Image2.png" width="150" />
  <img src="https://github.com/Tomeu7/AO-MARL/blob/main/img/Image3.jpg" width="150" />
  <img src="https://github.com/Tomeu7/AO-MARL/blob/main/img/Image4.png" width="150" />
</p>

## Requirements

+ Compass 5.1.0 (https://github.com/ANR-COMPASS/shesha)
+ Pytorch

Note: tested on both Compass 5.0.0 and Compass 5.1.0 on IBM Power9 8335-GTH CPU (40 cores) with a GPU NVIDIA V100 (16 GB).

## Brief explanation of the MARL implementation

1. The code implements a training loop of the Multi-Agent system in .../train_rpc.py. The training loop is parallel in the update thanks to the Pytorch RPC framework.
2. The training loop interacts with an environment constructed in .../ao_env.py, which builds the state via communicating with COMPASS in .../rlSupervisor.py.
3. The training loop uses a buffer from .../delayed_mdp.py to assign the tuples (state, action, next state, reward) correctly in regards to the delay.
4. The RL agents used consist of instances of Soft Actor Critic (SAC). The implementation of SAC has been extracted from https://github.com/pranz24/pytorch-soft-actor-critic and modified for AO needs, parallelisation and few minor general updates.

## Directory structure

```
project
│   README.md
│   main.py  # Script for training a RL controller.
└───src
│   └───reinforcement_learning # RL folder.
│       └───rpc_training # Training loop folder.
│       └───helper_functions # Helper functions and preprocessing folder.
│       └───environment # Environment folder.
│       └───config # RL configuration folder.
│   └───autoencoder # Autoencoder folder.
│   └───error_budget # Error budget loop folder.
└───shesha # Shesha package to execute COMPASS with python.
│       └───Supervisor # Scripts to communicate with COMPASS.
│       │       │RlSupervisor.py  Supervisor modifcation for a RL problem.
│       │       │...
│       │   ...
└───guardians # guardians package which ROKET forms part of.
│       │   roket_generalised_rl.py  # ROKET modification for a RL agent.
│       │   ...
└───data # Folder of parameter files for different simulations.
```

## Usage

### 1. Training

To train a policy you need to run main.py. The following arguments are required for its execution:

+ experiment_name: the name of the current experiment that will result in a folder with the same name.
+ parameter_telescope: which parameter file is used. All parameters files can be accessed in data/par/par4rl/.
+ world-size: number of agents minus one available. As the tip-tilt is controlled by a single-agent you will need 2 agents as the minimum (unless you do not control the tip-tilt which you can do by setting the argument --control_tip_tilt "False")
+ seed: the initial seed used for the experiment.
+ n_reverse_filtered_from_cmat: number of filtered low sensitivity modes.
+ n_zernike_start_end: number of modes controlled (without considering tip tilt). If you demand two agents and 80 modes, each agent will control 40 modes.
+ window_n_zernike: to build the "windowed" controller present in the paper.
+ port: port number required for the parallel execution.
+ num-gpus. The number of GPUs used, default 1. (Note: we recommend one GPU per approx 10 agents)

For instance, for 2 agents one controlling the tip-tilt and one controlling 80 modes for a 2m telescope with 10x10 subapertures you need to write:

```
python main.py --parameters_telescope "production_sh_10x10_2m.py" --world-size 3 --seed 1234 --experiment_name "training" --n_reverse_filtered_from_cmat 5 --n_zernike_start_end 0  80 --port 25003
```

The run will generate a performance file in output that you will be able to extract the training curves from. Also, the weights of both policies and critics from SAC will be saved every certain number of episodes.

### 2. Testing policies with ROKET

To obtain results similar to those in the paper, we provide the weights of the agents used in the article in the following google drive link https://drive.google.com/drive/folders/1LVsAGOvhu8mwSwR0v8rnH6GRArYxYCNo?usp=sharing.
You must add them to output/output_models/models_rpc/"name_of_the_experiment".

```
python src/error_budget/error_budget_multiple_agents.py
```

The simulation will generate a h5 file with all the results, as a default the experiment_name is the same as in the paper with the name "14_8_3l_dir_w20". To obtain plots similar to the ones in the paper you can write:

```
python
from src.error_budget.plot_results_error_budget import get_result_breakdown
get_result_breakdown("14_8_3l_dir_w20_steps_1000.h5")
```

### 3. Training an Autoencoder for denoising the WFS images

To obtain a working autoencoder you need to do 3 main steps.

a) Obtain the data.

To obtain the data for training run:

```
python src/autoencoder/obtain_dataset_autoencoder.py --parameter_file "production_sh_40x40_8m_3layers_d0_noise.py"
```

This is script is created for parameters files that have a 8m telescope with 40x40 SH-WFS, you can modify the obtain_config function for other configurations. As you may remember from the article, the dataset is created with a 0 delay. Also, you can provide a number of low sensitivity modes to filter out if desired as an argument with num_filtered. Be careful regarding disk space as the resulting dataset is a numpy array with a large size.

b) Train it

For training, we have included a separate repository where we explain in more detail the autoencoder process and how to train it.

https://github.com/Tomeu7/Denoising-wavefront-sensor-image-with-deep-neural-networks

c) Inject it in the MARL training with delay 2.

You can do that just by including the argument autoencoder_path in the MARL training. We have included a pretrained autoencoder trained with data from a closed-loop with a guide star of magnitude 9 with photon noise and 3 RMS e- readout noise. You can find the weights in output/autoencoder/autoencoder_weights/autoencoder_M9_rms_3.
For instance, for an 8m telescope with 40x40 SH-WFS and 43 agents and 4 GPUs:

```
python main.py --parameters_telescope "production_sh_40x40_8m_3layers_d1_noise.py" --world-size 44 --seed 1234 --experiment_name "training_autoencoder_8m_noise" --n_reverse_filtered_from_cmat 5 --n_zernike_start_end 0  1260 --port 25003 --autoencoder_path "output/autoencoder/autoencoder_weights/autoencoder_M9_rms_3" --num-gpus 4
```

Note that before training you must reoptimise the gain for the parameter file with an autoencoder. In the given parameter files the gain is already optimised.

### 4. Obtain optimal gain and number of modes to filter

For any new parameter file you want to create you need to optimise the gain and number of modes to filter. An example script to find the approximate optimal gain and number of low sensitivity modes is included in ../preprocessing_script.py. You may adapt that script for your purposes or just run it with:

```
python main.py src/reinforcement_learning/helper_functions/preprocessing/preprocessing_script.py --parameter_file "production_sh_10x10_2m.py"
```

## Acknowledgments

We would like to thank user pranz24 for providing a working version of Soft Actor Critic in Pytorch in https://github.com/pranz24/pytorch-soft-actor-critic.
