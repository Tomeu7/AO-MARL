from src.reinforcement_learning.helper_functions.preprocessing.obtain_gain.obtain_best_gain_and_modes_filtered import obtain_modes_filtered_and_gain

import argparse

if __name__ == "__main__":

    num_episodes = 1
    num_steps = 1000

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter_file', type=str, default='production_sh_10x10_2m.py')
    parser.add_argument('--pure_delay_0', type=str, default="False")
    parser.add_argument('--autoencoder_path', type=str, default=None)

    args = parser.parse_args()

    pure_delay_0 = True if args.pure_delay_0 == "True" else False
    process_dictionary = {args.parameter_file: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]}

    for parameters, gains_list in process_dictionary.items():

        output_name = parameters.split("/")[0]
        parameters_ = parameters
        obtain_modes_filtered_and_gain(parameter_file=parameters_,
                                       pure_delay_0=True if args.pure_delay_0 == "True" else False,
                                       use_autoencoder=True if args.autoencoder_path is not None else False,
                                       autoencoder_path=args.autoencoder_path,
                                       gains=gains_list,
                                       output_name=output_name,
                                       num_episodes=num_episodes,
                                       num_steps=num_steps)

