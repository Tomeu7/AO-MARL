import torch.distributed.rpc as rpc
import os
from src.reinforcement_learning.helper_functions.utils.help_initialization import obtain_args, print_and_assertions
from src.reinforcement_learning.config.GlobalConfig import Config
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

MASTER_NAME = "Compass"
AGENT_NAME = "Agent{}"


def initialize_master_worker_paradigm(rank,
                                      world_size):
    config = Config()

    args = obtain_args(config)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    # TODO changed

    if rank == 0:

        from src.reinforcement_learning.rpc_training.train_rpc import TrainerRPC

        # config = Config()

        # b) Modify config file (it is easier to input them on args if you want to do multiple experiments)
        # args = obtain_args(config)

        experiment_name = args.experiment_name
        seed = args.seed
        config.update_conf_with_args(args)

        # e) Set up the initial seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # f) Check the availability of configurations and do some prints
        print_and_assertions(config, seed)

        # g) Create summary writer
        writer_performance = SummaryWriter('output/runs/performance/performance_' + experiment_name)
        writer_metrics_1 = SummaryWriter('output/runs/metrics_1/metrics_' + experiment_name)

        rpc.init_rpc(MASTER_NAME, rank=rank, world_size=world_size)
        trainer = TrainerRPC(config_rl=config,
                             writer_performance=writer_performance,
                             writer_metrics_1=writer_metrics_1,
                             seed=seed,
                             experiment_name=experiment_name,
                             world_size=world_size,
                             num_gpus=args.num_gpus)
        trainer.train_agent()
    else:
        config = Config()
        args = obtain_args(config)
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # other ranks are the observer
        rpc.init_rpc(AGENT_NAME.format(rank), rank=rank, world_size=world_size)

    rpc.shutdown()
