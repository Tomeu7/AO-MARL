from src.reinforcement_learning.config.GlobalConfig import Config
from src.reinforcement_learning.helper_functions.utils.help_initialization import obtain_args
import torch.multiprocessing as mp
from src.reinforcement_learning.rpc_training.rpc_init import initialize_master_worker_paradigm

if __name__ == "__main__":

    config = Config()

    # b) Modify config file (it is easier to input them on args if you want to do multiple experiments)
    args = obtain_args(config)
    # h) Create trainer/test class and train/test
    if args.world_size > -1:
        # For RPC Inputs should be:
        # + args.world_size
        # + zernike_start_end
        # then it will divide in equals zernike_start_end

        mp.spawn(
            initialize_master_worker_paradigm,
            args=(args.world_size,),
            nprocs=args.world_size,
            join=True
        )
    else:
        raise NotImplementedError


