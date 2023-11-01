# run 2023/10/24 10:19
import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logger import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath


def run(_run, _config, _log):
    # 此处进行参数检测
    # Begin:

    # End;
    args = SN(**_config)
    _log.info(args)

    args.device = "cuda" if args.use_cuda else "cpu"
    logger = Logger(_log)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def run_sequential(args, logger):
    # this function run the main loop of the algorithm

    time_steps = []
    time_step_to_load = 0

    # load checkpoint
    # every checkpoint file is named by the time step number at that moment
    if args.checkpoint_path != "":

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directory {} doesn't exist".format(args.checkpoint_path))
            return
        for name in os.listdir(args.checkpoint_path):
            full_path = os.path.join(args.checkpoint_path, name)
            if os.path.isdir(full_path) and name.isdigit():
                time_steps.append(int(name))
        if args.load_step == 0:
            time_step_to_load = max(time_steps)
        else:
            # choose the time step closest to load_step
            time_step_to_load = min(time_steps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(time_step_to_load))
        logger.console_logger.info("Loading model from {}".format(model_path))

        learner.load_models(model_path)
        runner.t_env = time_step_to_load

        if args.evaluate or args.save_replay:
            # evaluate mode
            evaluate_sequential(args, runner)
            return

    # train mode
    episode = 0

    model_save_time = 0
    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} time steps".format(args.t_max))

    while runner.t_env <= args.t_max:

        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)





