import datetime
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from agent import  REGISTRY as multiagent_REGISTRY


def new_run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if (args.name == "pr_q"):
        scheme["ensemble_mask"] = {"vshape": (args.critic_ensemble_num,)}
        scheme["explore_flag"]={"vshape": (1,)}
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup agent
    multiagent=multiagent_REGISTRY[args.multiagent](args,buffer,groups,logger)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, agent=multiagent)

    if args.use_cuda:
        multiagent.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        multiagent.load_models(model_path)
        runner.t_env = timestep_to_load
        if args.q_table:
            episode_batch = runner.run(test_mode=True)
            buffer.insert_episode_batch(episode_batch)
            episode_sample = buffer.sample(1)
            multiagent.print_q_table(episode_sample)
            return
        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            multiagent.train(runner.t_env, episode)

            if(args.q_table):
                episode_sample_for_q_table = buffer.sample(1)
                print(" runner.t_env:", runner.t_env)
                multiagent.print_q_table(episode_sample_for_q_table)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            multiagent.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:

            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    if(args.q_table):
        multiagent.learner.save_q_table()
    runner.close_env()
    logger.console_logger.info("Finished Training")



def run(_run, _config, _log, pymongo_client=None):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")



    # configure tensorboard logger
    unique_token=None

    if(args.env=="sc2"):
        unique_token = "{}__{}__{}__{}".format( datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args.name,args.env,args.env_args['map_name'])
    elif(args.env=="stag_hunt") :
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                               datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif(args.env=="matrix_game"):
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif (args.env == "matrix_game_v1"):
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif (args.env == "matrix_game_v2"):
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif (args.env == "matrix_game_abnormal"):
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif (args.env == "join1"):
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif (args.env == "simple_game"):
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    elif (args.env == "matrix_game_with_huge_action_space"):
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        unique_token = "{}__{}__{}".format(args.name, args.env,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default



    logger.setup_sacred(_run)


    # Run and train
    if(not args.agent_module_frame):
        run_sequential(args=args, logger=logger)
    else:
        new_run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits

    if(os.sys.platform.startswith("win")):
        EX_OK=0
    else:
        EX_OK=os.EX_OK
    os._exit(EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

#stored process
t_env_list=[]
buffer_list=[]
episode_batch_list=[]


mac_list=[]
target_mac_list=[]
central_mac_list=[]
central_mac_prime_list=[]
target_central_mac_list=[]
target_central_mac_prime_list=[]

central_mixer_list=[]
central_mixer_prime_list=[]
target_central_mixer_list=[]
target_central_mixer_prime_list=[]



last_stored_data_T=0

total_name_list = ["t_env_list", "buffer_list",   "episode_batch_list", "mac_list", "target_mac_list", "central_mac_list", "central_mac_prime_list", "target_central_mac_list", \
                "target_central_mac_prime_list", "central_mixer_list", "central_mixer_prime_list", "target_central_mixer_list", "target_central_mixer_prime_list"]
total_list = [t_env_list, buffer_list, episode_batch_list,\
              mac_list, target_mac_list, central_mac_list, central_mac_prime_list, target_central_mac_list, \
    target_central_mac_prime_list, central_mixer_list, central_mixer_prime_list, target_central_mixer_list, \
              target_central_mixer_prime_list]
learner=None
def run_sequential(args, logger):
    global t_env_list,buffer_list,mac_list,learner_list,episode_batch_list,last_stored_data_T
    global mac_list,target_mac_list,central_mac_list,central_mac_prime_list,target_central_mac_list,\
    target_central_mac_prime_list,central_mixer_list,central_mixer_prime_list,target_central_mixer_list,target_central_mixer_prime_list
    global  learner
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    if(args.name=="self_enhanced_ddpg_continue"):
        args.action_spaces = env_info["action_spaces"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if(args.name=="self_enhanced_ddpg"):
        scheme["actions_prob"]= {"vshape": (1,), "group": "agents", "dtype": th.float}
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    print("------------------------episode_limit:", env_info["episode_limit"] + 1)
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_test_T_print=-args.test_interval - 1

    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            if(args.critic_update_ind):

                critic_update_time = 0

                due_times = 0
                while (due_times < 2 and critic_update_time<10):

                    episode_sample = buffer.sample(args.batch_size)
                    critic_update_time += 1
                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    critic_loss = learner.train_central_qmix(episode_sample, runner.t_env, episode)
                    if (critic_loss > 0.2):
                        # logger.console_logger.info("Updated central qmix network , critic loss:{} t_run {}".format(critic_loss,runner.t_env))
                        due_times = 0

                    else:
                        # logger.console_logger.info("++++ Updated central qmix network , critic loss:{}  t_run {}".format(critic_loss,runner.t_env))
                        due_times += 1
                logger.log_stat("critic_update_time", critic_update_time, runner.t_env)


                #
                # both_convergence=0
                # critic_update_time = 0
                # while(both_convergence!=2):
                #
                #     due_times=0
                #     while(due_times<2):
                #
                #         episode_sample = buffer.sample(args.batch_size)
                #         critic_update_time+=1
                #         # Truncate batch to only filled timesteps
                #         max_ep_t = episode_sample.max_t_filled()
                #         episode_sample = episode_sample[:, :max_ep_t]
                #
                #         if episode_sample.device != args.device:
                #             episode_sample.to(args.device)
                #         critic_loss=learner.train_central_qmix(episode_sample, runner.t_env, episode)
                #         if(critic_loss>0.1):
                #             #logger.console_logger.info("Updated central qmix network , critic loss:{} t_run {}".format(critic_loss,runner.t_env))
                #             due_times = 0
                #             both_convergence=0
                #         else:
                #             #logger.console_logger.info("++++ Updated central qmix network , critic loss:{}  t_run {}".format(critic_loss,runner.t_env))
                #             due_times+=1
                #             if(both_convergence==1 and due_times==2):
                #                 both_convergence=2
                #     if(both_convergence==2):break;
                #     logger.console_logger.info("t_env : {}".format(runner.t_env))
                #     learner._update_central_targets()
                #     both_convergence=1
                # logger.log_stat("critic_update_time",critic_update_time, runner.t_env)
                #logger.console_logger.info("Updated central qmix network finished,critic loss:{} t_env: {} -----".format(critic_loss, runner.t_env))
                if buffer.can_sample(args.batch_size_for_qmix):
                    for _ in range(args.training_iters):
                        critic_loss=1.0
                        episode_sample = buffer.sample(args.batch_size_for_qmix)

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)
                        while (critic_loss > 0.2):
                            critic_loss = learner.train_central_qmix(episode_sample, runner.t_env, episode)
                        learner.train_qmix(episode_sample, runner.t_env, episode)

            else:
                for _ in range(args.training_iters):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)




                    if(runner.t_env%1000==0):
                        if (args.q_table):
                            episode_sample_for_q_table = buffer.sample(1)
                            print(" runner.t_env:", runner.t_env)
                            learner.print_q_table(episode_sample_for_q_table,runner.t_env)

        t_env_list.append(runner.t_env)
        buffer_list.append(buffer)
        episode_batch_list.append(episode_batch)
        # --
        for i in range(3, len(total_name_list)):
            # if (total_name_list[i] == "mac_list"):
            #     tmp = getattr(learner, total_name_list[i][0:-5], None)
            #     print(tmp)
            tmp = getattr(learner, total_name_list[i][0:-5], None)
            total_list[i].append(tmp)
        if (runner.t_env - last_stored_data_T) / 10000 >= 1.0:
            import pickle


            for i in range(len(total_name_list)):
                #'/home/cwb/tmp/meta_qmix/src/results/tb_logs/self_enhanced_ddpg__matrix_game_with_huge_action_space__2022-12-19_01-34-35'
                # if (total_name_list[i] == "mac_list"):
                #     tmp = getattr(learner, total_name_list[i][0:-5], None)
                #     print(mac_list)
                unique_dir=os.path.split(logger.directory_name)[1]
                os.makedirs("../results/offline_data/{}/{}".format(unique_dir,total_name_list[i]), exist_ok=True)
                output_hal = open("../results/offline_data/{}/{}/{}_{}_{}.pkl".format(unique_dir, total_name_list[i],total_name_list[i],
                                                                                   last_stored_data_T, runner.t_env),
                                  'wb')
                #print(total_name_list[i])

                output_hal_s = pickle.dumps(total_list[i])
                output_hal.write(output_hal_s)
                output_hal.close()
                # rq = class()
                # with open("1.pkl",'rb') as file:
                #     rq  = pickle.loads(file.read())
                total_list[i].clear()

            last_stored_data_T = runner.t_env
        if (runner.t_env - last_test_T_print) / (args.test_interval/5) >= 1.0:
            if (args.q_table):
                episode_sample = buffer.sample(1)
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)
                #learner.print_q_table(episode_sample, runner.t_env)
            last_test_T_print=runner.t_env

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                test_episode_batch=runner.run(test_mode=True)

                #buffer.insert_episode_batch(test_episode_batch)


        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(dirname(dirname(abspath(__file__)))+save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


# TODO: Clean this up
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
