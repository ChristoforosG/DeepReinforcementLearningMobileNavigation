import argparse
import config
from training import training, evaluation
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class Environment_params_():
    def __init__(self,conf):
        self.time_div = float(conf.time_div)
        self.num_of_robots = int(conf.num_of_robots)
        self.map_flag = int(conf.map_flag)
        self.end_flag = int(conf.end_flag)
        self.start_flag = int(conf.start_flag)
        self.state = int(conf.state)
        self.reward_type = conf.reward_type
        self.collision_reward = float(conf.collision_reward)
        self.goal_reward = float(conf.goal_reward)
        self.run_mode = conf.run_mode
        self.simulation_mode = conf.simulation_mode
        self.ending_point = conf.ending_point.split()
        self.starting_point = conf.starting_point.split()
        temp1 = []
        temp2 = [self.ending_point[0], self.ending_point[1]]
        self.ending_point = copy.copy(temp2)
        for i in range(0,len(self.starting_point)):
            temp1.append(float(self.starting_point[i]))
        self.starting_point = temp1
        self.episode_steps = int(conf.episode_steps)

class Agent_params_():
    def __init__(self,conf):
        self.algorithm = conf.algorithm
        self.action_space = conf.action_space
        self.state_space = conf.state_space
        self.input = conf.input
        self.output = conf.output
        self.actor_learning_rate = conf.actor_learning_rate
        self.critic_learning_rate = conf.critic_learning_rate
        self.epsilon = conf.epsilon
        self.epsilon_decay = conf.epsilon_decay
        self.epsilon_min = conf.epsilon_min
        self.gamma = conf.gamma
        self.neural_network_actor = conf.neural_network_actor
        self.neural_network_critic = conf.neural_network_critic
        self.batch_size = conf.batch_size
        self.episodes_per_update = conf.episodes_per_update

class Surveillance_params_():
    def __init__(self,conf):
        self.success_rate = float(conf.success_rate)
        self.number_of_latest_episodes = int(conf.number_of_latest_episodes)

if __name__ == '__main__':
    # Run it from command line
    parser = argparse.ArgumentParser()
    # Write the parameters of ur experiment to a .ini file
    parser.add_argument("-i", "--ini", required=True, help="Ini file to use for this run")
    args = parser.parse_args()
    config = config.Config(args.ini)
    #pass each parameter of config to each specific class
    environment_parameters = Environment_params_(config)
    agent_parameters = Agent_params_(config)
    surveillance_parameters = Surveillance_params_(config)
    if environment_parameters.run_mode == 'training':
        # create environment and start the training of the agent(s)
        training(agent_parameters,  environment_parameters, surveillance_parameters)
    elif environment_parameters.run_mode == 'evaluation':
        # create environment and load neural model for evaluation
        evaluation(agent_parameters, environment_parameters, surveillance_parameters)
    else:
        print("-- FROM: main.py , function main -- ")
        print("ERROR: <<environment-run_mode>> variable must be either training or evaluation. Check spelling in my_env.ini")
        exit(0)

