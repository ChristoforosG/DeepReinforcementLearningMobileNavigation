import numpy
import copy
import os
import math
from environment import Environment_
from agent_dir.DQN import DQN_
from agent_dir.Policy_Gradient import policy_gradient_
from agent_dir.Actor_Critic import actor_critic_

if __name__ == '__main__':
    class Environment_params_():
        def __init__(self):
            self.time_div = 0.1
            self.num_of_robots = 1
            self.map_flag = 0
            self.end_flag = 1
            self.start_flag = 1
            # state: 1 = "position", 2 = "position + pose", 3 = "position + visual sensor input", 4 = "position + pose + visual sensor input"
            self.state = 2
            # reward type for each step that no event significant happens.'step' = -1. 'relative position' = -( | dx | + | dy | ) / max( | dx | + | dy | ).'relative pose'
            self.reward_type = 'new idea'
            self.collision_reward = -400.0
            self.goal_reward = 1000.0
            self.run_mode = 'training'
            self.simulation_mode = 'default'
            self.ending_point = [4.0, 4.0]
            self.starting_point = [1.0, 1.0, 0.0]
            self.episode_steps = 200


    class Agent_params_():
        def __init__(self):
            self.algorithm = 'DQN'
            self.action_space = 'action_type= default velocity_x= [ -1.0 , 1.0 ] velocity_y_or_z= [ -1.0 , 1.0 ] discrete_actions= [ 1.0 , 0.0 ] - [ 0.0 , -0.8 ] - [ 0.0 , 0.8 ]'
            self.state_space = 'position pose'
            self.input = 'R0 Goal'
            self.output = []
            self.actor_learning_rate = '0.001'
            self.critic_learning_rate = '0.001'
            self.epsilon = '1.0'
            self.epsilon_decay = '0.999'
            self.epsilon_min = '0.05'
            self.gamma = '0.95'
            self.neural_network_actor = '48 relu 48 relu softmax categorical_crossentropy'
            self.neural_network_critic = '12 relu 12 relu linear mse'
            self.batch_size = '200'
            self.episodes_per_update = '3'

    environment_params = Environment_params_()
    environment = Environment_(environment_params)

    agent_params = Agent_params_()
    agent = actor_critic_(agent_params, environment, 1)
    print(agent)