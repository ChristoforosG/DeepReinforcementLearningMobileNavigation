import copy
from datetime import datetime
import time


class Agent_():
    def __init__(self,params,env,eval):
        self.algorithm = params.algorithm
        self.action_space = params.action_space
        self.state_space = params.state_space
        self.state = self.get_state_space()
        self.input ,self.input_params = self.get_NN_input(params.input)
        self.input_params.pop(0)
        self.input_params.pop(len(self.input_params) - 1)
        for s in self.input_params:
            env.obstacles[s] = env.dictionary_obst_values[s]
        self.output = params.output
        self.actor_learning_rate = float(params.actor_learning_rate)
        self.critic_learning_rate = float(params.critic_learning_rate)
        self.epsilon = float(params.epsilon)
        self.epsilon_decay = float(params.epsilon_decay)
        self.epsilon_min = float(params.epsilon_min)
        self.gamma = float(params.gamma)
        self.batch_size = int(params.batch_size)
        self.neural_network_actor = params.neural_network_actor
        self.neural_network_critic = params.neural_network_critic
        if not eval:
            str_now_ = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            if self.algorithm == 'DQN':
                directory_name = 'DQN'
            elif self.algorithm == 'Policy Gradient':
                directory_name = 'Policy_Gradient'
            elif self.algorithm == 'Actor Critic':
                directory_name = 'Actor_Critic'
            path_name_ = "../Models/" + directory_name + '/' + str_now_
            self.configuration_name = path_name_
            file_input = open("my_env.ini", "r")
            self.configuration_contents = file_input.read()
            file_input.close()
            file_output = open(self.configuration_name + ".txt", "w+")
            file_output.write(self.configuration_contents)
            file_output.close()

    def get_NN_input(self,input):
        string = input.split()
        NN_input = []
        for i in range(0, len(string)):
            temp = 'R' + str(i)
            if string[i] == temp:
                for j in range(0,2):
                    NN_input.append(copy.copy(self.state[j]))
            elif string[i] == 'Sensor':
                print("-- FROM: agent.py , function get_NN_input -- ")
                print("ERROR: <<agent-input>> variable has not yet implement Sensor")
                exit(0)
            elif string[i] == 'Goal':
                NN_input.append(0.0)
                NN_input.append(0.0)
        if len(self.state) == 3:
            NN_input.append(0.0)
        return NN_input, string

    def get_state_space(self):
        #TO DO, write this function correctly.
        string = self.state_space
        words = string.split()
        state = []
        if len(words) == 1:
            state = [0.0, 0.0]
        elif len(words) == 2:
            state = [0.0, 0.0, 0.0]
        elif len(words) == 3:
            print("-- FROM: aget.py , function geet_state_space -- ")
            print("ERROR: Sensor and Multi Agent is not implemented yet")
            exit(0)
        else:
            print("-- FROM: aget.py , function geet_state_space -- ")
            print("ERROR: Sensor and Multi Agent is not implemented yet")
            exit(0)
        return state

    def get_state(self,env):
        current = copy.copy(env.current_state)
        previous = copy.copy(env.previous_state)
        for s in self.input_params:
            current.extend(env.obstacles[s])
            previous.extend(env.obstacles[s])
        current.extend(env.goal)
        previous.extend(env.goal)
        return current, previous

    def choose_action(self,env):
        pass

    def get_action_space(self):
        string = self.action_space
        words = string.split()
        velocity_limits = [[float(words[4]),float(words[6])],[float(words[10]),float(words[12])]]
        discrete_actions = []
        for i in range(14,len(words)):
            if words[i] == '[':
                vel_1 = words[i+1]
            elif words[i] == ']':
                vel_2 = words[i-1]
                action = [float(vel_1),float(vel_2)]
                discrete_actions.append(action)
        return velocity_limits, discrete_actions

