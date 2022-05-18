import copy
import math
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np

class Surveillance_():

    def __init__(self, params):
        self.value = 0.0
        self.episode = 0
        self.success_matrix = []
        self.values_matrix = []
        self.values_graph = []
        self.success_rate = 0.0
        self.step_count = 0
        self.success_condition = params.success_rate
        self.number_of_latest_episodes = params.number_of_latest_episodes
        return

    def statistics_per_episode(self,env, agent):
        self.value += env.reward
        self.step_count += 1
        if env.end_check != 0:
            self.episode += 1
            if env.end_check == 2:
                self.success_matrix.append(1)
            else:
                self.success_matrix.append(0)

            if len(self.success_matrix) > self.number_of_latest_episodes:
                self.success_matrix.pop(0)
                self.values_matrix.pop(0)
            self.values_matrix.append(self.value)
            self.values_graph.append(copy.copy(self.value))
            self.value = 0.0
            print('------------------------------------------------------------------------------------------')
            print('Episode: '),
            print(self.episode)
            print('Starting Position: '),
            print(env.old_start)
            print('Goal: '),
            print(env.old_goal)
            print('Time steps: '),
            print(env.time_steps)
            print('Epsilon Current Value: '),
            print(agent.epsilon)
            print('Values: '),
            print(self.values_matrix)
            print('Success Matrix: '),
            print(self.success_matrix)
            self.success_rate_calc()
            print('Success Rate: '),
            print(self.success_rate)
            print('------------------------------------------------------------------------------------------')
            return

    def success_rate_calc(self):
        self.success_rate = 0.0
        for i in range(0,len(self.success_matrix)):
            if self.success_matrix[i] == 1:
                self.success_rate += 1
        if len(self.success_matrix) > 0:
            self.success_rate = float(self.success_rate)/float(len(self.success_matrix))
        return

    def save_model(self,agent):
        if agent.algorithm == 'DQN':
            if (self.success_rate > self.success_condition) & (agent.epsilon <= agent.epsilon_min):
                agent.model.save(agent.configuration_name + '.h5')
                file_ = open(agent.configuration_name + "values" + ".txt", "w+")
                for s in self.values_graph:
                    file_.write(str(s) + "\n")
                file_.close()
                print("TRAINED")
                exit(0)
        elif agent.algorithm == 'Policy Gradient':
            if (self.success_rate > self.success_condition):
                agent.model.save(agent.configuration_name + '.h5')
                file_ = open(agent.configuration_name + "values" + ".txt", "w+")
                for s in self.values_graph:
                    file_.write(str(s) + "\n")
                file_.close()
                print("TRAINED")
                exit(0)
        elif agent.algorithm == 'Actor Critic':
            if (self.success_rate > self.success_condition):
                agent.actor_model.save(agent.configuration_name + '_actor' + '.h5')
                agent.critic_model.save(agent.configuration_name + '_critic' + '.h5')
                file_ = open(agent.configuration_name + "values" + ".txt", "w+")
                for s in self.values_graph:
                    file_.write(str(s) + "\n")
                file_.close()
                print("TRAINED")
                exit(0)
        return

    def load_model(self):
        #path_name_ = "./Models/DQN/29_09_2019_18_44_13.h5"
        #path_name_ = "./Models/Policy_Gradient/21_10_2019_03_24_49.h5"
        path_name_ = "./Models/Actor_Critic/21_10_2019_19_44_22_actor.h5"
        model = load_model(path_name_)
        return model

    def print_state_per_step(self,env,agent,action):
        temp = copy.copy(env.current_state)
        temp[2] = (temp[2]*180.0)/math.pi
        print('State of robot: '),
        print(temp)
        print('Goal: '),
        print(env.goal)
        print('Reward: ')
        print(env.reward)
        dx = env.goal[0]-env.current_state[0]
        dy = env.goal[1]-env.current_state[1]
        dr = [dx, dy]
        print('Relative position: '),
        print(dr)
        th_desired = math.atan2(dy, dx)
        th_desired = (th_desired*180.0)/math.pi
        print('Desired theta: '),
        print(th_desired)
        dr_reward = abs(dx) / env.grid[0] + abs(dy) / env.grid[1]
        d_th = abs(env.reward)   - dr_reward
        d_th = (d_th*180.0)/math.pi
        print('d_th: '),
        print(d_th)
        print('##############')
        return

    def print_step(self,env,agent,action):
        print('##########################################')
        print('Previous State: '),
        print(env.previous_state)
        print('Action: '),
        print(action)
        print('Current State: '),
        print(env.current_state)
        return


    def create_trajectory(self,X,Y,env):
        plt.figure(figsize=(7, 8))
        plt.plot(env.old_goal[0], env.old_goal[1], marker="x",color='red')
        circle1 = plt.Circle((X[len(X)-1], Y[len(Y)-1]),0.3,fill=False)
        circle2 = plt.Circle((env.old_start[0], env.old_start[1]), 0.3, fill=False)
        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.plot(X, Y)
        plt.gcf().gca().set_xlim([0.0,env.grid[0]])
        plt.gcf().gca().set_ylim([0.0, env.grid[1]])
        plt .show()
        return

    def create_reward_map(self,env,agent):
        x = np.linspace(7, 0, 41)
        y = np.linspace(0, 8, 36)
        z = []
        for i in x:
            for j in y:
                state = [i, j, env.goal[0], env.goal[1]]
                np_state = np.array([state])
                temp = agent.model.predict(np_state)
                val = np.amax(temp)
                z.append(val)
        z = np.array(z)
        Z = z.reshape(len(x), len(y))
        max_val = np.amax(Z)
        index = np.argmax(Z)
        min_val = np.amin(Z)
        print(max_val)
        print(Z[15, 25])
        print(index)
        print(min_val)
        plt.figure(figsize=(7, 8))
        plt.gcf().gca().set_xlim([0.0,env.grid[0]])
        plt.gcf().gca().set_ylim([0.0, env.grid[1]])
        plt.imshow(Z, interpolation='bilinear')
        plt.show()
        return

    """
    def show_trajectory(self, env, agent):
        if env.end_check != 0:
            X = []
            Y = []
            print(len(agent.history.state_array))
            for i in range(0, len(agent.history.state_array)):
                state = agent.history.state_array[i]
                X.append(state[0])
                Y.append(state[1])
            print(X)
            print(agent.history.state_array)
            plt.plot(X, Y)
            plt.show()
            exit(0)
        return
    """