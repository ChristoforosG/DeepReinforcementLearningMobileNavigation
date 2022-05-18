from agent_dir.agent import Agent_
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy
import copy
import random

class history_():
    def __init__(self):
        self.state_array, self.action_array, self.reward_array = [], [], []

class Experiance_():
    def __init__(self):
        self.current_state = []
        self.next_state = []
        self.reward = 0.0
        self.action = [0.0, 0.0]
        self.end = 0
        self.action_index = 0

class actor_critic_(Agent_):
    def __init__(self,params,env, eval):
        super(actor_critic_, self).__init__(params,env, eval)
        self.action_array, self.action_size = self.get_actions()
        self.history = history_()
        self.exp = []
        if not eval:
            self.actor_model = self.build_actor_model()
            self.critic_model = self.build_critic_model()

    def __str__(self):
        print('__________________________________________________________________________')
        print('-Actor Critic Agent with the following parameters: ')
        print('  -General: ')
        print('     Action Size = ', self.action_size)
        print('     Possible Actions = ', self.action_array)
        print('     Gamma Parameter = ', self.gamma)
        print('  -Actor: ')
        print('     Learning Rate = ', self.actor_learning_rate)
        print('     Neural Network = ', self.neural_network_actor)
        print('  -Critic: ')
        print('     Learning Rate = ', self.critic_learning_rate)
        print('     Neural Network = ', self.neural_network_critic)
        return '__________________________________________________________________________'

    def get_actions(self):
        limits, actions = self.get_action_space()
        return actions, len(actions)

    def build_actor_model(self):
        string = self.neural_network_actor
        words = string.split()
        model = Sequential()
        end_for = (len(words) - 2) / 2
        model.add(Dense(int(words[0]), input_shape=(len(self.input),), activation=words[1]))
        for i in range(1, int(end_for)):
            model.add(Dense(int(words[i + 1]), activation=words[i + 2]))
        model.add(Dense(self.action_size, activation=words[int(len(words) - 2)]))
        model.compile(loss=words[int(len(words) - 1)], optimizer=Adam(lr=self.actor_learning_rate))
        return model

    def build_critic_model(self):
        # TO DO, there are two variables for NN input and output that are not used and should be (??)
        string = self.neural_network_critic
        words = string.split()
        model = Sequential()
        end_for = (len(words) - 2) / 2
        model.add(Dense(int(words[0]), input_shape=(len(self.input),), activation=words[1]))
        for i in range(1, int(end_for)):
            model.add(Dense(int(words[i + 1]), activation=words[i + 2]))
        model.add(Dense(1, activation=words[int(len(words) - 2)]))
        model.compile(loss=words[int(len(words) - 1)], optimizer=Adam(lr=self.critic_learning_rate))
        return model

    def choose_action(self, env):
        self.state, ignore = self.get_state(env)
        np_state = numpy.array([self.state])
        policy = self.actor_model.predict(np_state, batch_size = 1).flatten()
        if env.end_check != 0:
            print('===========')
            print(policy)
            print('===========')
        if numpy.isnan(policy[0]):
            isopithana = 1.0/float(self.action_size)
            policy = [isopithana]*self.action_size
        action_index = numpy.random.choice(self.action_size, 1, p=policy)[0]
        return action_index

    def reset_trajectory(self):
        self.history.state_array, self.history.action_array, self.history.reward_array = [], [], []
        return

    def append_sample(self, state, action, reward):
        self.history.state_array.append(state)
        self.history.reward_array.append(reward)
        self.history.action_array.append(action)
        return


    def train_model(self):
        for i in range(0,len(self.history.state_array)-1):
            target = numpy.zeros((1,1))
            advantages = numpy.zeros((1, self.action_size))
            next_state, state  = self.history.state_array[i+1], self.history.state_array[i]
            np_next_state = numpy.array([next_state])
            np_state = numpy.array([state])
            value = self.critic_model.predict(np_state)
            next_value = self.critic_model.predict(np_next_state)
            advantages[0][self.history.action_array[i]] = self.history.reward_array[i] + self.gamma * (next_value) - value
            target[0][0] = self.history.reward_array[i] + self.gamma * next_value
            self.actor_model.fit(np_state, advantages, epochs=1, verbose=0)
            self.critic_model.fit(np_state, target, epochs=1, verbose=0)
        state = self.history.state_array[len(self.history.state_array)-1]
        np_state = numpy.array([state])
        target = numpy.zeros((1, 1))
        advantages = numpy.zeros((1, self.action_size))
        value = self.critic_model.predict(np_state)
        advantages[0][self.history.action_array[len(self.history.state_array)-1]] = self.history.reward_array[len(self.history.state_array)-1] - value
        target[0][0] = self.history.reward_array[len(self.history.state_array)-1]
        self.actor_model.fit(np_state, advantages, epochs=1, verbose=0)
        self.critic_model.fit(np_state, target, epochs=1, verbose=0)
        return

    def train_model_replay(self):
        minibatch = random.sample(self.exp, self.batch_size)
        target = numpy.zeros((self.batch_size, 1))
        advantages = numpy.zeros((self.batch_size, self.action_size))
        state_array = []
        for i in range(0,self.batch_size):
            next_state, state = minibatch[i].next_state, minibatch[i].current_state
            state_array.append(state)
            np_next_state = numpy.array([next_state])
            np_state = numpy.array([state])
            value = self.critic_model.predict(np_state)
            next_value = self.critic_model.predict(np_next_state)
            if minibatch[i].end == 0:
                advantages[i][minibatch[i].action_index] = minibatch[i].reward + self.gamma * (next_value) - value
                target[i][0] = minibatch[i].reward + self.gamma * next_value
            else:
                advantages[i][minibatch[i].action_index] = minibatch[i].reward - value
                target[i][0] =  minibatch[i].reward
        np_state_array = numpy.array(state_array)
        self.actor_model.fit(np_state_array, advantages, epochs=1, verbose=0)
        self.critic_model.fit(np_state_array, target, epochs=1, verbose=0)
        return

    def train_model_batch(self):
        target = numpy.zeros((len(self.history.state_array), 1))
        advantages = numpy.zeros((len(self.history.state_array), self.action_size))
        for i in range(0,len(self.history.state_array)-1):
            next_state, state  = self.history.state_array[i+1], self.history.state_array[i]
            np_next_state = numpy.array([next_state])
            np_state = numpy.array([state])
            value = self.critic_model.predict(np_state)
            next_value = self.critic_model.predict(np_next_state)
            advantages[i][self.history.action_array[i]] = self.history.reward_array[i] + self.gamma * (next_value) - value
            target[i][0] = self.history.reward_array[i] + self.gamma * next_value
        state = self.history.state_array[len(self.history.state_array)-1]
        np_state = numpy.array([state])
        value = self.critic_model.predict(np_state)
        advantages[len(self.history.state_array)-1][self.history.action_array[len(self.history.state_array)-1]] = self.history.reward_array[len(self.history.state_array)-1] - value
        target[len(self.history.state_array)-1][0] = self.history.reward_array[len(self.history.state_array)-1]
        state_array = numpy.array(self.history.state_array)
        self.actor_model.fit(state_array, advantages, epochs=1, verbose=0)
        self.critic_model.fit(state_array, target, epochs=1, verbose=0)
        return

    def save_to_buffer(self,env,action_index):
        exper = Experiance_()
        exper.next_state, exper.current_state = self.get_state(env)
        exper.reward = copy.copy(env.reward)
        exper.end = copy.copy(env.end_check)
        exper.action_index = copy.copy(action_index)
        exper.action = copy.copy(self.action_array[action_index])
        self.exp.append(exper)
        if len(self.exp) > 10000:
            self.exp.pop(0)

    def update_network(self):
        if len(self.exp) < 5000:
            return
        self.train_model_replay()
        return