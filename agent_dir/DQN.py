from agent_dir.agent import Agent_
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy
import copy
import random

class Experiance_():
    def __init__(self):
        self.current_state = []
        self.next_state = []
        self.reward = 0.0
        self.action = [0.0, 0.0]
        self.end_checking = 0
        self.action_index = 0

class DQN_(Agent_):
    def __init__(self,params,env, eval):
        super(DQN_, self).__init__(params,env, eval)
        self.exp = []
        self.actions = self.get_actions()
        self.learning_rate = self.critic_learning_rate
        if not eval:
            self.model = self.build_model()

    def get_actions(self):
        limits, actions = self.get_action_space()
        return actions

    def choose_action(self,env):
        self.state, ignore = self.get_state(env)
        np_state = numpy.array([self.state])
        if numpy.random.rand() <= self.epsilon:
            self.action_index = numpy.random.randint(0, len(self.actions))
        else:
            act_values = self.model.predict(np_state)
            self.action_index = numpy.argmax(act_values[0])
        return self.actions[self.action_index]

    def choose_action_eval(self,env):
        self.state, ignore = self.get_state(env)
        np_state = numpy.array([self.state])
        if numpy.random.rand() <= self.epsilon:
            self.action_index = numpy.random.randint(0, len(self.actions))
        else:
            act_values = self.model.predict(np_state)
            self.action_index = numpy.argmax(act_values[0])
        return act_values, self.action_index

    def build_model(self):
        #TO DO, there are two variables for NN input and output that are not used and should be (??)
        string = self.neural_network_critic
        words = string.split()
        model = Sequential()
        end_for = (len(words)-2)/2
        model.add(Dense(int(words[0]), input_shape=(len(self.input),), activation=words[1]))
        for i in range(1, int(end_for)):
            model.add(Dense(int(words[i+1]), activation=words[i+2]))
        model.add(Dense(len(self.actions), activation=words[int(len(words)-2)]))
        model.compile(loss=words[int(len(words)-1)], optimizer=Adam(lr=self.learning_rate))
        return model

    def save_to_buffer(self,env):
        exper = Experiance_()
        exper.next_state, exper.current_state = self.get_state(env)
        exper.reward = copy.copy(env.reward)
        exper.end = copy.copy(env.end_check)
        exper.action_index = copy.copy(self.action_index)
        exper.action = copy.copy(self.actions[self.action_index])
        self.exp.append(exper)
        if len(self.exp) > 10000:
            self.exp.pop(0)

    def update_network(self):
        if len(self.exp) < 6000:
            return
        self.replay()
        return

    def replay(self):
        minibatch = random.sample(self.exp, self.batch_size)
        """
        for i in range(1,11):
            print('Previous State: '),
            print(minibatch[i].current_state)
            print('Action: '),
            print(minibatch[i].action)
            print('Current State: '),
            print(minibatch[i].next_state)
            print('Reward: '),
            print(minibatch[i].reward)
            print('-----------------------------------------------')
        """
        for i in range(0,self.batch_size):
            state = numpy.array([minibatch[i].current_state])
            next_state = numpy.array([minibatch[i].next_state])
            target = minibatch[i].reward
            if minibatch[i].end == 0:
                target = minibatch[i].reward + self.gamma*numpy.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)
            target_f[0][minibatch[i].action_index] = target

            self.model.fit(state,target_f,epochs = 1,verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

