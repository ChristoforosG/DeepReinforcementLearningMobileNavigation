from agent_dir.agent import Agent_
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy
import copy

class history_():
    def __init__(self):
        self.state_array, self.action_array, self.reward_array = [], [], []

class policy_gradient_(Agent_):
    def __init__(self, params, env, eval):
        super(policy_gradient_, self).__init__(params, env, eval)
        self.action_array, self.action_size = self.get_actions()
        self.episodes_per_update = int(params.episodes_per_update)
        self.train_counter = 0
        self.history_array = []
        self.history = history_()
        self.reset_trajectory()
        self.state_size = len(env.current_state)
        self.learning_rate = self.actor_learning_rate
        if not eval:
            self.model = self.build_model()

    def get_actions(self):
        limits, actions = self.get_action_space()
        return actions, len(actions)

    def build_model(self):
        string = self.neural_network_actor
        words = string.split()
        model = Sequential()
        end_for = (len(words) - 2) / 2
        model.add(Dense(int(words[0]), input_shape=(len(self.input),), activation=words[1]))
        for i in range(1, int(end_for)):
            model.add(Dense(int(words[i + 1]), activation=words[i + 2]))
        model.add(Dense(self.action_size, activation=words[int(len(words) - 2)]))
        model.compile(loss=words[int(len(words) - 1)], optimizer=Adam(lr=self.learning_rate))
        return model

    def append_sample(self,state, action, reward):
        self.history.state_array.append(state)
        self.history.reward_array.append(reward)
        self.history.action_array.append(action)
        return

    def reset_trajectory(self):
        self.history.state_array, self.history.action_array, self.history.reward_array = [], [], []
        return

    def choose_action(self, env):
        self.state, ignore = self.get_state(env)
        np_state = numpy.array([self.state])
        policy = self.model.predict(np_state, batch_size = 1).flatten()
        #print('++++++++++++++++++++++++++++++++')
        #print(policy)
        if numpy.isnan(policy[0]):
            isopithana = 1.0/float(self.action_size)
            policy = [isopithana]*self.action_size
        action_index = numpy.random.choice(self.action_size, 1, p=policy)[0]
        #print(action_index)
        return action_index

    def choose_action_eval(self, env):
        self.state, ignore = self.get_state(env)
        np_state = numpy.array([self.state])
        policy = self.model.predict(np_state, batch_size = 1).flatten()
        #print('++++++++++++++++++++++++++++++++')
        #print(policy)
        if numpy.isnan(policy[0]):
            isopithana = 1.0/float(self.action_size)
            policy = [isopithana]*self.action_size
        action_index = numpy.argmax(policy)
        #print(policy)
        #print(action_index)
        return action_index

    def update_network(self,env):
        if env.end_check != 0:
            self.train_counter += 1
            self.history_array.append(copy.copy(self.history))
            self.reset_trajectory()
        if self.train_counter >= self.episodes_per_update:
            self.train_counter = 0
            self.train_model(env)
            self.history_array = []

    def train_model(self,env):
        cur, prev = self.get_state(env)
        state_size = len(cur)
        for i in range(0, len(self.history_array)-1):
            episode_length = len(self.history_array[i].state_array)
            discounted_rewards = numpy.zeros_like(self.history_array[i].reward_array)
            running_add = 0
            for t in reversed(range(0, len(self.history_array[i].reward_array))):
                running_add = running_add * self.gamma + self.history_array[i].reward_array[t]
                discounted_rewards[t] = running_add
            discounted_rewards -= numpy.mean(discounted_rewards)
            discounted_rewards /= numpy.std(discounted_rewards)
            update_inputs = numpy.zeros((episode_length, state_size))
            advantages = numpy.zeros((episode_length, self.action_size))
            for j in range(episode_length):
                temp = copy.copy(self.history_array[i].state_array[j])
                update_inputs[j] = copy.copy(temp)
                update_inputs[j] = copy.copy(temp)
                advantages[j][self.history_array[i].action_array[j]] = discounted_rewards[j]
            self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
            return