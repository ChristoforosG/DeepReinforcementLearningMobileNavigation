from environment import Environment_
from agent_dir.DQN import DQN_
from agent_dir.Policy_Gradient import policy_gradient_
from agent_dir.Actor_Critic import actor_critic_
from surveillance import Surveillance_
import time
import copy
import numpy

def training(agent_param, environment_param, surveillance_param):
    environment = Environment_(environment_param)
    surveillance = Surveillance_(surveillance_param)
    if agent_param.algorithm == "DQN":
        agent = DQN_(agent_param,environment, 0)
        while True:
            action = agent.choose_action(environment)
            environment.step_(action)
            agent.save_to_buffer(environment)
            agent.update_network()
            #surveillance.print_step(environment,agent,action)
            environment.check_if_done()
            surveillance.statistics_per_episode(environment, agent)
            surveillance.save_model(agent)

    elif agent_param.algorithm == "Policy Gradient":
        agent = policy_gradient_(agent_param, environment, 0)
        while True:
            action_index = agent.choose_action(environment)
            environment.step_(agent.action_array[action_index])
            current, previous = agent.get_state(environment)
            agent.append_sample(previous, action_index, environment.reward)
            if environment.end_check != 0:
                agent.append_sample(current, action_index, environment.reward)
            environment.check_if_done()
            surveillance.statistics_per_episode(environment, agent)
            agent.update_network(environment)
            surveillance.save_model(agent)

    elif agent_param.algorithm == "Actor Critic":
        agent = actor_critic_(agent_param, environment, 0)
        while True:
            action_index = agent.choose_action(environment)
            environment.step_(agent.action_array[action_index])
            agent.save_to_buffer(environment, action_index)
            if environment.end_check != 0:
                agent.update_network()
            """
            current, previous = agent.get_state(environment)
            agent.append_sample(previous, action_index, environment.reward)
            if environment.end_check != 0:
                agent.append_sample(current, action_index, environment.reward)
                agent.train_model_batch()
            """
            environment.check_if_done()
            surveillance.statistics_per_episode(environment, agent)
            surveillance.save_model(agent)
    else:
        print("-- FROM: agent.py , class Agent_(), function build_model -- ")
        print("ERROR: The model algorithm you typed is either not implemented yet or typed incorrectly")
        exit(0)

def evaluation(agent_param, environment_param, surveillance_param):
    environment = Environment_(environment_param)
    surveillance = Surveillance_(surveillance_param)
    if agent_param.algorithm == "DQN":
        agent = DQN_(agent_param, environment, 1)
        agent.model = surveillance.load_model()
        for i in range(1,200):
            X = []
            Y = []
            ST = []
            X.append(environment.current_state[0])
            Y.append(environment.current_state[1])
            end = 0
            prev_prev_state = [0.2, 0.0]
            prev_state = [0.3, 0.0]
            current_state = [0.0, 0.0]
            agent.epsilon = 0.0
            breakpoint = 0
            while (not end):
                prev_prev_state = copy.copy(prev_state)
                prev_state = copy.copy(environment.previous_state)
                current_state = copy.copy(environment.current_state)
                action_values, action_index = agent.choose_action_eval(environment)
                if prev_prev_state == current_state:
                    action_values[0][action_index] = -1000.0
                    action_index = numpy.argmax(action_values[0])
                ST.append(copy.copy(agent.actions[action_index]))
                environment.step_(agent.actions[action_index])
                X.append(environment.current_state[0])
                Y.append(environment.current_state[1])
                environment.check_if_done()
                end = environment.end_check
                surveillance.statistics_per_episode(environment, agent)
                breakpoint += 1
            #surveillance.create_trajectory(X, Y, environment)
            #surveillance.create_reward_map(environment,agent)
        print('Exiting test')
        exit(0)
    elif agent_param.algorithm == "Policy Gradient":
        agent = policy_gradient_(agent_param, environment, 1)
        agent.model = surveillance.load_model()
        end = 0
        X = []
        Y = []
        ST = []
        X.append(environment.current_state[0])
        Y.append(environment.current_state[1])
        ST.append([environment.current_state[0], environment.current_state[1]])
        while(not end):
            action_index = agent.choose_action(environment)
            environment.step_(agent.action_array[action_index])
            X.append(environment.current_state[0])
            Y.append(environment.current_state[1])
            ST.append([environment.current_state[0], environment.current_state[1]])
            surveillance.statistics_per_episode(environment, agent)
            environment.check_if_done()
            end = environment.end_check
        surveillance.create_trajectory(X,Y,environment)
    elif agent_param.algorithm == "Actor Critic":
        agent = actor_critic_(agent_param, environment, 1)
        agent.actor_model = surveillance.load_model()
        for i in range(1, 200):
            end = 0
            X = []
            Y = []
            ST = []
            X.append(environment.current_state[0])
            Y.append(environment.current_state[1])
            ST.append([environment.current_state[0], environment.current_state[1]])
            while (not end):
                action_index = agent.choose_action(environment)
                environment.step_(agent.action_array[action_index])
                X.append(environment.current_state[0])
                Y.append(environment.current_state[1])
                ST.append([environment.current_state[0], environment.current_state[1]])
                surveillance.statistics_per_episode(environment, agent)
                environment.check_if_done()
                end = environment.end_check
            surveillance.create_trajectory(X, Y, environment)