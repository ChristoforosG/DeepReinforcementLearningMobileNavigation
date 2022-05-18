import numpy
import copy
import os
import math


class Environment_():
    def __init__(self,params):
        self.grid = [0.0, 0.0]
        self.create_map(params.map_flag)
        self.end_flag = int(params.end_flag)
        self.start_flag = int(params.start_flag)
        self.create_state_configuration(params.state)
        self.starting_point = copy.copy(params.starting_point)
        self.create_simulation_environment(params.simulation_mode)
        self.end_check = 0
        self.collision_reward = float(params.collision_reward)
        self.goal_reward = float(params.goal_reward)
        self.reward = 0.0
        self.obstacles = {}
        self.dictionary_obst_values = {"R1": [4.0, 4.0], "R2": [3.0, 2.0]}
        self.goal = params.ending_point
        self.time_steps = 0
        self.time_div = float(params.time_div)
        self.episode_steps = float(params.episode_steps)
        self.create_reward_func(params.reward_type)
        self._init_state()
        self.reset_()

    def create_map(self,flag):
        """
        Empty map
        """
        self.grid = [7.0, 8.0]
        pass

    def create_state_configuration(self,state):
        state = int(state)
        if state == 1:
            self._init_state = self._init_state_1
            self.update_state = self.update_state_1
            self.reset_ = self.reset_1_
        elif state == 2:
            self._init_state = self._init_state_2
            self.update_state = self.update_state_2
            self.reset_ = self.reset_2_
        elif state == 3:
            self._init_state = self._init_state_3
            self.update_state = self.update_state_3
            self.reset_ = self.reset_3_
        elif state == 4:
            self._init_state = self._init_state_4
            self.update_state = self.update_state_4
            self.reset_ = self.reset_4_
        else:
            print("-- FROM: environment.py , class Environment_(), function create_state_configuration -- ")
            print("ERROR: Environment-State parameter can only be an integer between 1 and 4. Check 'my_env.ini' for more information.")
            exit(0)
        return

    def create_simulation_environment(self, simulation):
        if simulation == 'ROS':
            # TO DO
            self.step_ = self._step_1
            print('ROS is not implemented yet')
            exit(0)
        elif simulation == 'default':
            self.step_ = self._step_2
            pass
        else:
            print("-- FROM: environment.py , class Environment_(), function create_simulation_environment -- ")
            print("ERROR: <<Environment-run_mode>> parameter you typed does not exist, or is not implemented yet. Check 'my_env.ini' for more information.")
            exit(0)

    def create_reward_func(self,type):
        if type == 'step':
            self.step_reward = self.step_reward_1
        elif type == 'relative position':
            self.step_reward = self.step_reward_3
        elif type == 'relative pose':
            self.step_reward = self.step_reward_2
        elif type == 'angle':
            self.step_reward = self.step_reward_4
        elif type == 'relative distance':
            self.step_reward = self.step_reward_5
        elif type == 'euclidean':
            self.step_reward = self.step_reward_6
        elif type == 'reverse euclidean':
            self.step_reward = self.step_reward_7

    def step_reward_1(self):
        return -1.0

    def step_reward_2(self):
        dx = float(self.goal[0]) - self.current_state[0]
        dy = float(self.goal[1]) - self.current_state[1]
        temp = [math.cos(self.current_state[2]), math.sin(self.current_state[2])]
        arithmitis = dx*temp[0]+dy*temp[1]
        paranomastis = math.sqrt(dx * dx + dy * dy) * math.sqrt(temp[0] * temp[0] + temp[1] * temp[1])
        cosine_th = arithmitis / paranomastis
        d_th = math.acos(cosine_th)
        reward = (abs(dx) / self.grid[0]) + (abs(dy) / self.grid[1]) + (d_th/(2*math.pi))
        return -reward

    def step_reward_3(self):
        dx = float(self.goal[0]) - self.current_state[0]
        dy = float(self.goal[1]) - self.current_state[1]
        reward = abs(dx)/self.grid[0] + abs(dy)/self.grid[1]
        return -reward

    def step_reward_4(self):
        dx = float(self.goal[0]) - self.current_state[0]
        dy = float(self.goal[1]) - self.current_state[1]
        temp = [math.cos(self.current_state[2]), math.sin(self.current_state[2])]
        arithmitis = dx * temp[0] + dy * temp[1]
        paranomastis = math.sqrt(dx * dx + dy * dy) * math.sqrt(temp[0] * temp[0] + temp[1] * temp[1])
        cosine_th = arithmitis / paranomastis
        d_th = math.acos(cosine_th)
        reward = d_th / (2 * math.pi)
        return -reward

    def step_reward_5(self):
        dx = float(self.goal[0]) - self.current_state[0]
        dy = float(self.goal[1]) - self.current_state[1]
        dx_prev = float(self.goal[0]) - self.previous_state[0]
        dy_prev = float(self.goal[1]) - self.previous_state[1]
        #temp = [math.cos(self.current_state[2]), math.sin(self.current_state[2])]
        #arithmitis = dx*temp[0]+dy*temp[1]
        #paranomastis = math.sqrt(dx * dx + dy * dy) * math.sqrt(temp[0] * temp[0] + temp[1] * temp[1])
        #cosine_th = arithmitis / paranomastis
        #d_th = math.acos(cosine_th)
        dr = math.sqrt(dx*dx + dy*dy) - math.sqrt(dx_prev*dx_prev + dy_prev*dy_prev)
        reward = dr #+ (d_th/(2*math.pi))
        return -reward

    def step_reward_6(self):
        dx = float(self.goal[0]) - self.current_state[0]
        dy = float(self.goal[1]) - self.current_state[1]
        dr = math.sqrt(dx * dx + dy * dy)
        reward = dr
        return -reward

    def step_reward_7(self):
        dx = float(self.goal[0]) - self.current_state[0]
        dy = float(self.goal[1]) - self.current_state[1]
        dr = 1.0/(math.sqrt(dx * dx + dy * dy) - 0.299)
        reward = dr
        return reward

    def _step_1(self, action):
        return

    def _step_2(self, action):
        self.previous_state = copy.copy(self.current_state)
        self.current_state = self.update_state(self.current_state,action)
        self.get_reward()
        self.time_steps += 1
        return

    def update_state_1(self,state,action):
        state[0] += action[0] * self.time_div
        state[0] = copy.copy(float(format(state[0],'.5g')))
        state[1] += action[1] * self.time_div
        state[1] = copy.copy(float(format(state[1], '.5g')))
        return state

    def update_state_2(self, state, action):
        state[0] += action[0]*math.cos(state[2])*self.time_div
        state[1] += action[0]*math.sin(state[2])*self.time_div
        state[2] += action[1]*self.time_div
        if state[2] > 2*math.pi:
            state[2] -= 2*math.pi
        if state[2] < 0:
            state[2] += 2*math.pi
        state[0] = copy.copy(float(format(state[0],'.5g')))
        state[1] = copy.copy(float(format(state[1],'.5g')))
        state[2] = copy.copy(float(format(state[2],'.5g')))
        return state

    def update_state_3(self,state,action):
        print("3")
        return

    def update_state_4(self,state,action):
        print("4")
        return

    def _init_state_1(self):
        if self.start_flag:
            self.current_state = copy.copy([self.starting_point[0], self.starting_point[1]])
            self.previous_state = copy.copy([self.starting_point[0], self.starting_point[1]])
        else:
            x = numpy.random.rand() * self.grid[0]
            y = numpy.random.rand() * self.grid[1]
            self.starting_point = [x, y]
            self.current_state = copy.copy(self.starting_point)
            self.previous_state = self.current_state
        return

    def _init_state_2(self):
        if self.start_flag:
            self.current_state = copy.copy(self.starting_point)
            self.previous_state = copy.copy(self.starting_point)
        else:
            x = numpy.random.rand()*self.grid[0]
            y = numpy.random.rand()*self.grid[1]
            th = numpy.random.random()*6.28
            self.starting_point = [float(x), float(y), th]
            self.current_state = copy.copy(self.starting_point)
            self.previous_state = copy.copy(self.current_state)
        return

    def _init_state_3(self):
        print("ERROR: state of type 3 is not implemented yet")
        exit(0)
        return

    def _init_state_4(self):
        print("ERROR: state of type 4 is not implemented yet")
        exit(0)
        return

    def get_reward(self):
        crushed = self.check_if_crushed()
        goal = self.check_goal()
        if crushed == 1:
            self.reward = self.collision_reward
            self.end_check = 1
        elif goal == 1:
            self.reward = self.goal_reward
            self.end_check = 2
        elif self.time_steps >= self.episode_steps:
            self.reward = self.step_reward()
            self.end_check = 3
        else:
            self.reward = self.step_reward()
            self.end_check = 0

    def check_if_crushed(self):
        for s in self.obstacles:
            obstacle = self.obstacles[s]
            dx = abs(self.current_state[0] - float(obstacle[0]))
            dy = abs(self.current_state[1] - float(obstacle[1]))
            if dx < 0.6 and dy < 0.6:
                return 1
        return 0

    def check_goal(self):
        dx = abs(self.current_state[0] - float(self.goal[0]))
        dy = abs(self.current_state[1] - float(self.goal[1]))
        if dx < 0.3 and dy < 0.3:
            return 1
        else:
            return 0

    def check_if_done(self):
        if self.end_check == 1:
            print('TRAKARES MALAKA')
            self.reset_()
        elif self.end_check == 2:
            print('KATI KANEIS')
            self.reset_()
        elif self.end_check == 3:
            print('ARGEIS VLAKA')
            self.reset_()

    def reset_1_(self):
        self.old_start = copy.copy(self.starting_point)
        self.old_goal = copy.copy(self.goal)
        self._init_state()
        if self.end_flag == 0:
            self.goal[0] = numpy.random.rand()*self.grid[0]
            self.goal[1] = numpy.random.rand()*self.grid[1]

        self.goal[0] = float(self.goal[0])
        self.goal[1] = float(self.goal[1])
        self.time_steps = 0

    def reset_2_(self):
        self.old_start = copy.copy(self.starting_point)
        self.old_goal = copy.copy(self.goal)
        self._init_state()
        if self.end_flag == 0:
            self.goal[0] = numpy.random.rand()*self.grid[0]
            self.goal[1] = numpy.random.rand()*self.grid[1]
        self.goal[0] = float(self.goal[0])
        self.goal[1] = float(self.goal[1])
        self.time_steps = 0


    def reset_3_(self):
        exit(0)

    def reset_4_(self):
        exit(0)

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

    environment_params = Environment_params_()
    environment = Environment_(environment_params)
    for i in range(1,50):
       environment.step_([1.0, 0.0])
       environment.get_reward()
       print(environment.current_state)
       print(environment.reward)
       print('------------------------------')



