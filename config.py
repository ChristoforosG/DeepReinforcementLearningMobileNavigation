import logging
import configparser

class Config:
    """
    Configuration for train/test/solve
    """
    log = logging.getLogger("Config")

    def __init__(self, file_name):
        self.data = configparser.ConfigParser()
        self.log.info("Reading config file %s", file_name)
        if not self.data.read(file_name):
            raise ValueError("Config file %s not found" % file_name)

    # sections acessors
    @property
    def sect_environment(self):
        return self.data['environment']

    @property
    def sect_agent(self):
        return self.data['agent']

    @property
    def sect_surveillance(self):
        return self.data['surveillance']

    # environment section
    @property
    def time_div(self):
        return self.sect_environment['time_div']

    @property
    def num_of_robots(self):
        return self.sect_environment['num_of_robots']

    @property
    def map_flag(self):
        return self.sect_environment['map_flag']

    @property
    def end_flag(self):
        return self.sect_environment['end_flag']

    @property
    def start_flag(self):
        return self.sect_environment['start_flag']

    @property
    def state(self):
        return self.sect_environment['state']

    @property
    def reward_type(self):
        return self.sect_environment['reward_type']

    @property
    def collision_reward(self):
        return self.sect_environment['collision_reward']

    @property
    def goal_reward(self):
        return self.sect_environment['goal_reward']

    @property
    def run_mode(self):
        return self.sect_environment['run_mode']

    @property
    def simulation_mode(self):
        return self.sect_environment['simulation_mode']

    @property
    def starting_point(self):
        return self.sect_environment['starting_point']

    @property
    def ending_point(self):
        return self.sect_environment['ending_point']

    @property
    def episode_steps(self):
        return self.sect_environment['episode_steps']

    # agent section
    @property
    def algorithm(self):
        return self.sect_agent['algorithm']

    @property
    def state_space(self):
        return self.sect_agent['state_space']

    @property
    def action_space(self):
        return self.sect_agent['action_space']

    @property
    def input(self):
        return self.sect_agent['input']

    @property
    def output(self):
        return self.sect_agent['output']

    @property
    def actor_learning_rate(self):
        return self.sect_agent['actor_learning_rate']

    @property
    def critic_learning_rate(self):
        return self.sect_agent['critic_learning_rate']

    @property
    def epsilon(self):
        return self.sect_agent['epsilon']

    @property
    def epsilon_decay(self):
        return self.sect_agent['epsilon_decay']

    @property
    def epsilon_min(self):
        return self.sect_agent['epsilon_min']

    @property
    def gamma(self):
        return self.sect_agent['gamma']

    @property
    def neural_network_actor(self):
        return self.sect_agent['neural_network_actor']

    @property
    def neural_network_critic(self):
        return self.sect_agent['neural_network_critic']

    @property
    def batch_size(self):
        return self.sect_agent['batch_size']

    @property
    def episodes_per_update(self):
        return self.sect_agent['episodes_per_update']
    # surveillance section
    @property
    def success_rate(self):
        return self.sect_surveillance['success_rate']

    @property
    def number_of_latest_episodes(self):
        return self.sect_surveillance['number_of_latest_episodes']