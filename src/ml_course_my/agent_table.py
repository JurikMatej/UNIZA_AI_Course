import agent
import numpy

class QLearningAgentTable(agent.Agent):

    def __init__(self, env, gamma = 0.9, epsilon_training = 0.1, epsilon_testing = 0.1):

        agent.Agent.__init__(self, env)

        self.gamma                  = gamma
        self.epsilon_training       = epsilon_training
        self.epsilon_testing        = epsilon_testing

        self.state              = 0
        self.state_prev         = self.state

        self.action              = 0
        self.action_prev         = self.action


        self.actions_count       = self.env.get_actions_count()
        self.states_count        = self.env.get_size()

        self.q_table             = numpy.zeros((self.states_count, self.actions_count))

    def main(self):

        if self.is_run_best_enabled():
            epsilon = self.epsilon_testing
        else:
            epsilon = self.epsilon_training

        self.state_prev  = self.state
        self.state       = self.env.get_observation().argmax()

        self.action_prev    = self.action
        self.action         = self.select_action(self.q_table[self.state], epsilon)


        reward = self.env.get_reward()

        max_q = self.q_table[self.state].max()

        alpha = 0.1

        tmp = (1.0 - alpha)*self.q_table[self.state_prev][self.action_prev]

        self.q_table[self.state_prev][self.action_prev] = tmp + alpha*(reward + self.gamma*max_q)

        self.env.do_action(self.action)

    def print_q_table(self):
        print(self.q_table)


class SarsaAgentTable(agent.Agent):

    def __init__(self, env, gamma = 0.9, epsilon_training = 0.2, epsilon_testing = 0.02):

        agent.Agent.__init__(self, env)

        self.gamma                  = gamma
        self.epsilon_training       = epsilon_training
        self.epsilon_testing        = epsilon_testing

        self.state              = 0
        self.state_prev         = self.state

        self.action              = 0
        self.action_prev         = self.action


        self.actions_count       = self.env.get_actions_count()
        self.states_count        = self.env.get_size()

        self.q_table             = numpy.zeros((self.states_count, self.actions_count))

    def main(self):

        if self.is_run_best_enabled():
            epsilon = self.epsilon_testing
        else:
            epsilon = self.epsilon_training

        self.state_prev  = self.state
        self.state       = self.env.get_observation().argmax()

        self.action_prev    = self.action
        self.action         = self.select_action(self.q_table[self.state], epsilon)


        reward = self.env.get_reward()

        alpha = 0.1

        tmp = (1.0 - alpha)*self.q_table[self.state_prev][self.action_prev]

        self.q_table[self.state_prev][self.action_prev] = tmp + alpha*(reward + self.gamma*self.q_table[self.state][self.action])

        self.env.do_action(self.action)

    def print_q_table(self):
        print(self.q_table)
