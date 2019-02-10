#this is blackbox challenge test

import libs_env.blackbox.env_black_box as blackbox
import libs_agent.agent_dqn
import numpy


class BlackBoxTrial:

    def __init__(self, verbose = False):

        self.verbose = verbose
        self.env = blackbox.EnvBlackBox()

        #print environment info
        if (verbose):
            self.env.print_info()

        #init DQN agent
        self.agent = libs_agent.agent_dqn.DQNAgent(self.env, "networks/black_box_network/parameters.json", 0.4, 0.05, 0.99999)

        self.training_iterations = 100000
        self.testing_iterations = 10000

    #process training
    def train(self):

        for iteration in range(0, self.training_iterations):
            self.agent.main()
            #print training progress %, ane score, every 100th iterations

        if self.verbose:
            if iteration%100 == 0:
                print(iteration*100.0/self.training_iterations, env.get_score())
                self.env._print()

            self.env.show()

    #process testing run
    def test(self):

        #reset score
        self.env.reset_score()
        self.env.reset()

        #choose only the best action
        self.agent.run_best_enable()

        #process testing iterations
        for iteration in range(0, self.testing_iterations):
            self.agent.main()
            if (self.verbose):
                print("move=", self.env.get_move(), " score=", self.env.get_score())

    def get_score(self):
        return self.env.get_score()

def main():
    trials_count = 32
    print("starting ", trials_count, " trials")

    trials_results = numpy.zeros(trials_count)

    for i in range(0, trials_count):
        trial = BlackBoxTrial()
        trial.train()
        trial.test()
        trials_results[i] = trial.get_score()
        print(i, trial.get_score())

    average_score = numpy.average(trials_results)
    std_score = numpy.std(trials_results)
    max_score = numpy.max(trials_results)
    min_score = numpy.min(trials_results)

    print()
    print("average score = ", average_score)
    print("std score = ", std_score)
    print("max score = ", max_score)
    print("min score = ", min_score)


    print("program done")



if __name__== "__main__":
    main()
