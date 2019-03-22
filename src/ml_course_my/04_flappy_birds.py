import sys
sys.path.append("..") # Adds higher directory to python modules path.

import libs.libs_env.env_birds
import agent
import agent_dqn

env = libs.libs_env.env_birds.EnvBirds()

env.print_info()

#agent = agent.Agent(env)
agent = agent_dqn.DQNAgent(env, "flappy_bird_net.json")


training_iterations = 100000

for i in range(0, training_iterations):
    agent.main()

    if (i%100) == 0:
        progress = 100.0*i/training_iterations
        print("training done = ", progress, " score = ", env.get_score())


env.reset_score()
agent.run_best_enable()

testing_iterations = 100000

for i in range(0, testing_iterations):
    agent.main()

    env.render()

    if (i%1000) == 0:
        score = env.get_score()
        print("iterations = ", i, "score = ", score)
