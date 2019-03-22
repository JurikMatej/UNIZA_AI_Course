import sys
sys.path.append("..") # Adds higher directory to python modules path.

import libs.libs_env.env_cliff_gui
import libs.libs_env.env_cliff
import agent

env = libs.libs_env.env_cliff_gui.EnvCliffGui()
#env = libs.libs_env.env_cliff.EnvCliff()

env.print_info()

agent = agent.Agent(env)


training_iterations = 10000

for i in range(0, training_iterations):
    agent.main()


env.reset_score()
agent.run_best_enable()

testing_iterations = 100000

for i in range(0, testing_iterations):
    agent.main()

    env.render()

    if (i%1000) == 0:
        score = env.get_score()
        print("iterations = ", i, "score = ", score)
