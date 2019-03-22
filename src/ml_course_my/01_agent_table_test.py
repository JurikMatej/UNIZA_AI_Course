import sys
sys.path.append("..") # Adds higher directory to python modules path.

import libs.libs_env.env_cliff_gui
import agent_table

env = libs.libs_env.env_cliff_gui.EnvCliffGui()

env.print_info()

agent = agent_table.QLearningAgentTable(env)
#agent = agent_table.SarsaAgentTable(env, 0.4)


training_iterations = 100000

for i in range(0, training_iterations):
    agent.main()

agent.print_q_table()

'''
env.reset_score()
agent.run_best_enable()

testing_iterations = 100000

for i in range(0, testing_iterations):
    agent.main()

    env.render()

    if (i%1000) == 0:
        score = env.get_score()
        print("iterations = ", i, "score = ", score)
'''
