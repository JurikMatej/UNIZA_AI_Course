import sys
sys.path.append("..") # Adds higher directory to python modules path.

import libs.libs_env.env_pong
import libs.libs_env.env_arkanoid
import agent
import agent_dqn

#env = libs.libs_env.env_pong.EnvPong()
env = libs.libs_env.env_arkanoid.EnvArkanoid()

env.print_info()

#agent = agent.Agent(env)
agent = agent_dqn.DQNAgent(env, "arkanoid_network.json")

training_iterations = 250000

for i in range(0, training_iterations):
    agent.main()

    if (i%100) == 0:
        progress = 100.0*i/training_iterations
        print("training done = ", progress, " score = ", env.get_score())

agent.save("arkanoid_network/")

env.reset_score()
agent.run_best_enable()

testing_iterations = 10000
for i in range(0, testing_iterations):
    agent.main()

print("********************************")
print("testing score = ", env.get_score())
print("********************************")

while True:
    agent.main()
    env.render()
