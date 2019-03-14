import libs.libs_env.env_arkanoid
import libs.libs_agent.agent_dqn
import libs.libs_agent.agent

#init cliff environment
env = libs.libs_env.env_arkanoid.EnvArkanoid()

#print environment info
env.print_info()

'''
agent = libs.libs_agent.agent.Agent(env)
while True:
    agent.main()

    print("move=", env.get_move(), " score=",env.get_score())
    env.render()
'''


#init DQN agent
agent = libs.libs_agent.agent_dqn.DQNAgent(env, "networks/arkanoid_network_b/parameters.json", 0.2, 0.02, 0.99999)


#process training
training_iterations = 250000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        env._print()

agent.save("networks/arkanoid_network_b/trained/")


agent.load("networks/arkanoid_network_b/trained/")


#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 10000
for iteration in range(0, testing_iterations):
    agent.main()
    env._print()


while True:
    agent.main()
    env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score())
