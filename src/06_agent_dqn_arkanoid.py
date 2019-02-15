import libs_env.env_arkanoid
import libs_agent.agent_dqn
import libs_agent.agent

#init cliff environment
env = libs_env.env_arkanoid.EnvArkanoid()

#print environment info
env.print_info()

'''
agent = libs_agent.agent.Agent(env)
while True:
    agent.main()

    print("move=", env.get_move(), " score=",env.get_score())
    env.render()
'''


#init DQN agent
agent = libs_agent.agent_dqn.DQNAgent(env, "networks/arkanoid_network_a/parameters.json", 0.2, 0.02, 0.999999)



#process training
training_iterations = 200000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        print(iteration*100.0/training_iterations, env.get_score())

agent.save("networks/arkanoid_network_a/trained/")

agent.load("networks/arkanoid_network_a/trained/")


#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 10000
for iteration in range(0, testing_iterations):
    agent.main()
    print("move=", env.get_move(), " score=",env.get_score())


while True:
    agent.main()
    env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score())
