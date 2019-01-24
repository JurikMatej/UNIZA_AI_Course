import libs_env.env_arkanoid
import libs_agent.agent_dqn
import libs_agent.agent

#init cliff environment
env = libs_env.env_arkanoid.EnvArkanoid()

#print environment info
env.print_info()

'''
agent = lib.agent.Agent(env)
while True:
    agent.main()

    print("move=", env.get_move(), " score=",env.get_score())
    env.render()
'''

#init DQN agent
agent = lib.agent_dqn.DQNAgent(env, "networks/arkanoid_network_parameters.json")

#simulate training
training_iterations = 100000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        print(iteration*100.0/training_iterations, env.get_score())

#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 2000
#for iteration in range(0, testing_iterations):
while True:
    agent.main()

    print("move=", env.get_move(), " score=",env.get_score())
    env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score())
