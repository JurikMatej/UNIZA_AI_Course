import libs_env.env_warehouse
import libs_agent.agent
import libs_agent.agent_dqn


#init cliff environment
env = libs_env.env_warehouse.EnvWarehouse()


#print environment info
env.print_info()


#init DQN agent
agent = libs_agent.agent_dqn.DQNAgent(env, "networks/warehouse_network/parameters.json", 0.3, 0.1) #0.2, 0.1
#agent = libs_agent.agent.Agent(env)

#process training
training_iterations = 1000000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        env._print()
        print(iteration*100.0/training_iterations, env.get_score())

#agent.save("networks/warehouse_network/trained/")

#agent.load("networks/settlers_network/trained/")

#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 2000
for iteration in range(0, testing_iterations):
    agent.main()
    print("move=", env.get_move(), " score=",env.get_score())


while True:
    agent.main()
    env.render()


print("program done")
print("move=", env.get_move(), " score=",env.get_score())
