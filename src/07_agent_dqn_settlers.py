import libs.libs_env.env_settlers
import libs.libs_agent.agent_dqn
import libs.libs_agent.agent

#init cliff environment
env = libs.libs_env.env_settlers.EnvSettlers()

#print environment info
env.print_info()



#init DQN agent
agent = libs.libs_agent.agent_dqn.DQNAgent(env, "networks/settlers_network/parameters.json", 0.2, 0.1) #0.2, 0.1
#agent = libs_agent.agent.Agent(env)


#process training
training_iterations = 500000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        env._print()
        print(iteration*100.0/training_iterations, env.get_score())

agent.save("networks/settlers_network/trained/")


agent.load("networks/settlers_network/trained/")

#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 10000
for iteration in range(0, testing_iterations):
    agent.main()
    print("move=", env.get_move(), " score=",env.get_score(), " moves to win=",env.get_moves_to_win())


while True:
    agent.main()
    env.render()


print("program done")
print("move=", env.get_move(), " score=",env.get_score())
