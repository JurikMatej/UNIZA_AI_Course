import mylibs.agent
import mylibs.QAgent
import libs_env.env_cliff


env = libs_env.env_cliff.EnvCliff()

#agent_ervin = mylibs.agent.Agent(env)
agent_ervin = mylibs.QAgent.QAgent(env)

training_iterations = 10000

for iteration in range(0, training_iterations):
    agent_ervin.main()




testing_iterations  = 10000

env.reset_score()
agent_ervin.run_best_enable()

for iteration in range(0, testing_iterations):
    agent_ervin.main()

    env._print()


print("program done")
