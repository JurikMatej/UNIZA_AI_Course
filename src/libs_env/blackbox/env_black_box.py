import libs_env.env
import libs_env.blackbox.black_box_map
from scipy.misc import toimage

import numpy
import time
import random


class EnvBlackBox(libs_env.env.Env):

    def __init__(self, seed = 0, convolution_mode = False):
        #init parent class -> environment interface
        libs_env.env.Env.__init__(self)

        self.map_size       = 64

        prime = 60013

        if convolution_mode:
            self.features_count = 4 + (seed*prime)%8
            self.state_size     = 8 #state planar size 8x8
        else:
            self.features_count = 8 + (seed*prime)%24
            self.state_size     = 1 #state planar size 1x1



        #state dimensions
        self.width  = self.state_size
        self.height = self.state_size
        self.depth  = self.features_count

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation    = numpy.zeros(self.get_size())

        #4 actions for movements
        self.actions_count  = 4

        #random permutate actions - to prevent hardcoding bots
        self.actions_ids = numpy.random.permutation(self.actions_count)

        self.agent_position_x = self.map_size//2
        self.agent_position_y = self.map_size//2

        #create feature maps
        self.maps = []
        for i in range(0, self.features_count):
            init_seed_x = (seed + i)*61057
            init_seed_y = (seed + i)*62141
            map = libs_env.blackbox.black_box_map.BlackBoxMap(self.map_size, self.map_size, init_seed_x, init_seed_y)
            self.maps.append(map)

        #compute reward by averaging and thresholding feature maps
        self.rewards = self.__make_rewards()

        self.fields_occupation = numpy.zeros((self.map_size, self.map_size))


    def __make_rewards(self, negative_threshold = 0.4, positive_threshold = 0.7):
        result = numpy.zeros((self.map_size, self.map_size))

        for i in range(0, self.features_count):
            result+= self.maps[i].get()

        max = numpy.max(result)
        min = numpy.min(result)
        k = (1.0 - 0.0)/(max-min)
        q = 1.0 - k*max
        result = result*k + q

        for y in range(self.map_size):
            for x in range(self.map_size):
                if result[y][x] > positive_threshold:
                    result[y][x] = 1.0
                elif result[y][x] < negative_threshold:
                    result[y][x] = -1.0
                else:
                    result[y][x] = 0.0

        return result


    def reset(self):
        self.agent_position_x = self.map_size//2
        self.agent_position_y = self.map_size//2

        self.__position_to_state()

    def _print(self):
        print(self.get_move(), self.get_score(), self.get_observation(), self.reward)
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score(), "state = ", self.get_observation())


    def do_action(self, action):

        action_id = self.actions_ids[action%self.actions_count]

        self.set_no_terminal_state()

        step = 1 + random.randint(0, 1)

        if action_id == 0:
            self.agent_position_x+= step
        elif action_id == 1:
            self.agent_position_x-= step
        elif action_id == 2:
            self.agent_position_y+= step
        elif action_id == 3:
            self.agent_position_y-= step
        elif action_id == 4:
            self.agent_position_x+= step
            self.agent_position_y+= step
        elif action_id == 5:
            self.agent_position_x-= step
            self.agent_position_y+= step
        elif action_id == 6:
            self.agent_position_x+= step
            self.agent_position_y-= step
        elif action_id == 7:
            self.agent_position_x-= step
            self.agent_position_y-= step

        state_size_half = self.state_size//2

        respawn = False
        if self.agent_position_x <= state_size_half:
            respawn = True
        if self.agent_position_x >= (self.map_size-1-state_size_half):
            respawn = True
        if self.agent_position_y <= state_size_half:
            respawn = True
        if self.agent_position_y >= (self.map_size-1-state_size_half):
            respawn = True

        if respawn == True:
            self.agent_position_x = self.map_size//2
            self.agent_position_y = self.map_size//2


        self.reward = self.rewards[self.agent_position_y][self.agent_position_x]

        self.fields_occupation[self.agent_position_y][self.agent_position_x]+= 1

        self.__position_to_state()
        self.next_move()


    def __position_to_state(self):
        self.observation.fill(0.0)

        idx = 0
        for f in range(0, self.features_count):
            for y in range(0, self.state_size):
                for x in range(0, self.state_size):

                    v = self.maps[f].get()[y + self.agent_position_y][x + self.agent_position_x]
                    noise = random.random()*0.01

                    self.observation[idx] = v + noise

                    idx+= 1


    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value
