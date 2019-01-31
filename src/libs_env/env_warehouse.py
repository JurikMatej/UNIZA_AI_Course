import numpy
import random
import time

import libs_env.env
import libs_gl_gui.gl_gui as gl_gui

class Bot:

    def __init__(self, max_x, max_y, type = 0, position_x = -1, position_y = -1):
        self.position_x = position_x
        self.position_y = position_y
        self.max_x      = max_x
        self.max_y      = max_y
        self.type       = type

        self.action     = 0

        if self.position_x < 0 or self.position_y < 0:
            self.random_position()

    def get_position_x(self):
        return self.position_x

    def get_position_y(self):
        return self.position_y

    def get_position(self):
        return [self.position_x, self.position_y]

    def get_type(self):
        return self.type

    def get_action(self):
        return self.action

    def random_position(self):
        self.position_x = random.randint(0, self.max_x)
        self.position_y = random.randint(0, self.max_y)

    def do_action(self, action):

        if action == 0:
            self.position_x+= 1
        elif action == 1:
            self.position_x-= 1
        elif action == 2:
            self.position_y+= 1
        elif action == 3:
            self.position_y-= 1

        if self.position_x < 0:
            self.position_x = 0
        if self.position_x >= self.max_x:
            self.position_x = self.max_x

        if self.position_y < 0:
            self.position_y = 0
        if self.position_y >= self.max_y:
            self.position_y = self.max_y

        self.action = action

    def _print(self):
        print(self.position_x, self.position_y, self.type)



class EnvWarehouse(libs_env.env.Env):

    def __init__(self):

        #init parent class -> environment interface
        libs_env.env.Env.__init__(self)

        self.bots_count  = 1
        self.size        = 8

        self.actions_count  = 4*self.bots_count

        #dimensions
        self.width  = self.size
        self.height = self.size
        self.depth  = self.bots_count + 1

        self.start_x = self.width//2
        self.start_y = self.height//2

        #initial bots positions
        #create bots
        self.__spawn_bots(self.bots_count)


        self.reward_fields = []
        #target fields
        #negative rewards
        self.reward_fields.append(Bot(self.width - 1, self.height - 1, -1.0, 0, 0))
        self.reward_fields.append(Bot(self.width - 1, self.height - 1, -1.0, self.width - 1, 0))
        self.reward_fields.append(Bot(self.width - 1, self.height - 1, -1.0, 0, self.height - 1))
        self.reward_fields.append(Bot(self.width - 1, self.height - 1, -1.0, self.width - 1, self.height - 1))

        #positive rewards
        self.reward_fields.append(Bot(self.width - 1, self.height - 1,  1.0, 0, (self.height - 1)//2))
        self.reward_fields.append(Bot(self.width - 1, self.height - 1,  1.0, self.width - 1, (self.height - 1)//2))
        self.reward_fields.append(Bot(self.width - 1, self.height - 1,  1.0, (self.width - 1)//2, 0))
        self.reward_fields.append(Bot(self.width - 1, self.height - 1,  1.0, (self.width - 1)//2, self.height - 1))

        self.item = Bot(self.width - 1, self.height - 1, 0.0, self.start_x, self.start_y)

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation    = numpy.zeros(self.get_size())

        self.__update_observation()

        self.gui = gl_gui.GLVisualisation()


    def _print(self):

        print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        #self.render()

    def render(self):
        self.gui.start()

        element_size = 1.9/self.width

        for y in range(0, self.height):
            for x in range(0, self.width):
                self.gui.push()

                self.gui.translate(self.x_to_gui_x(x), self.y_to_gui_y(y), -0.001)
                self.gui.set_color(0.2, 0.2, 0.2)
                self.gui.paint_square(element_size)

                self.gui.pop()

        for i in range(0, len(self.bots)):
            self.gui.push()

            x = self.bots[i].get_position_x()
            y = self.bots[i].get_position_y()

            self.gui.translate(self.x_to_gui_x(x), self.y_to_gui_y(y), 0.001)
            self.gui.set_color(0.9, 0.9, 0.0)
            self.gui.paint_square(element_size)

            self.gui.pop()

        for i in range(0, len(self.reward_fields)):
            self.gui.push()

            x = self.reward_fields[i].get_position_x()
            y = self.reward_fields[i].get_position_y()
            v = self.reward_fields[i].get_type()



            self.gui.translate(self.x_to_gui_x(x), self.y_to_gui_y(y), 0.0)

            if v > 0.0:
                self.gui.set_color(0.0, 0.9, 0.0)
            elif v < 0.0:
                self.gui.set_color(0.9, 0.0, 0.0)
            else:
                self.gui.set_color(0.0, 0.0, 0.9)

            self.gui.paint_square(element_size)

            self.gui.pop()

            self.gui.push()

        x = self.item.get_position_x()
        y = self.item.get_position_y()

        self.gui.translate(self.x_to_gui_x(x), self.y_to_gui_y(y), 0.001)
        self.gui.set_color(0.9, 0.9, 0.9)
        self.gui.paint_square(element_size)

        self.gui.pop()

        self.gui.finish()
        time.sleep(0.05)





    def do_action(self, action):

        bot_idx     = action//4
        action_idx  = action%4

        self.bots[bot_idx].do_action(action_idx)

        if self.bots[bot_idx].get_position() == self.item.get_position():
            bot_action = self.bots[bot_idx].get_action()
            self.item.do_action(bot_action)

        self.reward = -0.001
        self.set_no_terminal_state()

        for i in range(0, len(self.reward_fields)):
            if self.item.get_position() == self.reward_fields[i].get_position():
                self.reward = self.reward_fields[i].get_type()
                self.set_terminal_state()
                self.reset()

        self.__update_observation()
        self.next_move()




    def x_to_gui_x(self, x):
        return (x*1.0/self.width - 0.5)*2.0

    def y_to_gui_y(self, y):
        return (y*1.0/self.height - 0.5)*2.0

    def reset(self):
        self.item = Bot(self.width - 1, self.height - 1, 0.0, self.start_x, self.start_y)
        self.__spawn_bots(self.bots_count)

    def __update_observation(self):
        #clear state tensor, one only on agent position
        self.observation.fill(0.0)

        self.observation[self.__to_idx(self.item.get_position_x(), self.item.get_position_y(), 0)] = 1.0
        for i in range(0, len(self.bots)):
            self.observation[self.__to_idx(self.bots[i].get_position_x(), self.bots[i].get_position_y(), i+1)] = 1.0

        for i in range(0, len(self.observation)):
            r = (random.random() - 0.5)*2.0*0.1
            self.observation[i]+= r

    def __to_idx(self, x, y, z):
        return int((z*self.height + y)*self.width + x)

    def __spawn_bots(self, bots_count):
        self.bots = [ ]
        for i in range(0, bots_count):
            self.bots.append(Bot(self.width - 1, self.height - 1, 0))

        for i in range(0, bots_count):
            self.bots[i].random_position()
            while (self.bots[i].get_position_x() == self.start_x) and (self.bots[i].get_position_y() == self.start_y):
                self.bots[i].random_position()
