import numpy
import random
import time

import libs_env.env
import libs_gl_gui.gl_gui as gl_gui

class Bot:
    def __init__(self, position_x, position_y, max_x, max_y):
        self.position_x = position_x
        self.position_y = position_y
        self.max_x      = max_x
        self.max_y      = max_y


    def get_position_x(self):
        return self.position_x

    def get_position_y(self):
        return self.position_y

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

    def _print(self):
        print(self.position_x, self.position_y, self.state)



class EnvWarehouse(libs_env.env.Env):

    def __init__(self):

        #init parent class -> environment interface
        libs_env.env.Env.__init__(self)

        bots_count    = 1
        sources_count = 10

        #dimensions 1x4x8
        self.width  = 8
        self.height = 8
        self.depth  = 3


        self.target_x = self.width//2
        self.target_y = self.height//2

        #4 actions for movements
        self.actions_count  = 4
        self.active_bot     = 0


        #initial bots positions
        #create bots
        self.bots = [ ]
        for i in range(0, bots_count):
            self.bots.append(Bot(self.target_x, self.target_y, self.width - 1, self.height - 1))

        #create sources
        self.sources = [ ]
        for i in range(0, sources_count):
            self.sources.append(Bot(0, 0, self.width - 1, self.height - 1))
        for i in range(0, len(self.sources)):
            self.sources[i].random_position()


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

            self.gui.set_color(1.0, 1.0, 1.0)
            self.gui.paint_square(element_size)

            self.gui.pop()

        for i in range(0, len(self.sources)):
            self.gui.push()

            x = self.sources[i].get_position_x()
            y = self.sources[i].get_position_y()

            self.gui.translate(self.x_to_gui_x(x), self.y_to_gui_y(y), 0.0)


            self.gui.set_color(0.0, 1.0, 0.0)
            self.gui.paint_square(element_size)

            self.gui.pop()

        x = self.target_x
        y = self.target_y

        self.gui.push()
        self.gui.set_color(1.0, 1.0, 0.0)
        self.gui.translate(self.x_to_gui_x(x), self.y_to_gui_y(y), 0.001)
        self.gui.paint_square(element_size)
        self.gui.pop()

        print(self.move, self.score, self.reward)
        self.print_state()

        self.gui.finish()
        time.sleep(0.0005)





    def do_action(self, action):
        self.bots[self.active_bot].do_action(action)

        self.reward = -0.01

        self.set_no_terminal_state()

        for i in range(0, len(self.sources)):
            if self.bots[self.active_bot].get_position_x() == self.sources[i].get_position_x():
                if self.bots[self.active_bot].get_position_y() == self.sources[i].get_position_y():
                    self.sources[i].random_position()
                    self.reward+= 1.0
                    self.set_terminal_state()

        '''
        if self.bots[self.active_bot].get_state() == 0:
            for i in range(0, len(self.sources)):
                if self.sources[i].get_state() != 0:
                    if self.bots[self.active_bot].get_position_x() == self.sources[i].get_position_x():
                        if self.bots[self.active_bot].get_position_y() == self.sources[i].get_position_y():
                            self.sources[i].set_state(0)
                            self.bots[self.active_bot].set_state(1)
                            self.reward+= 1.0


        if self.bots[self.active_bot].get_position_x() == self.target_x:
            if self.bots[self.active_bot].get_position_y() == self.target_y:
                self.bots[self.active_bot].set_state(0)
                self.reward = 1.0
                self.set_terminal_state()
        '''
        self.active_bot = (self.active_bot+1)%len(self.bots)

        self.__update_observation()
        self.next_move()




    def __update_observation(self):
        #clear state tensor, one only on agent position
        self.observation.fill(0.0)


        x = self.bots[self.active_bot].get_position_x()
        y = self.bots[self.active_bot].get_position_y()
        self.observation[self.__to_idx(x, y, 0)] = 1.0


        for i in range(0,len(self.bots)):
            x = self.bots[i].get_position_x()
            y = self.bots[i].get_position_y()
            self.observation[self.__to_idx(x, y, 1)] = 1.0


        for i in range(0, len(self.sources)):
            x = self.sources[i].get_position_x()
            y = self.sources[i].get_position_y()
            self.observation[self.__to_idx(x, y, 2)] = 1.0




    def __to_idx(self, x, y, z):
        return int((z*self.height + y)*self.width + x)


    def x_to_gui_x(self, x):
        return (x*1.0/self.width - 0.5)*2.0

    def y_to_gui_y(self, y):
        return (y*1.0/self.height - 0.5)*2.0
