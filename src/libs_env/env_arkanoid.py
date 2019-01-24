import libs_env.env
import gl_gui.gl_gui as gl_gui

import numpy
import time
import random



class EnvArkanoid(libs_env.env.Env):

    def __init__(self):

        #init parent class -> environment interface
        libs_env.env.Env.__init__(self)

        width  = 8
        height = 20

        #dimensions 1x10x10
        self.width  = 2*width
        self.height = height
        self.depth  = 3

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation    = numpy.zeros(self.get_size())

        #4 actions for movements
        self.actions_count  = 3

        self.player_size = 3

        self.reset()
        self.__position_to_state()

        self.gui = gl_gui.GLVisualisation()

        self.size_ratio = self.width/self.height



    def reset(self):
        #initial player position
        self.player = self.width/2

        self.board    = numpy.zeros((self.height, self.width))

        for x in range(0, self.width):
            self.board[4][x] = 1
            self.board[5][x] = 2
            self.board[6][x] = 3
            self.board[7][x] = 4
            self.board[8][x] = 5
            self.board[9][x] = 6

        self.__reset_ball()

    def __reset_ball(self):
        self.ball_x = int(self.width/2) + random.randint(-1, 1)
        self.ball_y = int(self.height/2)+ random.randint(-1, 1)

        if random.randint(0, 1) == 0:
            self.ball_dx = 1
        else:
            self.ball_dx = -1

        if random.randint(0, 1) == 0:
            self.ball_dy = 1
        else:
            self.ball_dy = -1



    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        self.render()

    def render(self):
        self.gui.init("arkanoid", 32*self.width, 32*self.height)

        self.gui.start()

        if self.height > self.width:
            element_size = 1.9/self.height
        else:
            element_size = 1.9/self.width

        for y in range(0, self.height):
            for x in range(0, self.width):
                    self.gui.push()

                    color = self.__item_to_color(self.board[y][x])

                    self.gui.set_color(color[0], color[1], color[2])

                    self.gui.translate(self.x_to_gui_x(x), self.y_to_gui_y(y), 0.0)
                    self.gui.paint_square(element_size)
                    self.gui.pop()

        self.gui.push()
        color = self.__item_to_color(10)
        self.gui.set_color(color[0], color[1], color[2])
        self.gui.translate(self.x_to_gui_x(self.ball_x), self.y_to_gui_y(self.ball_y), 0.0)
        self.gui.paint_square(element_size)
        self.gui.pop()

        for i in range(0, self.player_size):
            self.gui.push()
            color = self.__item_to_color(11)
            self.gui.set_color(color[0], color[1], color[2])
            self.gui.translate(self.x_to_gui_x(self.player + i - self.player_size//2), self.y_to_gui_y(self.height-1), 0.0)
            self.gui.paint_square(element_size)
            self.gui.pop()

        self.gui.finish()
        time.sleep(0.001)

    def do_action(self, action):

        self.ball_x+= self.ball_dx
        self.ball_y+= self.ball_dy

        self.ball_x = self.__saturate(self.ball_x, 0, self.get_width()-1)
        self.ball_y = self.__saturate(self.ball_y, 0, self.get_height()-1)

        colission = False
        if self.board[self.ball_y][self.ball_x] != 0:
            self.board[self.ball_y][self.ball_x] = 0
            if self.ball_x+1 < self.get_width():
                if self.board[self.ball_y][self.ball_x+1] != 0:
                    self.board[self.ball_y][self.ball_x+1] = 0
            colission = True


        if colission:
            self.ball_dy*= -1

        if action == 0:
            self.player+= 0
        elif action == 1:
            self.player+= 1
        else:
            self.player-= 1

        self.player = self.__saturate(self.player, 0, self.width)

        self.reward = 0.0

        if colission:
            self.reward = 0.25

        if self.ball_x <= 0:
            self.ball_dx = 1


        if self.ball_x >=  self.width-1:
            self.ball_dx = -1

        if self.ball_y >= self.height-1:
            if self.player-self.ball_x == 0:
                self.ball_dy = -1
            elif self.player-self.ball_x == 1:
                self.ball_dy = -1
                self.ball_dx*= -1
            elif self.player-self.ball_x == -1:
                self.ball_dy = -1
                self.ball_dx*= -1
            else:
                self.__reset_ball()
                self.reward = -1.0

        if self.ball_y <= 0:
            self.ball_dy = 1

        self.set_no_terminal_state()
        if self.__count_remaining() < 10:
            self.reward = 1.0
            self.reset()
            self.set_terminal_state()

        self.__position_to_state()

        self.next_move()

    def x_to_gui_x(self, x):
        return self.size_ratio*(x*1.0/self.width - 0.5)*2.0

    def y_to_gui_y(self, y):
        return -(y*1.0/self.height - 0.5)*2.0

    def __position_to_state(self):
        ball_x = self.__saturate(int(self.ball_x), 0, self.width-1)
        ball_y = self.__saturate(int(self.ball_y), 0, self.height-1)

        player_0 = self.__saturate(int(self.player_0), 0, self.height-1)
        player_1 = self.__saturate(int(self.player_1), 0, self.height-1)

        self.observation.fill(0.0)
        self.observation[ball_y*self.get_width() + ball_x] = 1.0
        self.observation[player_0*self.get_width() + 0] = 1.0
        self.observation[player_1*self.get_width() + self.get_width()-1] = 1.0

    def __item_to_color(self, item_idx):
        result = [0.2, 0.2, 0.2]

        if item_idx == 1:
            result = [1.0, 0.0, 0.0]
        elif item_idx == 2:
            result = [1.0, 0.5, 0.0]
        elif item_idx == 3:
            result = [1.0, 0.75, 0.0]
        elif item_idx == 4:
            result = [1.0, 1.0, 0.0]
        elif item_idx == 5:
            result = [0.0, 1.0, 0.0]
        elif item_idx == 6:
            result = [0.0, 0.0, 1.0]
        elif item_idx == 10:
            result = [1.0, 1.0, 1.0]
        elif item_idx == 11:
            result = [1.0, 1.0, 1.0]
        return result

    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value

    def __position_to_state(self):
        ball_x = self.__saturate(int(self.ball_x), 0, self.width-1)
        ball_y = self.__saturate(int(self.ball_y), 0, self.height-1)

        player_x = self.__saturate(int(self.player), 0, self.width-1)
        player_y = self.height-1

        self.observation.fill(0.0)

        for y in range(0, self.height):
            for x in range(0, self.width):
                if self.board[y][x]!= 0:
                    color = self.__item_to_color(self.board[y][x])

                    self.observation[(0*self.height + y)*self.get_width() + x] = color[0]
                    self.observation[(1*self.height + y)*self.get_width() + x] = color[1]
                    self.observation[(2*self.height + y)*self.get_width() + x] = color[2]

        color = self.__item_to_color(10)
        self.observation[(0*self.height + ball_y)*self.get_width() + ball_x] = color[0]
        self.observation[(1*self.height + ball_y)*self.get_width() + ball_x] = color[1]
        self.observation[(2*self.height + ball_y)*self.get_width() + ball_x] = color[2]

        color = self.__item_to_color(11)
        self.observation[(0*self.height + player_y)*self.get_width() + player_x] = color[0]
        self.observation[(1*self.height + player_y)*self.get_width() + player_x] = color[1]
        self.observation[(2*self.height + player_y)*self.get_width() + player_x] = color[2]

    def __count_remaining(self):
        count = 0
        for y in range(0, self.height):
            for x in range(0, self.width):
                if self.board[y][x] > 0.0:
                    count+= 1

        return count
