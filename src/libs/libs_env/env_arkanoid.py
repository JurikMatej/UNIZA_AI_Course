import libs.libs_env.env as libs_env
import libs.libs_gl_gui.gl_gui as libs_gl_gui


import numpy
import time
import random



class EnvArkanoid(libs_env.Env):

    def __init__(self):

        #init parent class -> environment interface
        libs_env.Env.__init__(self)

        #dimensions
        self.width  = 16
        self.height = 20
        self.depth  = 3

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation    = numpy.zeros(self.get_size())

        #3 actions for movements
        self.actions_count  = 3

        #player paddle size
        self.player_size = 3

        #init game
        self.reset()
        self.__update_state()

        self.gui = libs_gl_gui.GLVisualisation()
        self.size_ratio = self.width/self.height

        self.moves_old = 0
        self.game_idx = 0
        self.moves_to_win = 1000


    def reset(self):
        self.board    = numpy.zeros((self.height, self.width))

        for x in range(0, self.width):
            self.board[4][x] = 1
            self.board[5][x] = 2
            self.board[6][x] = 3
            self.board[7][x] = 4
            self.board[8][x] = 5
            self.board[9][x] = 6

        self.__respawn()

    def __respawn(self):
        #init player position
        self.player_position = self.width/2

        #init ball position
        self.ball_x = int(self.width/2) + random.randint(-1, 1)
        self.ball_y = int(self.height/2 + 1) + random.randint(-1, 1)

        self.ball_dx = 1
        self.ball_dy = 1


        if random.randint(0, 1) == 0:
            self.ball_dx = 1
        else:
            self.ball_dx = -1

        if random.randint(0, 1) == 0:
            self.ball_dy = 1
        else:
            self.ball_dy = -1


    def _print(self):
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        print("done game ", self.game_idx, " moves ", self.moves_to_win, " score ", self.get_score())

    def do_action(self, action):

        '''
        if self.player_position < self.ball_x:
            action = 0
        else:
            action = 1
        '''
        if action == 0:
            self.player_position+= 1

        if action == 1:
            self.player_position-= 1

        self.player_position = self.__saturate(self.player_position, 1, self.width-2)

        result = self.__process_ball()


        self.reward = 0.0
        self.set_no_terminal_state()


        if result == "brick":
            self.reward = 0.5

        if result == "miss":
            self.reward = -1.0
            self.set_terminal_state()
            self.__respawn()

        if self.__count_remaining() <= 2*4:
            self.reward = 1.0
            self.set_terminal_state()
            self.reset()

            self.moves_to_win = self.get_move() - self.moves_old
            self.moves_old = self.get_move()
            self.game_idx+= 1



        self.__update_state()
        self.next_move()

    def __update_state(self):
        self.observation.fill(0.0)


        for y in range(0, self.height):
            for x in range(0, self.width):
                if self.board[y][x]!= 0:
                    self.__observation_set_dirrect(x, y, 2, 1.0)



        ball_x = self.__saturate(int(self.ball_x), 0, self.width-1)
        ball_y = self.__saturate(int(self.ball_y), 0, self.height-1)

        player_x = self.__saturate(int(self.player_position), 0, self.width-1)
        player_y = self.height-1

        self.__observation_set_dirrect(ball_x, ball_y, 0, 1.0)
        self.__observation_set_dirrect(player_x, player_y, 1, 1.0)

        for i in range(0, len(self.observation)):
            self.observation[i]+= random.random()*0.01




    def __observation_set(self, x, y, color):
        self.observation[(0*self.height + y)*self.get_width() + x] = color[0]
        self.observation[(1*self.height + y)*self.get_width() + x] = color[1]
        self.observation[(2*self.height + y)*self.get_width() + x] = color[2]

    def __observation_set_dirrect(self, x, y, z, value):
        self.observation[(z*self.height + y)*self.get_width() + x] = value


    def render(self):
        self.gui.init("arkanoid", 32*self.width, 32*self.height)

        self.gui.start()

        if self.height > self.width:
            element_size = 1.99/self.height
        else:
            element_size = 1.99/self.width

        for y in range(0, self.height):
            for x in range(0, self.width):
                    self.gui.push()

                    color = self.__item_to_color(self.board[y][x])

                    self.gui.set_color(color[0], color[1], color[2])

                    self.gui.translate(self.__x_to_gui_x(x), self.__y_to_gui_y(y), 0.0)
                    self.gui.paint_square(element_size)
                    self.gui.pop()

        self.gui.push()
        color = self.__item_to_color(10)
        self.gui.set_color(color[0], color[1], color[2])
        self.gui.translate(self.__x_to_gui_x(self.ball_x), self.__y_to_gui_y(self.ball_y), 0.0)
        self.gui.paint_square(element_size)
        self.gui.pop()

        for i in range(0, self.player_size):
            self.gui.push()
            color = self.__item_to_color(11)
            self.gui.set_color(color[0], color[1], color[2])
            self.gui.translate(self.__x_to_gui_x(self.player_position + i - self.player_size//2), self.__y_to_gui_y(self.height-1), 0.0)
            self.gui.paint_square(element_size)
            self.gui.pop()

        self.gui.set_color(1.0, 1.0, 1.0)
        count = "SCORE = " + str(round(self.get_score(), 3))
        self.gui._print(-0.3, 0.95, 0.1, count)

        self.gui.finish()
        time.sleep(0.05)

    def __process_ball(self):

        result = "none"

        if self.board[self.ball_y][self.ball_x] != 0:
            x1 = 2*(self.ball_x//2) + 0
            x2 = 2*(self.ball_x//2) + 1

            x1 = self.__saturate(x1, 0, self.width-1)
            x2 = self.__saturate(x2, 0, self.width-1)

            self.board[self.ball_y][x1] = 0
            self.board[self.ball_y][x2] = 0

            self.ball_dy*= -1
            result = "brick"

        if self.ball_x <= 0:
            self.ball_dx = 1

        if self.ball_x >= self.width-1:
            self.ball_dx = -1

        if self.ball_y <= 0:
            self.ball_dy = 1

        if self.ball_y >= self.height-2:
            if self.player_position == self.ball_x or self.player_position-1 == self.ball_x or self.player_position+1 == self.ball_x:
                result = "hit"
                self.ball_dy = -1
            else:
                result = "miss"


        self.ball_x+= self.ball_dx
        self.ball_y+= self.ball_dy

        self.ball_x = self.__saturate(self.ball_x, 0, self.width-1)
        self.ball_y = self.__saturate(self.ball_y, 0, self.height-1)


        return result

    def __x_to_gui_x(self, x):
        return self.size_ratio*(x*1.0/self.width - 0.5)*2.0

    def __y_to_gui_y(self, y):
        return -(y*1.0/self.height - 0.5)*2.0

    def __item_to_color(self, item_idx):
        result = [0.0, 0.0, 0.0]

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

    def __count_remaining(self):
        count = 0
        for y in range(0, self.height):
            for x in range(0, self.width):
                if self.board[y][x] > 0.0:
                    count+= 1

        return count
