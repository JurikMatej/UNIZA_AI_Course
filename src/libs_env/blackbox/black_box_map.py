import noise
import random
import numpy

from scipy.misc import toimage

class BlackBoxMap:
    def __init__(self, width = 256, height = 256, init_seed_x = -1, init_seed_y = -1):
        self.width   = width
        self.height  = height


        scale = 100.0
        octaves = 10
        persistence = 0.3
        lacunarity = 2.0


        if init_seed_x == -1:
            seed_x = random.randint(0, 65536)
        else:
            seed_x = init_seed_x

        if init_seed_y == -1:
            seed_y = random.randint(0, 65536)
        else:
            seed_y = init_seed_y

        self.map = numpy.zeros((self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                x_ = x + seed_x
                y_ = y + seed_y
                self.map[y][x] = noise.pnoise2( x_*10.0/width,
                                                y_*10.0/height,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=self.width,
                                                repeaty=self.height,
                                                base=0)


        self.__normalise()

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get(self):
        return self.map

    def __normalise(self):
        max = numpy.max(self.map)
        min = numpy.min(self.map)
        k = (1.0 - 0.0)/(max-min)
        q = 1.0 - k*max
        self.map = self.map*k + q
