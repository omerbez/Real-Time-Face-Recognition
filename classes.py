import numpy as np


class Point:

    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    def getX(self):
        return self.__x

    def getY(self):
        return self.__y

    def setX(self, newX):
        self.__x = newX

    def setY(self, newY):
        self.__y = newY

    def distanceFrom(self, other):
        return np.sqrt((self.__x - other.__x)**2 + (self.__y - other.__y)**2)
