import math
import numpy as np


class Variable:
    def __init__(self, name, is_binary, value):
        self.name = name
        self.is_binary = is_binary
        self.value = value

