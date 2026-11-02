from collections import deque
import random


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def append(self, new_experience):
        self.memory.append(new_experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
