import numpy as np


class Transition(object):
    '''Store Q learning transition

    Attributes:
        state (tensor): 80 x 80 2d tensor containing old game state data
        action (int): Action taken during old game state
        next_state (tensor): 80 x 80 2d tensor containing new game state data
        reward (int): Reward given to agent by environment
        done (bool): Indicates whether next_state is a terminal state

    '''

    def __init__(self, state, action, next_state, reward, done):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done


class ReplayMemory(object):
    '''Circular buffer to store pervious state transitions

    Attributes:
        capacity (int): The number of transisions to maintain
        memory (list(Transition)): List containing previous state transitions
        position (int): Next position to be evicted when attempting to push
            into full buffer
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        '''Add transistion to replay memory if buffer is full replace least
        recently added transition

        Attributes:
            transition (Transition): Transition to be added to replay memory
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, replace=True):
        '''Sample random batch of transition

        Attributes:
            batch_size (int): The number of transisions to return
            replace (bool): Indicates whether batches are sample with
                replacement defaults to True
        '''
        return np.random.choice(self.memory, batch_size, replace=True)
