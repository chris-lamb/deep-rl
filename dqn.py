import sys
import argparse
import pickle

import gym
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from models import DQN
from replay_memory import Transition, ReplayMemory
from utils import preprocess_state


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', action='store', dest='game', default='Breakout-v0')
    parser.add_argument('-w', action='store_true', dest='warm_start', default=False)

    args = parser.parse_args()
    game = args.game
    warm_start = args.warm_start

    return game, warm_start


def initialize(game, model_name, warm_start):
    # Initialize environment
    env = gym.make(game)
    num_actions = env.action_space.n

    # Initialize constants
    num_frames = 4
    capacity = int(5e4)

    # Cold start
    if not warm_start:
        # Initialize model
        model = DQN(in_channels=num_frames, num_actions=num_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=1.0e-4, weight_decay=0.01)
        # Initialize replay memory
        memory_buffer = ReplayMemory(capacity)

        # Initialize statistics
        running_reward = None
        running_rewards = []

    # Warm start
    if warm_start:

        data_file = 'results/{}_{}.p'.format(game, model_name)

        try:
            with open(data_file, 'rb') as f:
                running_rewards = pickle.load(f)
                running_reward = running_rewards[-1]

            prior_eps = len(running_rewards)

            model_file = 'saved_models/{}_{}_ep_{}.p'.format(game, model_name, prior_eps)
            with open(model_file, 'rb') as f:
                saved_model = pickle.load(f)
                model, optimizer, memory_buffer = saved_model

        except OSError:
            print('Saved file not found. Creating new cold start model.')
            model = DQN(in_channels=num_frames, num_actions=num_actions)
            optimizer = optim.RMSprop(model.parameters(), lr=1.0e-4, weight_decay=0.01)
            # Initialize replay memory
            memory_buffer = ReplayMemory(capacity)

            running_reward = None
            running_rewards = []

    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda()

    criterion = torch.nn.MSELoss()

    return env, model, optimizer, criterion, memory_buffer, cuda, running_reward, running_rewards


def select_epilson_greedy_action(model, state, t, cuda):
    sample = np.random.rand()
    # Anneal epsilon from 1.0 down to 0.05 over 20,000 iterations
    epsilon = max(0.05, 1.0 - 0.95 * (t / 2.0e4))
    if epsilon > sample:
        # Select random Action
        action = np.random.randint(model.num_actions)
    else:
        # Select best action
        num_frames, height, width = state.shape
        state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

        if cuda:
            state = state.cuda()

        state = Variable(state)
        action = model(state).data.max(1)[1]

    return action


def main():

    model_name = 'dqn'

    # Parse arguments
    game, warm_start = parse_arguments()

    # Initialize enviroment/model
    data = initialize(game, model_name, warm_start)
    env, model, optimizer, criterion, memory_buffer, cuda, running_reward, running_rewards = data

    # Initialize constants
    max_episodes = 500000
    batch_size = 10
    gamma = 0.95
    num_frames = 4

    for ep in range(max_episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)
        reward_sum = 0.0

        while True:
            # Select action
            action = select_epilson_greedy_action(model, state, ep, cuda)

            # Perform step
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.stack([next_state]*num_frames)
            next_state[1:, :, :] = state[:-1, :, :]

            reward_sum += reward

            # Add transition to replay memory
            transition = Transition(state, action, next_state, reward, done)
            memory_buffer.push(transition)

            # Update state
            state = next_state

            # Sample mini-batch from replay memory_buffer
            batch = memory_buffer.sample(batch_size, replace=True)

            # Compute targets
            targets = np.zeros((batch_size,), dtype=float)
            for i, transition in enumerate(batch):
                targets[i] = transition.reward
                if not transition.done:
                    next_state = transition.next_state
                    num_frames, height, width = next_state.shape
                    next_state = next_state.reshape(-1, num_frames, height, width)
                    next_state = torch.FloatTensor(next_state)

                    if cuda:
                        next_state = next_state.cuda()

                    next_state = Variable(next_state)
                    targets[i] += gamma * model(next_state).data.max(1)[0]

            targets = torch.FloatTensor(targets)

            if cuda:
                targets = targets.cuda()

            targets = Variable(targets)

            # Compute predictions
            model.zero_grad()

            states = [transition.state for transition in batch]
            states = torch.FloatTensor(states)

            if cuda:
                states = states.cuda()

            states = Variable(states)

            actions = [int(transition.action) for transition in batch]
            actions = torch.LongTensor(actions)

            if cuda:
                actions = actions.cuda()

            actions = Variable(actions)

            outputs = model(states).gather(1, actions.unsqueeze(1))

            # Perform gradient descent step
            loss = criterion(outputs, targets)
            loss.backward()
            # Clip gradient at 20,000
            torch.nn.utils.clip_grad_norm(model.parameters(), 20000)
            optimizer.step()

            if done:
                break

        # Compute/display statistics
        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = running_reward * 0.99 + reward_sum * 0.01

        running_rewards.append(running_reward)

        verbose_str = 'Episode {} complete'.format(ep+1)
        verbose_str += '\tReward total:{}'.format(reward_sum)
        verbose_str += '\tRunning mean: {:.4}'.format(running_reward)
        sys.stdout.write('\r' + verbose_str)
        sys.stdout.flush()

        # Save model every 1000 episodes
        if (ep+1) % 1000 == 0:
            model_file = 'saved_models/{}_{}_ep_{}.p'.format(game, model_name, ep+1)

            with open(model_file, 'wb') as f:
                pickle.dump((model.cpu(), optimizer, memory_buffer), f)

            if cuda:
                model = model.cuda()

            data_file = 'results/{}_{}.p'.format(game, model_name)

            with open(data_file, 'wb') as f:
                pickle.dump(running_rewards, f)


if __name__ == '__main__':
    main()
