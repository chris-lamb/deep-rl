import sys

import gym
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from models import DQN
from replay_memory import Transition, ReplayMemory
from utils import preprocess_state


def main():
    render = False

    # Initialize environment
    env = gym.make("Breakout-v0")
    num_actions = env.action_space.n

    # Initialize constants
    num_frames = 4
    batch_size = 10
    capacity = int(5e4)
    max_episodes = 200000
    gamma = 0.95

    # Initialize model
    model = DQN(in_channels=num_frames, num_actions=num_actions)

    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1.0e-4)
    criterion = torch.nn.MSELoss()

    # Initialize replay memory
    memory_buffer = ReplayMemory(capacity)

    running_reward = None
    running_rewards = []
    for ep in range(max_episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)
        reward_sum = 0.0
        while True:
            if render:
                env.render()

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
                break
    return


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


if __name__ == '__main__':
    main()
