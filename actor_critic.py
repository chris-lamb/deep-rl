import sys
import argparse
import pickle
from collections import deque

import gym
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from models import MODELS
from utils import preprocess_state


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', action='store', dest='game', default='Breakout-v0')
    parser.add_argument('-m', action='store', dest='model', default='a2c-lstm')
    parser.add_argument('-w', action='store_true', dest='warm_start', default=False)

    args = parser.parse_args()
    game = args.game
    model_name = args.model
    warm_start = args.warm_start

    return game, model_name, warm_start


def initialize(game, model_name, warm_start):
    # Initialize environment
    env = gym.make(game)
    num_actions = env.action_space.n

    # Initialize constants
    num_frames = 4

    # Cold start
    if not warm_start:
        # Initialize model
        model = MODELS[model_name](in_channels=num_frames, num_actions=num_actions)
        optimizer = optim.RMSprop(model.parameters(), lr=1.0e-4, weight_decay=0.01)

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
                model, optimizer = saved_model

        except OSError:
            print('Saved file not found. Creating new cold start model.')
            model = MODELS[model_name](in_channels=num_frames, num_actions=num_actions)
            optimizer = optim.RMSprop(model.parameters(), lr=1.0e-4, weight_decay=0.01)

            running_reward = None
            running_rewards = []

    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda()

    return env, model, optimizer, cuda, running_reward, running_rewards


def select_action(model, state, cuda):
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    if cuda:
        state = state.cuda()

    probs, state_value = model(Variable(state))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.data[0], log_prob, state_value


def select_action_lstm(model, state, hc, cuda):
    hx, cx = hc
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    if cuda:
        state = state.cuda()

    probs, state_value, (hx, cx) = model((Variable(state), (hx, cx)))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.data[0], log_prob, state_value, (hx, cx)


def backpropagate(model, optimizer, gamma, cuda):
    current_reward = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = deque()

    for r in model.rewards[::-1]:
        current_reward = r + gamma * current_reward
        rewards.appendleft(current_reward)
    rewards = list(rewards)
    rewards = torch.Tensor(rewards)

    if cuda:
        rewards = rewards.cuda()

    # z-score rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    for (log_prob, state_value), r in zip(saved_actions, rewards):
        reward = r - state_value.data[0]
        policy_losses.append(-log_prob * Variable(reward))
        r = torch.Tensor([r])
        if cuda:
            r = r.cuda()
        value_losses.append(torch.nn.functional.smooth_l1_loss(state_value, Variable(r)))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    # Clip gradient at 20,000
    torch.nn.utils.clip_grad_norm(model.parameters(), 20000)
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():

    # Parse arguments
    game, model_name, warm_start = parse_arguments()

    # initialize enviroment/model
    data = initialize(game, model_name, warm_start)
    env, model, optimizer, cuda, running_reward, running_rewards = data

    # Initialize constants
    max_episodes = 500000
    max_frames = 1000
    gamma = 0.95
    num_frames = 4

    for ep in range(len(running_rewards), max_episodes):
        # Anneal temperature from 1.8 down to 0.8 over 20,000 episodes
        model.temperature = max(0.8, 1.8 - 1.0 * ((ep) / 2.0e4))

        # Reset LSTM hidden units when episode begins
        if model_name == 'a2c-lstm':
            cx = Variable(torch.zeros(1, 100))
            hx = Variable(torch.zeros(1, 100))
            if cuda:
                cx = cx.cuda()
                hx = hx.cuda()

        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state]*num_frames)

        reward_sum = 0.0
        done = False
        while not done:
            for frame in range(max_frames):
                # Select action
                if model_name == 'a2c-lstm':
                    action, log_prob, state_value, (hx, cx) = select_action_lstm(model, state,
                                                                                 (hx, cx), cuda)
                else:
                    action, log_prob, state_value = select_action(model, state, cuda)

                model.saved_actions.append((log_prob, state_value))

                # Perform step
                next_state, reward, done, info = env.step(action)

                # Add reward to reward buffer
                model.rewards.append(reward)
                reward_sum += reward

                # Compute latest state
                next_state = preprocess_state(next_state)

                # Evict oldest frame add new frame to state
                next_state = np.stack([next_state]*num_frames)
                next_state[1:, :, :] = state[:-1, :, :]
                state = next_state

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

            # Update model
            backpropagate(model, optimizer, gamma, cuda)
            # Hidden values are carried forward
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        # Save model every 1000 episodes
        if (ep+1) % 1000 == 0:
            model_file = 'saved_models/{}_{}_ep_{}.p'.format(game, model_name, ep+1)

            with open(model_file, 'wb') as f:
                pickle.dump((model.cpu(), optimizer), f)

            if cuda:
                model = model.cuda()

            data_file = 'results/{}_{}.p'.format(game, model_name)

            with open(data_file, 'wb') as f:
                pickle.dump(running_rewards, f)


if __name__ == '__main__':
    main()
