import matplotlib.pyplot as plt
import pickle
import argparse

human_scores = {'Asteroids-v0': 13157,
                'Boxing-v0': 4.3,
                'Breakout-v0': 31.8,
                'Pong-v0': 9.3,
                'SpaceInvaders-v0': 1652,
                }

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-g', action='store', dest='game')
parser.add_argument('-m', action='store', dest='model')

args = parser.parse_args()
game = args.game
model_name = args.model
data_file = 'results/{}_{}.p'.format(game, model_name)

with open(data_file, 'rb') as f:
    data = pickle.load(f)

human = [human_scores[game] for _ in range(len(data))]
out_file = 'results/{}_{}_plot.png'.format(game, model_name)

plt.plot(data, label=model_name)
plt.plot(human, label='Human Expert')
plt.title('{} {} Rewards versus Episodes'.format(game, model_name))
plt.xlabel('Episodes')
plt.ylabel('Average Rewards')
plt.legend()
plt.tight_layout()
plt.savefig(out_file)
plt.show()
