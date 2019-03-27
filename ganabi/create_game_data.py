import pickle
from collections import defaultdict
import sys
sys.path.insert(0,'..')
import rl_env
from agents.rainbow_agent_rl import RainbowAgent
from agents.simple_agent import SimpleAgent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("games", type=int)
parser.add_argument("players", type=int)
parser.add_argument("data_path", type=str)
parser.add_argument("agent_classes", nargs="+")
args = parser.parse_args()

AGENT_CLASSES = {}
NUM_GAMES = args.games
NUM_PLAYERS = args.players
DATA_PATH = args.data_path

for agent in args.agent_classes:
    if agent == "SimpleAgent":
        AGENT_CLASSES[agent] = SimpleAgent
        data[agent] = []
    elif agent == "RainbowAgent":
        AGENT_CLASSES[agent] = RainbowAgent
        data[agent] = []

environment = rl_env.make('Hanabi-Full', num_players=NUM_PLAYERS)
agent_config = {'players': NUM_PLAYERS,
                'num_moves': environment.num_moves(),
                'observation_size': environment.vectorized_observation_shape()[0]}
rainbow_agent = RainbowAgent(agent_config)
simple_agent = SimpleAgent(agent_config)

for game_num in range(NUM_GAMES):
  observations = environment.reset()
  game_done = False
  data[agent_type].append([[],[]])

  while not game_done:
    for agent_id in range(NUM_PLAYERS):
        observation = observations['player_observations'][agent_id]

        for agent_type in AGENT_CLASSES:
            action = agent.act(observation)
            data[agent_type][game_num][0].append(observation)
            data[agent_type][game_num][1].append(action)

        observations, _, game_done, _ = environment.step(current_player_action)

        if game_done:
            break

pickle.dump(data, open(DATA_PATH, "wb"))
