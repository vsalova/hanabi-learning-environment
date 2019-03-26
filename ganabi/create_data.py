import pickle
from collections import defaultdict
import sys
sys.path.insert(0,'..') 
import rl_env
from agents.rainbow_agent_rl import RainbowAgent
from agents.simple_agent import SimpleAgent

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RainbowAgent': RainbowAgent}
NUM_GAMES = 1000
NUM_PLAYERS = 2
DATA_PATH = "./data/discriminator_test.pkl"

def one_hot_vectorized_action(agent, obs):
  action = agent.act(obs)

  assert NUM_PLAYERS == 2 # FIXME: support more than 20 legal moves for 3+ player games
  one_hot_vector = [0]*20
  action_idx = obs['legal_moves_as_int'][obs['legal_moves'].index(action)]
  one_hot_vector[action_idx] = 1

  return one_hot_vector, action

environment = rl_env.make('Hanabi-Full', num_players=NUM_PLAYERS)
gameplay_data = defaultdict(list)
agent_config = {'players': NUM_PLAYERS,
                'num_moves': environment.num_moves(),
                'observation_size': environment.vectorized_observation_shape()[0]}
rainbow_agent = RainbowAgent(agent_config)
simple_agent = SimpleAgent(agent_config)

for _ in range(NUM_GAMES):
  observations = environment.reset()
  move_num = 0
  game_done = False
  
  while not game_done:
    for agent_id in range(NUM_PLAYERS):
      observation = observations['player_observations'][agent_id]
      rainbow_action_vec, rainbow_action = one_hot_vectorized_action(
              rainbow_agent, observation)
      # FUTURE NOTE: ensure that half of all datapoints are "same agent" if you
      # implement more agents than just rainbow and simple.
      simple_action_vec, _ = one_hot_vectorized_action(simple_agent, observation)
      gameplay_data[move_num].append((observation['vectorized'], 
          rainbow_action_vec, "rainbow"))
      gameplay_data[move_num].append((observation['vectorized'], 
          simple_action_vec, "simple"))
      move_num += 1
      
      # seems like a superfluous check but ok
      if observation['current_player'] == agent_id:
        assert rainbow_action is not None
        current_player_action = rainbow_action
      else:
        assert rainbow_action is None
      
      #print('Agent: {} action: {}'.format(observation['current_player'], current_player_action))
      observations, _, game_done, _ = environment.step(current_player_action)
      if game_done:
        break

pickle.dump(gameplay_data, open(DATA_PATH, "wb"))
