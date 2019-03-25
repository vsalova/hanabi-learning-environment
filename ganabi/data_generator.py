import rl_env
import pickle
from collections import defaultdict
from ..agents.rainbow_agent_rl import RainbowAgent
from ..agents.simple_agent import SimpleAgent

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RainbowAgent': RainbowAgent}
NUM_GAMES = 10
NUM_PLAYERS = 2
DATA_PATH = "./models/discriminator_test.pkl"


environment = rl_env.make('Hanabi-Full', num_players=NUM_PLAYERS)
gameplay_data = defaultdict(list)
agent_config = {'players': NUM_PLAYERS,
                'num_moves': self.environment.num_moves(),
                'observation_size': self.environment.vectorized_observation_shape()[0]}
rainbow_agent = RainbowAgent(agent_config)
simple_agent = SimpleAgent(agent_config)
  
for _ in range(NUM_GAMES):
  observations = self.environment.reset()
  move_num = 0
  game_done = False
  
  while not game_done:
    for agent_id in range(NUM_PLAYERS):
      observation = observations['player_observations'][agent_id]
      
      rainbow_action = rainbow_agent.act(observation)
      simple_action = simple_agent.act(observation)
      
      gameplay_data[move_num].append((observation, rainbow_action, "rainbow"))
      gameplay_data[move_num].append((observation, simple_action, "simple"))
      
      # seems like a superfluous check but ok
      if observation['current_player'] == agent_id:
        assert rainbow_action is not None
        current_player_action = rainbow_action
      else:
        assert rainbow_action is None
      
      import time; time.sleep(.100)
      print('Agent: {} action: {}'.format(observation['current_player'], current_player_action))
      observations, _, game_done, _ = self.environment.step(current_player_action)

pickle.dump(gameplay_data, open(DATA_PATH, "wb"))