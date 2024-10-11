"""
try monte-carlo tree search
"""
from pdb import set_trace as bp
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.envs.base import BoardGameEnv
from torch import nn
from alpha_zero.core.pipeline import create_mcts_player
import torch




if __name__ == '__main__':
    env = BoardGameEnv(board_size = 4, num_stack = 2)
    num_actions = env.action_space.n 
    input_shape = env.observation_space.shape

    network = AlphaZeroNet(input_shape = input_shape, num_actions = num_actions, 
                           num_res_block = 1, num_filters = 32, 
                           num_fc_units = 1
                           )  
    player = create_mcts_player(network = network, 
                                device = None, 
                                num_simulations = 1, 
                                num_parallel = 1, 
                                root_noise = False, 
                                deterministic = False)

    bp()
