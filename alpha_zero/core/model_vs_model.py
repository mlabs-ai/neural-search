
import os
import shutil
from typing import Any, Text, Callable, Mapping, Iterable, Tuple
import time
from pathlib import Path
from collections import OrderedDict, deque
import queue
import multiprocessing as mp
import threading
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

torch.autograd.set_detect_anomaly(True)

import numpy as np
from copy import copy, deepcopy

# from alpha_zero.core.mcts_v1 import Node, parallel_uct_search, uct_search

from alpha_zero.core.mcts_v2 import Node, parallel_uct_search, uct_search

from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.core.eval_dataset import build_eval_dataset
from alpha_zero.core.rating import EloRating
from alpha_zero.core.replay import UniformReplay, Transition
from alpha_zero.utils.csv_writer import CsvWriter
from alpha_zero.utils.transformation import apply_random_transformation
from alpha_zero.utils.util import Timer, create_logger, get_time_stamp


# =================================================================
# Helper functions
# =================================================================



def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_create_dir(dir) -> None:
    if dir is not None and dir != '' and not os.path.exists(dir):
        p = Path(dir)
        p.mkdir(parents=True, exist_ok=False)


def save_to_file(obj: Any, file_name: str) -> None:
    pickle.dump(obj, open(file_name, 'wb'))


def load_from_file(file_name: str) -> Any:
    return pickle.load(open(file_name, 'rb'))


def round_it(v, places=4) -> float:
    return round(v, places)


def _encode_bytes(in_str) -> Any:
    return str(in_str).encode('utf-8')


def _decode_bytes(b) -> str:
    return b.decode('utf-8')


def create_mcts_player(
    network: torch.nn.Module,
    device: torch.device,
    num_simulations: int,
    num_parallel: int,
    root_noise: bool = False,
    deterministic: bool = False,
) -> Callable[[BoardGameEnv, Node, float, float, bool], Tuple[int, np.ndarray, float, float, Node]]:

    @torch.no_grad()
    def eval_position(
        state: np.ndarray,
        batched: bool = False,
    ) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
        """Give a game state tensor, returns the action probabilities
        and estimated state value from current player's perspective."""

        if not batched:
            state = state[None, ...]

        state = torch.from_numpy(state).to(dtype=torch.float32, device=device, non_blocking=True)
        pi_logits, v = network(state)

        pi_logits = torch.detach(pi_logits)
        v = torch.detach(v)

        pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
        v = v.cpu().numpy()

        B, *_ = state.shape

        v = np.squeeze(v, axis=1)
        v = v.tolist()  # To list

        # Unpack the batched array into a list of NumPy arrays
        pi = [pi[i] for i in range(B)]

        if not batched:
            pi = pi[0]
            v = v[0]

        return pi, v

    def act(
        env: BoardGameEnv,
        root_node: Node,
        c_puct_base: float,
        c_puct_init: float,
        warm_up: bool = False,
    ) -> Tuple[int, np.ndarray, float, float, Node]:
        if num_parallel > 1:
            return parallel_uct_search(
                env=env,
                eval_func=eval_position,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=root_noise,
                warm_up=warm_up,
                deterministic=deterministic,
            )
        else:
            return uct_search(
                env=env,
                eval_func=eval_position,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations,
                root_noise=root_noise,
                warm_up=warm_up,
                deterministic=deterministic,
            )

    return act

##black player = resnet alphago, white player = quantum_search net





@torch.no_grad()
def play_one_game(
    env,
    black_player,
    white_player,
    black_elo,
    white_elo,
    c_puct_base,
    c_puct_init,
) -> Mapping[Text, Any]:
    _ = env.reset()
    mcts_player = None
    done = False
    num_passes = 0

    while not done:
        if env.to_play == env.black_player:
            mcts_player = black_player
        else:
            mcts_player = white_player
        move, *_ = mcts_player(
            env=env,
            root_node=None,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            warm_up=False,
        )

        _, _, done, _ = env.step(move)

        if env.has_pass_move and move == env.pass_move:
            num_passes += 1

    stats = {
        'game_length': env.steps,
        'game_result': env.get_result_string(),
    }

    # Update elo rating for both players
    if env.winner is not None:
        if env.winner == env.black_player:
            winner, loser = black_elo, white_elo
        elif env.winner == env.white_player:
            winner, loser = white_elo, black_elo

        winner.update_rating(loser.rating, 1)
        loser.update_rating(winner.rating, 0)

    stats['black_elo_rating'] = black_elo.rating
    stats['white_elo_rating'] = white_elo.rating
    return stats

