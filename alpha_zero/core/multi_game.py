import os
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt



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
import matplotlib.pyplot as plt

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



@torch.no_grad()
def run_evaluator_loop(
    seed: int,
    network_1: torch.nn.Module,
    network_2: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    num_simulations: int,
    num_parallel: int,
    num_games: int,
    c_puct_base: float,
    c_puct_init: float,
    default_rating: float,
    logs_dir: str,
    load_ckpt_1: str,  # ResNet checkpoint
    load_ckpt_2: str,  # QuantumNet checkpoint
    log_level: str,
) -> None:
    """
    Evaluate multiple games between ResNet and QuantumNet, alternating roles.
    """
    set_seed(seed)
    logger = create_logger(log_level)

    # Load networks
    resnet_network = network_1.to(device=device)
    quantum_net = network_2.to(device=device)
    disable_auto_grad(resnet_network)
    disable_auto_grad(quantum_net)

    if load_ckpt_1 and os.path.exists(load_ckpt_1):
        loaded_state_1 = torch.load(load_ckpt_1, map_location=device)
        resnet_network.load_state_dict(loaded_state_1['network'])
        logger.info(f"ResNet model loaded from checkpoint: {load_ckpt_1}")

    if load_ckpt_2 and os.path.exists(load_ckpt_2):
        loaded_state_2 = torch.load(load_ckpt_2, map_location=device)
        quantum_net.load_state_dict(loaded_state_2['network'])
        logger.info(f"QuantumNet model loaded from checkpoint: {load_ckpt_2}")

    resnet_network.eval()
    quantum_net.eval()

    # Elo ratings
    resnet_elo = EloRating(rating=default_rating)
    quantum_elo = EloRating(rating=default_rating)
    resnet_player = create_mcts_player(
                network=resnet_network,
                device=device,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=False,
                deterministic=True,
            )
    quantum_player = create_mcts_player(
                network=quantum_net,
                device=device,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=False,
                deterministic=True,
            )

    # Results tracking
    resnet_wins, quantum_net_wins, draws = 0, 0, 0

    for game_idx in range(num_games):
    # Alternate Black and White roles
        r = random.choice([0,1])
        if r == 0:
            black_player = resnet_player
            white_player = quantum_player
            black_is_resnet = True
        else:
            black_player = create_mcts_player(
                network=quantum_net,
                device=device,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=False,
                deterministic=True,
            )
            white_player = create_mcts_player(
                network=resnet_network,
                device=device,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=False,
                deterministic=True,
            )
            black_is_resnet = False

        logger.info(f"Playing game {game_idx + 1}/{num_games}")

        # Play the game
        game_stats = play_one_game(
            env,
            black_player,
            white_player,
            resnet_elo,
            quantum_elo,
            c_puct_base,
            c_puct_init,
            logger,
        )

        # Process results
        if game_stats["game_result"] == "Black wins":
            if black_is_resnet:
                resnet_wins += 1
            else:
                quantum_net_wins += 1
        elif game_stats["game_result"] == "White wins":
            if not black_is_resnet:
                resnet_wins += 1
            else:
                quantum_net_wins += 1
        else:
            draws += 1

        logger.info(
            f"Game {game_idx + 1} result: {game_stats['game_result']} - "
            f"ResNet Elo: {resnet_elo.rating}, QuantumNet Elo: {quantum_elo.rating}"
        )

    # Final results
    logger.info(f"Final Results after {num_games} games:")
    logger.info(f"ResNet Wins: {resnet_wins}, QuantumNet Wins: {quantum_net_wins}, Draws: {draws}")
    logger.info(f"Final Elo Ratings - ResNet: {resnet_elo.rating}, QuantumNet: {quantum_elo.rating}")

    # Visualize Elo ratings
    visualize_elo_ratings(resnet_elo.rating, quantum_elo.rating, logs_dir)


@torch.no_grad()
def play_one_game(
    env,
    black_player,
    white_player,
    resnet_elo,
    quantum_elo,
    c_puct_base,
    c_puct_init,
    logger,
    black_is_resnet: bool,
) -> Mapping[str, Any]:
    """
    Simulate a single game between two players.
    """
    env.reset()
    done = False

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
        # logger.debug(f"Player {env.to_play} chose move: {move}")
        # logger.debug(f"State before move: {env.render()}")

        _, _, done, _ = env.step(move)
        # logger.debug(f"State after move: {env.render()}")

     # Determine winner
    if env.winner == env.black_player:
        game_result = "Black wins"
        if black_is_resnet:
            winner, loser = resnet_elo, quantum_elo
        else:
            winner, loser = quantum_elo, resnet_elo
    elif env.winner == env.white_player:
        game_result = "White wins"
        if not black_is_resnet:
            winner, loser = resnet_elo, quantum_elo
        else:
            winner, loser = quantum_elo, resnet_elo
    else:
        game_result = "Draw"
        winner, loser = None, None

    # Update Elo ratings
    if winner and loser:
        winner.update_rating(loser.rating, 1)
        loser.update_rating(winner.rating, 0)

    stats = {
        "game_length": env.steps,
        "game_result": game_result,
        "resnet_elo_rating": resnet_elo.rating,
        "quantum_elo_rating": quantum_elo.rating,
    }

    return stats


def visualize_elo_ratings(resnet_elo: float, quantum_elo: float, logs_dir: str) -> None:
    """
    Create a bar chart of Elo ratings.
    """
    models = ["ResNet", "QuantumNet"]
    ratings = [resnet_elo, quantum_elo]

    plt.figure(figsize=(8, 6))
    plt.bar(models, ratings, color=["blue", "green"])
    plt.xlabel("Models")
    plt.ylabel("Elo Rating")
    plt.title("Elo Ratings Comparison")
    plt.savefig(os.path.join(logs_dir, "elo_ratings_comparison.png"))
    plt.show()
