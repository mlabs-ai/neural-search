import os
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import random


import os
from typing import Any, Text, Callable, Mapping, Iterable, Tuple
from pathlib import Path
from collections import OrderedDict, deque

import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

torch.autograd.set_detect_anomaly(True)

import numpy as np
import matplotlib.pyplot as plt

# from alpha_zero.core.mcts_v1 import Node, parallel_uct_search, uct_search

from alpha_zero.core.mcts_v2 import Node, parallel_uct_search, uct_search

from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.core.rating import EloRating
from alpha_zero.core.replay import UniformReplay, Transition
from alpha_zero.utils.csv_writer import CsvWriter
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


def strip_prefix(state_dict, prefix="_orig_mod."):
    """
    Strip a given prefix from the keys in the state_dict.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # Remove the prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


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
def run_pair_game_series(
    seed: int,
    network_1: torch.nn.Module,
    network_2: torch.nn.Module,
    name_1 : str,
    name_2 : str,
    device: torch.device,
    env: BoardGameEnv,
    num_games: int,
    num_simulations: int,
    num_parallel: int,
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
    network_1 = network_1.to(device=device)
    network_2 = network_2.to(device=device)
    disable_auto_grad(network_1)
    disable_auto_grad(network_2)
  # Validate checkpoints
    if not (load_ckpt_1 and os.path.exists(load_ckpt_1)):
        raise ValueError(f"ResNet checkpoint path is invalid: {load_ckpt_1}")
    if not (load_ckpt_2 and os.path.exists(load_ckpt_2)):
        raise ValueError(f"QuantumNet checkpoint path is invalid: {load_ckpt_2}")



    loaded_state_1 = torch.load(load_ckpt_1, map_location=device)
    # Strip `_orig_mod.` prefix
    cleaned_state_dict_1 = strip_prefix(loaded_state_1['network'])
    network_1.load_state_dict(cleaned_state_dict_1)
    logger.info(f"network 1 model loaded from checkpoint: {load_ckpt_1}")


    loaded_state_2 = torch.load(load_ckpt_2, map_location=device)
    cleaned_state_dict_2 = strip_prefix(loaded_state_2['network'])
    network_2.load_state_dict(cleaned_state_dict_2)
    logger.info(f"network 2 model loaded from checkpoint: {load_ckpt_2}")


    network_1.eval()
    network_2.eval()

    # Elo ratings
    black_elo = EloRating(rating=default_rating)
    white_elo = EloRating(rating=default_rating)

    black_player = create_mcts_player(
                network=network_1,
                device=device,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=False,
                deterministic=True,
            )
    white_player = create_mcts_player(
                network=network_2,
                device=device,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=False,
                deterministic=True,
            )

    # Results tracking
    black_wins, white_wins, draws = 0, 0, 0

    for game_idx in range(num_games):


        logger.info(f"Playing game {game_idx + 1}/{num_games}")

        # Play the game
        game_stats = play_one_game(
            env,
            black_player,
            white_player,
            black_elo,
            white_elo,

            c_puct_base,
            c_puct_init,
            logger,
        )

        # Process results
        if game_stats["game_result"] == "Black wins":
            black_wins+=1
        elif game_stats["game_result"] == "White wins":
            white_wins+=1
        else:
            draws += 1

        logger.info(
            f"Game {game_idx + 1} result: {game_stats['game_result']} - "
            f"{name_1}: {black_elo.rating}, {name_2}: {white_elo.rating}"
        )


    # Final logging
    logger.info(f"Game result: {game_stats['game_result']}")
    logger.info(f"Elo rating : {name_1}: {black_elo.rating}, {name_2}: {white_elo.rating}")
    logger.info(f"Game length: {game_stats['game_length']} moves")


    # Visualize Elo ratings
    visualize_elo_ratings(black_elo.rating, white_elo.rating, logs_dir)

@torch.no_grad()
def run_tournament(
    seed: int,
    agents: dict,
    env: BoardGameEnv,
    device: torch.device,
    num_games: int,
    num_simulations: int,
    num_parallel: int,
    c_puct_base: float,
    c_puct_init: float,
    default_rating: float,
    log_level: str,
    logs_dir: str = "./logs",

):
    """
    Run a tournament where all agents play against each other,
    updating Elo ratings after each match.


    """

    set_seed(seed)
    logger = create_logger(log_level)
    # Initialize Elo ratings
    for agent_name, agent in agents.items():
        if "elo_rating" not in agent:
            agent["elo_rating"] = default_rating
    agents_list = list(agents.keys())



    # Play multiple games
    for game in range(num_games):
        #randomly select two agents
        agent_1_name, agent_2_name = random.sample(agents_list, 2)
        logger.info(f"Starting match between {agent_1_name} and {agent_2_name}")
        logger.info(f"Playing game {game + 1}/{num_games}")

        network_1 = agents[agent_1_name]["network"]
        network_2 = agents[agent_2_name]["network"]

        network_1 = network_1.to(device)
        network_2 = network_2.to(device)

        network_1.eval()
        network_2.eval()

        black_elo = EloRating(rating=agents[agent_1_name]["elo_rating"])
        white_elo = EloRating(rating=agents[agent_2_name]["elo_rating"])


        loaded_state_1 = torch.load(agents[agent_1_name]["checkpoint"], map_location=device)
        cleaned_state_dict_1 = strip_prefix(loaded_state_1['network'])   # Strip `_orig_mod.` prefix
        network_1.load_state_dict(cleaned_state_dict_1)
        logger.info(f"network 1 model loaded from checkpoint: {agents[agent_1_name]['checkpoint']}")


        loaded_state_2 = torch.load(agents[agent_2_name]["checkpoint"], map_location=device)
        cleaned_state_dict_2 = strip_prefix(loaded_state_2['network'])
        network_2.load_state_dict(cleaned_state_dict_2)
        logger.info(f"network 2 model loaded from checkpoint: {agents[agent_2_name]['checkpoint']}")

         # Create MCTS players
        black_player = create_mcts_player(
            network=network_1,
            device=device,
            num_simulations=num_simulations,
            num_parallel=num_parallel,
            root_noise=False,
            deterministic=False,
        )
        white_player = create_mcts_player(
            network=network_2,
            device=device,
            num_simulations=num_simulations,
            num_parallel=num_parallel,
            root_noise=False,
            deterministic=False,
        )

        # Randomize roles
        if np.random.rand() < 0.5:
            logger.info(f"Game {game + 1}: {agent_1_name} is Black, {agent_2_name} is White")
            current_black_player, current_white_player = black_player, white_player
            current_black_name, current_white_name = agent_1_name, agent_2_name
            current_black_elo, current_white_elo = black_elo, white_elo
        else:
            logger.info(f"Game {game + 1}: {agent_2_name} is Black, {agent_1_name} is White")
            current_black_player, current_white_player = white_player, black_player
            current_black_name, current_white_name = agent_2_name, agent_1_name
            current_black_elo, current_white_elo = white_elo, black_elo

        game_stats = play_one_game(
            env = env,
            black_player = current_black_player,
            white_player = current_white_player,
            black_elo = current_black_elo,
            white_elo = current_white_elo,
            c_puct_base = c_puct_base,
            c_puct_init = c_puct_init,
            logger = logger,
        )

        # Track results
        if game_stats["game_result"] == "Black wins":

            agents[current_black_name]["wins"] += 1
            agents[current_white_name]["lost"] += 1

        elif game_stats["game_result"] == "White wins":

            agents[current_white_name]["wins"] += 1
            agents[current_black_name]["lost"] += 1


        # Update Elo ratings
        agents[current_black_name]["elo_rating"] = current_black_elo.rating
        agents[current_white_name]["elo_rating"] = current_white_elo.rating


    # Log results
    for agent_name, agent in agents.items():
        logger.info(
            f"Agent {agent_name} -> Wins: {agent['wins']}, Losses: {agent['lost']}, Elo Rating: {agent['elo_rating']:.2f}"
        )

    visualize_all_elo_ratings(agents)



@torch.no_grad()
def play_one_game(
    env,
    black_player,
    white_player,
    black_elo,
    white_elo,
    c_puct_base,
    c_puct_init,
    logger,
) -> Mapping[str, Any]:
    """
    Simulate a single game between ResNet (Black) and QuantumNet (White).
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
        logger.debug(f"Player {env.to_play} chose move: {move}")
        # logger.debug(f"State before move: {env.render()}")

        _, _, done, _ = env.step(move)
        # logger.debug(f"State after move: {env.render()}")

    if env.winner == env.black_player:
        game_result = "Black wins"
        winner, loser = black_elo, white_elo
    elif env.winner == env.white_player:
        game_result = "White wins"
        winner, loser = white_elo, black_elo
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
        "black_elo_rating": black_elo.rating,
        "white_elo_rating": white_elo.rating,
    }

    return stats



def visualize_elo_ratings(black_elo: float, white_elo: float,
                          black_name: str, white_name: str,
                          logs_dir: str) -> None:
    """
    Create a bar chart of Elo ratings.
    """


    models = [black_name, white_name]
    ratings = [black_elo, white_elo]

    plt.figure(figsize=(8, 6))
    plt.bar(models, ratings, color=["blue", "green"])
    plt.xlabel("Models")
    plt.ylabel("Elo Rating")
    plt.title("Elo Ratings Comparison")

    # Add Elo values above the bars
    for i, rating in enumerate(ratings):
        plt.text(i, rating + 5, f"{rating:.2f}", ha="center", fontsize=10)

    # Save and show plot
    output_path = os.path.join(logs_dir, "elo_ratings_comparison.png")
    plt.savefig(output_path)
    plt.show()


def visualize_all_elo_ratings(agents: dict) -> None:
    """
    Create a single bar chart of Elo ratings for all agents.

    """
    agent_names = list(agents.keys())
    elo_ratings = [agent["elo_rating"] for agent in agents.values()]
    # Use a colormap for distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(agent_names)))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(agent_names, elo_ratings, color= colors)
    plt.xlabel("Agents")
    plt.ylabel("Elo Rating")
    plt.title("Final Elo Ratings for All Agents")
    plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability

    # Add Elo values above the bars
    for i, rating in enumerate(elo_ratings):
        plt.text(i, rating + 5, f"{rating:.2f}", ha="center", fontsize=10)


    plt.show()

