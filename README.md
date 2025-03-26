## Code Structure

This repository contains the implementation of Neural Search and AlphaGo. Below is the detailed structure of the project:

### Main Source Code (`alpha_zero` Directory)
Contains the main source code for the project.

### Core Modules (`core` Directory)
Contains essential modules for AlphaZero, including the MCTS search algorithm, self-play training pipeline, rating systems, etc.

- **`quantum_net.py`**: **Inplementation of the ResNet based Quantum Neural search**
- **`mcts_v1.py`**: Naive implementation of the MCTS search algorithm used by AlphaZero.
- **`mcts_v2.py`**: Faster implementation (~3x faster than `mcts_v1.py`), adapted from the Minigo project.
- **`pipeline.py`**: Core functions for the AlphaZero training pipeline, including self-play actor, learner, evaluator, supervised learning loop
- **`eval_dataset.py`**: Code to build an evaluation dataset using professional human-play games in SGF format.
- **`network.py`**: Implementation of the neural network class.
- **`quantum_net.py`**: Implementation of the neural search class.
- **`rating.py`**: Elo ratings computation code.
- **`replay.py`**: Uniform random replay buffer implementation.
- **`Alphago_series.py`**: Implementation of a game series among multiple agents (supervised search agents)
- **`Alphago_series_RL.py`**: Implementation of a game series among multiple agents (agents trained with RL)
- **`testing_data.ipynb`**: Generation of the AlphaGo game dataset from sgf files

### Neural Search Library (`lib` Directory)

This directory contains the core modules for neural search, including the implementation of the search architecture and quantum search module:

- **`architecture.py`** :Implements the main search module.
- **`layers.py`** : Contains wrapper functions such as `Unpack`, `Residual` layer implementations.
- **`quantumsearch.py`** :Provides the implementation of the Quantum Search module.


### Board Game Environments (`envs` Directory)
Modules for various board games, implemented in OpenAI Gym API:

- **`base.py`**: Basic board game environment.
- **`coords.py`**: Core logic for board games, adapted from the Minigo project.
- **`go_engine.py`**: Core logic and scoring functions for Go (adapted from Minigo).
- **`go.py`**: Implementation of the Go board game using `go_engine.py`.
- **`gui.py`**: Basic GUI program for board games.

### Utilities (`utils` Directory)
Contains helper modules like logging, data transformation, and SGF wrappers:

- **`sgf_wrapper.py`**: Code for reading and replaying Go game records in SGF format (adapted from Minigo).
- **`transformation.py`**: Functions for random rotation and mirroring of training samples.

### Neural Search Application
- **maze_entropy_quantum.ipynb**: Implements neural search with reinforcement learning (RL).
- **Quantum_mnist**: Implements neural search for classifying MNIST data.

### Training Programs
Driver programs for training agents:

- **`training_go.py`**: Launches training for the baseline model on a 5x5 Go board.
- **`training_search.py`**:  Launches training for search-based models on a 5x5 Go board.


## Reference code
- https://github.com/michaelnny/alpha_zero
