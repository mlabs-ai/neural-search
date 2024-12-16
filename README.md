## Code Structure

This repository contains the implementation of AlphaZero with modifications, along with support for various board games like Go and Gomoku. Below is the detailed structure of the project:

### Main Source Code (`alpha_zero` Directory)
Contains the main source code for the project.

### Core Modules (`core` Directory)
Contains essential modules for AlphaZero, including the MCTS search algorithm, self-play training pipeline, rating systems, etc.

- **`mcts_v1.py`**: Naive implementation of the MCTS search algorithm used by AlphaZero.
- **`mcts_v2.py`**: Faster implementation (~3x faster than `mcts_v1.py`), adapted from the Minigo project.
- **`pipeline.py`**: Core functions for the AlphaZero training pipeline, including self-play actor, learner, evaluator, supervised learning loop
- **`eval_dataset.py`**: Code to build an evaluation dataset using professional human-play games in SGF format.
- **`network.py`**: Implementation of the neural network class.
- **`quantum_net.py`**: Implementation of the neural search class.
- **`rating.py`**: Elo ratings computation code.
- **`replay.py`**: Uniform random replay buffer implementation.
- **`multi_game.py`**: Implementation of a game series among multiple agents
- **`testing_data.ipynb`**: Generation of the AlphaGo game dataset from sgf files

### Neural Search librarry (`lib` Directory)
Contains essential modules for neural search, including the search architecture, quantum search modeule
- **architecture.py** : Search module implementation
- **layers.py** : implementation of many wrapper functions like Unpack, residual etc
- **quantumsearch.py** : implementation of Quantum search

### Board Game Environments (`envs` Directory)
Modules for various board games, implemented in OpenAI Gym API:

- **`base.py`**: Basic board game environment.
- **`coords.py`**: Core logic for board games, adapted from the Minigo project.
- **`go_engine.py`**: Core logic and scoring functions for Go (adapted from Minigo).
- **`go.py`**: Implementation of the Go board game using `go_engine.py`.
- **`gomoku.py`**: Implementation of freestyle Gomoku board game (a.k.a five in a row).
- **`gui.py`**: Basic GUI program for board games.

### Utilities (`utils` Directory)
Contains helper modules like logging, data transformation, and SGF wrappers:

- **`sgf_wrapper.py`**: Code for reading and replaying Go game records in SGF format (adapted from Minigo).
- **`transformation.py`**: Functions for random rotation and mirroring of training samples.


### Training Programs
Driver programs for training agents:

- **`training_go.py`**: Initialize the training session for 9x9 Go board.
- **`training_go_jumbo.py`**: Initialize training for 19x19 Go board (requires powerful computational resources).
- **`training_gomoku.py`**: Initialize training for 13x13 Gomoku board.


