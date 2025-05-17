# Deep Reinforcement Learning Agents in Pacman and Gridworld

This project implements reinforcement learning and planning agents in two environments: a simplified Gridworld and the classic Pacman game. It is based on the UC Berkeley CS188: Introduction to Artificial Intelligence course. It features tabular Q-Learning, Value Iteration, and Deep Q-Learning (DQN) with custom neural networks and visualization tools.

## Project Highlights

- Deep Q-Learning agent using a custom NN and experience replay
- Tabular Q-Learning and Value Iteration agents
- Feature extractors for high-level state representation
- Pacman and Gridworld simulation environments
- Robotic crawler simulation (GUI-based)
- Autograder and test suite for evaluation

## File Structure

| File | Description |
|------|-------------|
| `deep_q_learning_agents.py` | Deep Q-Network (DQN) implementation for Pacman |
| `learning_agents.py` | Base Q-Learning and Value Iteration agents |
| `feature_extractors.py` | Feature extraction for state representations |
| `model.py` | Deep Q-Network model structure |
| `nn.py` | Low-level neural network primitives (layers, loss, updates) |
| `gridworld.py` | Gridworld MDP simulator |
| `pacman.py` | Pacman game engine |
| `game.py` | Shared core logic for agents and actions |
| `ghost_agents.py` | Ghost behavior classes |
| `keyboard_agents.py` | Keyboard-controlled Pacman |
| `crawler.py` | Robotic crawler simulation |
| `environment.py` | Environment abstraction for RL agents |
| `layout.py` | Maze parsing and map layout tools |
| `graphics_display.py` | GUI for Pacman |
| `graphics_gridworld_display.py` | GUI for Gridworld |
| `graphics_crawler_display.py` | GUI for crawler agent |
| `graphics_utils.py` | Low-level drawing primitives (Tkinter) |
| `text_display.py` | Console Pacman display |
| `text_gridworld_display.py` | Console Gridworld display |
| `util.py` | Utility code: counters, sampling, distances |
| `autograder.py` | Command-line autograder for evaluating agents |
| `grading.py` | Test framework and output formatter |
| `test_classes.py` | General testing logic |
| `reinforcement_test_classes.py` | Test suite for RL agents |
| `test_parser.py` | Test file parser |
| `project_params.py` | Autograder configuration constants |
| `backend.py` | Backend stats and visualization for perceptron demo |

## How to Use

### Gridworld (Value Iteration)
```bash
python gridworld.py -a value
```

### Gridworld (Q-Learning)
```bash
python gridworld.py -a q -n 100 -e 0.1 -g 0.9
```

### Pacman (Keyboard Control)
```bash
python pacman.py -p KeyboardAgent
```

### Pacman (Deep Q-Learning)
Make sure to call the agent from `deep_q_learning_agents.py`:
```bash
python pacman.py -p PacmanDeepQAgent
```
### Crawler Simulation
```bash
python crawler.py
```

## Autograder

Run all questions:
```bash
python autograder.py
```

Run a specific question:
```bash
python autograder.py -q q4
```

Mute output:
```bash
python autograder.py --mute
```
## Agent Parameters

Example usage:
```bash
python pacman.py -p PacmanQAgent -a alpha=0.2, epsilon=0.05, gamma=0.8, numTraining=200
```

| Parameter | Meaning |
|----------|---------|
| `alpha` | Learning rate |
| `epsilon` | Exploration probability |
| `gamma` | Discount factor |
| `numTraining` | Number of training episodes |

## Notes

- Neural networks are implemented from scratch using NumPy (no PyTorch/TensorFlow)
- GUI uses Tkinter
- Supports both manual and automated evaluation
