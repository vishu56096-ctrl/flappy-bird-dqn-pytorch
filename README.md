# Flappy Bird DQN Agent

A Deep Q-Network (DQN) agent written in PyTorch that learns to play Flappy Bird
through the Gymnasium `FlappyBird-v0` environment. The agent chooses between the
environment's available actions using learned Q-values and epsilon-greedy
exploration; there are no hand-coded rules for avoiding pipes.

## Features

- DQN policy network and target network
- Experience replay with randomly sampled mini-batches
- Epsilon-greedy exploration and decay
- Resume training from the latest saved network weights
- Save the best-reward model separately from the latest model
- Per-episode CSV history and a text log of new best rewards
- Hyperparameters loaded from `parameters.yaml`
- Optional rendered training, headless training, and rendered evaluation

## Project structure

```text
FlappyBird_RL/
|-- agent.py
|-- dqn.py
|-- experience_replay.py
|-- parameters.yaml
|-- requirements.txt
|-- README.md
|-- assets/
|   |-- demo.gif                 # Inline README preview
|   `-- Video Project 1.mp4      # Full gameplay/training demo
`-- runs/
    |-- flappybirdv0.pt           # Best-reward policy weights
    |-- flappybirdv0_latest.pt    # Latest policy weights for resuming
    |-- flappybirdv0.log          # New best rewards during training
    `-- flappybirdv0_episodes.csv # Per-episode metrics
```

Training creates and updates files in `runs/`. The `.gitignore` excludes
temporary files but keeps this agent's checkpoints and training history in Git.
A clone can inspect the CSV and play the best saved model immediately.

## Requirements and installation

Python 3.13.9 was used to verify this project. Create and activate a virtual
environment, then install the dependencies:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, open a new terminal with the
environment selected, or run the project's commands with the virtual
environment's Python executable directly.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependencies are `flappy-bird-gymnasium`, `gymnasium`, `PyYAML`, and
`torch`. If you already use Anaconda, activate the environment where these
packages are installed before running the commands below.

## Run the agent

Run commands from the project directory, where `agent.py` is located. The
`flappybirdv0` argument selects the hyperparameter set in `parameters.yaml`.

### Train and watch the agent learn

```bash
python agent.py flappybirdv0
```

Training is the default unless `--play` is specified. This command opens the
game window and trains for up to five minutes (with a maximum of 10,000
episodes). It prints the episode reward, exploration rate (`epsilon`), and
best reward. The latest policy is saved after every completed episode.

You can stop training with `Ctrl+C`; the latest network weights are saved so
you can continue later. Resuming loads the latest **network weights**, but
starts a new replay buffer and optimizer and resets epsilon to its initial
value.

### Train for a chosen number of episodes

```bash
python agent.py flappybirdv0 --episodes 5000 --no-render
```

This runs headlessly for up to 5,000 episodes, unless the reward threshold is
reached first. To train for a time limit without opening a game window:

```bash
python agent.py flappybirdv0 --minutes 5 --no-render
```

To explicitly show the game while training:

```bash
python agent.py flappybirdv0 --train --render --minutes 5
```

### Watch the best saved model play

```bash
python agent.py flappybirdv0 --play
```

This loads `runs/flappybirdv0.pt`, does not train, and shows five evaluation
episodes by default. Change the number of episodes with `--episodes`, for
example:

```bash
python agent.py flappybirdv0 --play --episodes 20
```

Use `--no-render` to evaluate without opening a game window.

## Training CSV

Training appends one row for each completed training episode to
`runs/flappybirdv0_episodes.csv`. The CSV can be opened in Excel, Google Sheets,
or another spreadsheet program to inspect or graph learning progress.

| Column | Meaning |
|---|---|
| `episode` | Episode number, continued from the existing CSV when training resumes. |
| `reward` | Total reward collected during that training episode. |
| `epsilon` | Exploration probability after the episode; lower values mean the agent more often chooses the action with the highest predicted Q-value. |
| `best_reward_so_far` | Highest reward recorded so far in the training log. |

Evaluation with `--play` does not add rows to this CSV. The separate
`runs/flappybirdv0.log` records only new best rewards, while the `.pt` files
contain model weights rather than episode metrics.

## Hyperparameters

Values below are read from `parameters.yaml`:

| Parameter | Value | Description |
|---|---:|---|
| Environment | `FlappyBird-v0` | Gymnasium environment ID. |
| Learning rate (`alpha`) | `0.001` | Adam optimizer learning rate. |
| Discount factor (`gamma`) | `0.99` | Weight given to future rewards. |
| Replay memory capacity | `100,000` | Maximum transitions retained for sampling. |
| Mini-batch size | `32` | Transitions sampled for each gradient update. |
| Target network sync interval | `10` steps | Copy policy weights to the target network every 10 environment steps. |
| Initial epsilon | `1.0` | Start by exploring with random actions. |
| Minimum epsilon | `0.05` | Minimum exploration probability. |
| Epsilon decay | `0.9995` | Multiplier applied to epsilon after each episode. |
| Reward threshold | `1,000` | Episode stopping threshold configured by the script. |
| Hidden layer width | `256` | Number of units in the DQN's hidden layer. |

## Results

Results from the current saved files (not a guarantee of future performance):

- Training history: **17,346 episodes** recorded in the CSV.
- Best reward recorded during training: **9.1**.
- Best saved model evaluation: **2.69 average reward over 20 episodes**, evaluated
  without rendering using seeds 2026 through 2045.
- The highest reward in that 20-episode evaluation was **6.4**.

Training rewards vary between episodes. Use the CSV to review the full training
history and run more episodes if you want to compare results.

## Demo

![Flappy Bird DQN demo](assets/demo.gif)

[Open or download the full video](assets/Video%20Project%201.mp4)
(about 26 MB).

For more information about the project and its learning process, see the
[Flappy Bird DQN learning document (PDF)](assets/Flappy_Bird_DQN_Project_Report.pdf).

## Author

Vishal Kumar - [GitHub](https://github.com/vishu56096-ctrl)
