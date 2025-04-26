# Snake Game with Reinforcement Learning

This project implements a **Reinforcement Learning (RL)** agent that learns to play the classic Snake game!  
The agent is trained using **Deep Q-Learning (DQN)** and progressively improves its performance over time.

## Project Structure

- `agent.py` — Contains the **DQN agent** logic, including memory replay and training.
- `game.py` — Defines the **game environment** and its rules.
- `helper.py` — Utility functions for data handling, plotting results, etc.
- `model.py` — Defines the **neural network architecture** for the agent.
- `snake_game.py` — Script to **run the game manually** (without RL).

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Akaanksh/MyCode.git
cd MyCode/1-Snake_Game-RL
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Train the Agent
```bash
python3 agent.py
```

## Reinforcement Learning Details

- **Algorithm:** Deep Q-Learning (DQN)
- **State Space:** Information about the snake’s environment (danger ahead, food location, direction, etc.)
- **Action Space:** Move left, right, or straight
- **Reward Scheme:**
  - Eating food: +5
  - Dying: -5

## Training Progress
![image](https://github.com/user-attachments/assets/45c67637-b1a2-4e18-aeb5-52541aaaef95)


## Demo
[Screencast from 04-26-2025 03:35:52 PM.webm](https://github.com/user-attachments/assets/e6678620-30c0-4e3b-99cf-e04e0a40de68)


## Future Improvements

- Implement Double DQN
- Add Prioritized Experience Replay
- Explore different network architectures (e.g., LSTM for temporal patterns)
- Hyperparameter tuning for better performance
