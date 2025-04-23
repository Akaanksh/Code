from rl_env import Game2048Env

env = Game2048Env()
state = env.reset()
env.render()

for i in range(10):
    action = i % 4  # just cycle through directions
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    env.render()
    if done:
        print("Game Over!")
        break