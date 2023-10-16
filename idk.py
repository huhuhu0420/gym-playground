import sys
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def train():
    # Parallel environments
    # vec_env = make_vec_env("ALE/ElevatorAction-v5", n_envs=4)
    # vec_env2 = make_vec_env("ALE/Centipede-v5", n_envs=4)

    # venv = gym.make("ALE/Centipede-v5")
    # venv2 = gym.make("ALE/IceHockey-v5")

    venv = make_vec_env("ALE/Centipede-v5", n_envs=4)
    venv2 = make_vec_env("ALE/IceHockey-v5", n_envs=4)

    model = PPO(
            policy="MlpPolicy",
            env=venv,
            device="cuda",
            learning_rate=lambda f: f * 2.5e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
        )

    timesteps = 50000
    i = 0
    while i < 10:
        # env1
        if i != 0:
            model = PPO.load("idk", env=venv)
        model.learn(total_timesteps=timesteps)
        model.save("idk")

        # env2
        model = PPO.load("idk", env=venv2)
        model.learn(total_timesteps=timesteps)
        model.save("idk")

        i += 1

def play():
    venv = gym.make("ALE/IceHockey-v5", render_mode='human')
    venv2 = gym.make("ALE/Centipede-v5", render_mode='human')

    model = PPO.load("idk", env=venv)
    venv = model.get_env()
    obs = venv.reset()
    model2 = PPO.load("idk", env=venv2)
    venv2 = model2.get_env()
    obs2 = venv2.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = venv.step(action)
        if done:
            venv.close()

        action, _states = model2.predict(obs2)
        obs2, reward, done, info = venv2.step(action)
        if done:
            venv2.close()
    venv.close()
    venv2.close()

def main ():
    args = sys.argv[1:][0]
    if args == "play":
        play()
    elif args == "train":
        train()

main()
