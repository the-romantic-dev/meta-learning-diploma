import torch
import numpy as np
from gymnasium.spaces import Discrete, Box
import gymnasium as gym

from gymnasium import wrappers as w
import torch.nn as nn
from typing import List

from hebbian_update import hebbian_update
from nn_models import CNN_heb, MLP_heb
from wrappers import FireEpisodicLifeEnv, ScaledFloatFrame


def _weights_init(m, init_weights):
    if isinstance(m, torch.nn.Linear):
        if init_weights == 'xa_uni':
            torch.nn.init.xavier_uniform(m.weight.data, 0.3)
        elif init_weights == 'sparse':
            torch.nn.init.sparse_(m.weight.data, 0.8)
        elif init_weights == 'uni':
            torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
        elif init_weights == 'normal':
            torch.nn.init.normal_(m.weight.data, 0, 0.024)
        elif init_weights == 'ka_uni':
            torch.nn.init.kaiming_uniform_(m.weight.data, 3)
        elif init_weights == 'uni_big':
            torch.nn.init.uniform_(m.weight.data, -1, 1)
        elif init_weights == 'xa_uni_big':
            torch.nn.init.xavier_uniform(m.weight.data)
        elif init_weights == 'ones':
            torch.nn.init.ones_(m.weight.data)
        elif init_weights == 'zeros':
            torch.nn.init.zeros_(m.weight.data)
        elif init_weights == 'default':
            pass

def make_env(env_name: str) -> tuple[gym.Env, bool, int]:
    env = gym.make(env_name, verbose=0)
    if hasattr(env.unwrapped, 'get_action_meanings') and 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireEpisodicLifeEnv(env)
    shape = env.observation_space.shape
    if len(shape) == 3:
        env = ScaledFloatFrame(w.ResizeObservation(env, (84, 84)))
        return env, True, 3
    else:
        return env, False, shape[0] if shape else env.observation_space.n

def get_action_dim(env: gym.Env) -> int:
    if isinstance(env.action_space, Box):
        return env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        return env.action_space.n
    else:
        raise ValueError('Only Box and Discrete action spaces supported')

def make_policy(is_pixel_env, action_dim, input_dim):
    return CNN_heb(input_dim, action_dim) if is_pixel_env else MLP_heb(input_dim, action_dim)

def init_policy_weights(init_weights_type, is_pixel_env, policy, initial_weights_co):
    def weights_init(m):
        _weights_init(m, init_weights_type)

    if init_weights_type == 'coevolve':
        nn.utils.vector_to_parameters(initial_weights_co, policy.parameters())
    else:
        # Randomly sample initial weights from chosen distribution
        policy.apply(weights_init)

          # Load CNN paramters
        if is_pixel_env:
            cnn_weights1 = initial_weights_co[:162]
            cnn_weights2 = initial_weights_co[162:]
            list(policy.parameters())[0].data = cnn_weights1.reshape((6,3,3,3))
            list(policy.parameters())[1].data = cnn_weights2.reshape((8,6,5,5))
    return policy.float()

def fitness_hebb(hebb_rule : str, environment : str, init_weights_type = 'uni' , *evolved_parameters: List[np.array]) -> float:
    """
    Evaluate an agent 'evolved_parameters' controlled by a Hebbian network in an environment 'environment' during a lifetime.
    The initial weights are either co-evolved (if 'init_weights' == 'coevolve') along with the Hebbian coefficients or randomly sampled at each episode from the 'init_weights' distribution.
    Subsequently the weights are updated following the hebbian update mechanism 'hebb_rule'.
    Returns the episodic fitness of the agent.
    """

    # Unpack evolved parameters
    hebb_coeffs = torch.from_numpy(evolved_parameters[0])
    initial_weights_co = torch.from_numpy(evolved_parameters[1]) if len(evolved_parameters) > 1 else None

    with torch.no_grad():
        env, pixel_env, input_dim = make_env(environment)
        env: gym.Env = env
        action_dim = get_action_dim(env)
        policy = make_policy(pixel_env, action_dim, input_dim)
        policy = init_policy_weights(init_weights_type, pixel_env, policy, initial_weights_co)

        weights = [w.detach() for w in policy.parameters()]
        weights = weights[2:] if pixel_env else weights

        observation = env.reset()[0]
        observation = observation if not pixel_env else np.swapaxes(observation,0,2) #(3, 84, 84)

        # Burnout phase for the bullet quadruped so it starts off from the floor
        if 'Bullet' in environment:
            action = np.zeros(8)
            for _ in range(40):
                __ = env.step(action)

        # Normalize weights flag for non-bullet envs
        normalised_weights = ('Bullet' not in environment)

        # Inner loop
        neg_count = 0
        rew_ep = 0
        t = 0
        while True:
            # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
            if isinstance(env.observation_space, Discrete):
                observation = (observation == torch.arange(env.observation_space.n)).float()

            outputs = list(policy([observation]))
            # outputs[0] = outputs[0].numpy()
            # outputs[1] = outputs[1].numpy()
            # outputs[2] = outputs[2].numpy()

            # Bounding the action space
            if 'CarRacing' in environment:
                action = np.array([
                    torch.tanh(outputs[3][0]), torch.sigmoid(outputs[3][1]), torch.sigmoid(outputs[3][2]) ]).astype(np.float32)
                # outputs[3] = outputs[3].numpy()
            elif 'AntBulletEnv' in environment:
                outputs[3] = torch.tanh(outputs[3])
                action = outputs[3].numpy()
            else:
                if isinstance(env.action_space, Box):
                    action = outputs[3].numpy()
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                elif isinstance(env.action_space, Discrete):
                    action = np.argmax(outputs[3]).numpy()
                # outputs[3] = outputs[3].numpy()


            # Environment simulation step
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if 'AntBulletEnv' in environment:
              reward = env.unwrapped.rewards[1] # Distance walked
            rew_ep += reward

            # env.render('human') # Gym envs

            if pixel_env: observation = np.swapaxes(observation,2,0) #(3, 84, 84)

            # Early stopping conditions
            if 'CarRacing' in environment:
                neg_count = neg_count+1 if reward < 0.0 else 0
                if (done or neg_count > 20):
                    break
            elif 'AntBulletEnv' in environment:
                if t > 200:
                    neg_count = neg_count+1 if reward < 0.0 else 0
                    if (done or neg_count > 30):
                        break
            else:
                if done:
                    break
            # else:
            #     neg_count = neg_count+1 if reward < 0.0 else 0
            #     if (done or neg_count > 50):
            #         break

            t += 1

            #### Episodic/Intra-life hebbian update of the weights
            weights = hebbian_update(hebb_rule, hebb_coeffs, weights, outputs)

            # Normalise weights per layer
            if normalised_weights:
              params = list(policy.parameters())
              indices = (0, 1, 2) if not pixel_env else (2, 3, 4)
              for i in indices:
                p = params[i]
                p.data /= p.abs().max()
        env.close()

    return rew_ep