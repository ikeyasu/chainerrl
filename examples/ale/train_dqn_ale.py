from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse
import os

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl.envs import ale
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer

import gym
gym.undo_logger_setup()
from gym import spaces
import gym.wrappers

from dqn_phi import dqn_phi


def parse_activation(activation_str):
    if activation_str == 'relu':
        return F.relu
    elif activation_str == 'elu':
        return F.elu
    elif activation_str == 'lrelu':
        return F.leaky_relu
    else:
        raise RuntimeError(
            'Not supported activation: {}'.format(activation_str))

class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4,
                            initial_bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias),
            L.Linear(22528, n_output_channels, initial_bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h

def parse_arch(arch, n_actions, activation):
    if arch == 'nature':
        return links.Sequence(
            NatureDQNHead(activation=activation, n_input_channels=3),
            L.Linear(512, n_actions),
            DiscreteActionValue)
    elif arch == 'nips':
        return links.Sequence(
            links.NIPSDQNHead(activation=activation, n_input_channels=3),
            L.Linear(256, n_actions),
            DiscreteActionValue)
    elif arch == 'dueling':
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def parse_agent(agent):
    return {'DQN': agents.DQN,
            'DoubleDQN': agents.DoubleDQN,
            'PAL': agents.PAL}[agent]

def clip_action_filter(a):
    return np.clip(a, action_space.low, action_space.high)

def make_env(rom, for_eval):
    env = gym.make(rom)
    #if args.monitor:
    #    env = gym.wrappers.Monitor(env, args.outdir)
    #if isinstance(env.action_space, spaces.Box):
    #    misc.env_modifiers.make_action_filtered(env, clip_action_filter)
    if not for_eval:
        misc.env_modifiers.make_reward_filtered(
            env, lambda x: x * 1e-3)#args.reward_scale_factor)
    #if ((args.render_eval and for_eval) or
    #        (args.render_train and not for_eval)):
    #    misc.env_modifiers.make_rendered(env)
    return env

def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('rom', type=str)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true', default=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--arch', type=str, default='nature',
                        choices=['nature', 'nips', 'dueling'])
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=10 ** 4)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--agent', type=str, default='DQN',
                        choices=['DQN', 'DoubleDQN', 'PAL'])
    args = parser.parse_args()

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    # In training, life loss is considered as terminal states
    env = make_env(args.rom, for_eval=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    action_space = env.action_space
    # In testing, an episode is terminated  when all lives are lost
    eval_env = make_env(args.rom, for_eval=True)

    action_space = env.action_space
    n_actions = action_space.n
    activation = parse_activation(args.activation)
    q_func = parse_arch(args.arch, n_actions, activation)

    # Draw the computational graph and save it in the output directory.
    #chainerrl.misc.draw_computational_graph(
    #    [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
    #    os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, 0.1,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))
    Agent = parse_agent(args.agent)
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=args.clip_delta,
                  update_interval=args.update_interval,
                  batch_accumulator='sum', phi=dqn_phi)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        # In testing DQN, randomly select 5% of actions
        eval_explorer = explorers.ConstantEpsilonGreedy(
            5e-2, lambda: np.random.randint(n_actions))
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_explorer=eval_explorer,
            eval_env=eval_env)

if __name__ == '__main__':
    main()
