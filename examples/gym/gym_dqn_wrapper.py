import random
import warnings

import gym
import numpy as np

imgindex = 0
try:
    import cv2

    def imresize(img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    def grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def imsave(img, filename='screen{0:04d}.png'):
        global imgindex
        filename = filename.format(imgindex)
        imgindex = (imgindex + 1) % 100
        cv2.imwrite(filename, img)

except Exception:
    from PIL import Image

    warnings.warn(
        'Since cv2 is not available PIL will be used instead to resize images.'
        ' This might affect the resulting images.')

    def imresize(img, size):
        return np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))

    def grayscale(img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def imsave(img, filename='screen.png'):
        raise Exception('Not implemented yet')

class GymDQNWrapper(gym.Env):
    def __init__(self, base_env_name="Breakout-v0"):
        self.env = gym.make(base_env_name)
        self.action_space = self.env.action_space
        self.reward_range = self.reward_range
        assert self.env.observation_space.shape == (210, 160, 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84))  # 4 frames 84x84
        self._last_obs = None
        self.metadata = self.env.metadata

    @property
    def spec(self):
        return self.env.spec

    def step(self, action):
        obs_frames = []
        max_reward = 0
        is_done = False
        latest_info = None

        for i in range(0, 4):
            obs, reward, done, info = self.env.step(action)
            # image type observation
            img = self._reform_obs(obs)
            obs_frames.append(img)

            max_reward = max(max_reward, reward)
            is_done = is_done | done
            latest_info = info

        obs_frames = np.array(obs_frames, np.uint8)
        return obs_frames, max_reward, is_done, latest_info

    def _reform_obs(self, obs):
        obs = np.array(obs)
        maximized_img = np.maximum(obs, self._last_obs) if self._last_obs is not None else obs
        self._last_obs = np.array(obs)
        resize_image = np.array(imresize(maximized_img, (84, 84)))
        img = np.dot(resize_image, np.array([0.299, 0.587, 0.114]))
        return img

    def reset(self):
        obs_frames = []
        obs = self.env.reset()
        obs = self._reform_obs(obs)
        for i in range(0, 4):
            obs_frames.append(obs)
        return np.array(obs_frames, np.uint8)


    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


# test
if __name__ == '__main__':
    env = GymDQNWrapper('Breakout-v0')
    env.render()
    env.reset()
    for i in range(0, 1000):
        obs, reward, done, info = env.step(random.randint(0, 3))
        if done:
            env.reset()
        env.render()
