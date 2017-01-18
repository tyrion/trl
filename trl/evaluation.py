import numpy as np

from ifqi.envs import get_space_info



class RandomPolicy:
    def __init__(self, mdp):
        self.mdp = mdp

    def draw_action(self, *args):
        return self.mdp.action_space.sample()


def allocate_episodes(mdp, n=1):
    state_dim, action_dim, reward_dim = get_space_info(mdp)

    episodes = np.zeros(mdp.horizon * n, [
        ('state', float, state_dim),
        ('action', float, action_dim),
        ('reward', float, reward_dim),
        ('next_state', float, state_dim),
        ('absorbing', float, 1),
        ('done', float, 1),
    ])
    episodes = np.rec.array(episodes, copy=False)
    return episodes


def collect_episodes(mdp, policy=None, n=1):
    episodes = allocate_episodes(mdp, n)

    i = 0
    for _ in range(n):
        e = collect_episode(mdp, policy, episodes[i:i+mdp.horizon])
        i += e.size

    return episodes[:i]


def collect_episode(mdp, policy=None, episodes=None):
    if policy is None:
        policy = RandomPolicy(mdp)

    if episodes is None:
        episodes = allocate_episodes(mdp, 1)

    state = mdp.reset()

    for t in range(mdp.horizon):
        action = policy.draw_action(state)
        next_state, reward, done, _ = mdp.step(action)
        episodes[t] = (state, action, reward, next_state, 0, done)

        if done:
            episodes = episodes[:t+1]
            break
        state = next_state
    episodes[t].absorbing = True
    episodes.flags['WRITEABLE'] = False

    return episodes
