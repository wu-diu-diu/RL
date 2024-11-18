from collections import defaultdict
from ch3.policy_iter import greed_policy
from common.GridWorld import gridworld


def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(action, state, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        V[state] = max(action_values)
    return V


def value_iter(V, env, gamma, threshold=0.001, is_render=True):

    while True:
        if is_render:
            env.render_v(V)
        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)
        delta = 0
        for state in V.keys():
            t = abs(old_V[state] - V[state])
            if delta < t:
                delta = t

        if delta < threshold:  ## 如果最大改变量小于设定阈值，则认为迭代已经收敛
            break
    return V


if __name__ == '__main__':
    V = defaultdict(lambda: 0)
    env = gridworld()
    gamma = 0.9
    V = value_iter(V, env, gamma)
    pi = greed_policy(V, env, gamma)
    env.render_v(V, pi)
