from collections import defaultdict
from common.utils import timer_decorator
from common.GridWorld import gridworld
from policy_eval import policy_eval


def argmax(d):
    """
    :param d: a dict
    :return: the id of max-value in the dict
    """
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:  ## 如果有多个最大值，函数返回最后一个最大值对应的键
            max_key = key
    return max_key


def greed_policy(V, env, gamma):
    pi = {}
    action_dict = {}
    for state in env.states():
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(action, state, next_state)
            value = r + gamma * V[next_state]
            action_dict[action] = value  ## 计算每个action对应的value并保存
        max_action = argmax(action_dict)  ## 得到最大的value对应的索引，即action
        prob = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        prob[max_action] = 1.0
        pi[state] = prob  ## 得到新的最优策略，是一个确定性策略
    return pi


def policy_iter(env, gamma, threshold=0.001, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)
    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greed_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break

        pi = new_pi
    return pi


@timer_decorator
def main():
    env = gridworld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)


if __name__ == '__main__':
    main()
