from collections import defaultdict

from common.GridWorld import gridworld


def eval_onestep(pi, V, env, gamma=0.9):
    """
    :param pi: policy
    :param V: Value func
    :param env: environment
    :param gamma: 折现率
    :return: 一个时间步后，更新后的状态价值函数
    """
    for state in env.states():  ## 每个state都是一个元组
        if state == env.goal_state:  ## 按照从上到下，从左到右的顺序与遍历每个state，计算其状态价值
            V[state] = 0
            continue  ##  goal state has no value

        action_prob = pi[state]  ## pi是defaultdict 即键值没有初始化，只初始化了value 输入任何键值，对应value均为0
        ## 这里取出的每个状态的策略概率分布都是相同的，即初始化时的值
        new_V = 0

        for action, prob in action_prob.items():  ## aciton_prob是一个字典，键是行动， 值是概率
            next_state = env.next_state(state, action)  ## 根据action对应行动，更新state,同时注意撞墙和障碍物时是动不了的
            reward = env.reward(action, state, next_state)  ## 根据reward_map得到下一个状态的奖励值, 为了和书中公式保持一致
            ## 加了action和state
            new_V += prob * (reward + gamma * V[next_state])  ## 贝尔曼公式

        V[state] = new_V  ## 迭代更新每个state的状态价值
    return V


def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)  ## 迭代一次，更新所有状态的状态价值

        delta = 0
        for state in V.keys():  ## 此时V在eval_onestep函数中已经添加了键
            t = abs(V[state] - old_V[state])  ## 计算本次迭代，状态价值的该变量
            if delta < t:  ## 每次迭代都计算改变量，但只记录最大的改变量
                delta = t

        if delta < threshold:  ## 如果最大改变量小于设定阈值，则认为迭代已经收敛
            break
    return V


if __name__ == '__main__':
    env = gridworld()
    gamma = 0.9
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
