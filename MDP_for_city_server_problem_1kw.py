import copy
import inspect
import io
import itertools
import os
import pickle
import random
import sys

import numpy as np
import pulp

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def generate_state(
    I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd
):
    """
    生成一个系统的随机状态，该系统负责将任务分配给员工。

    参数:
    I_citys (int): 城市的数量。
    L_levels (int): 等级的数量。
    W_workdays (int): 员工距离放假的最大工作天数, 一般为7。
    M_servers (int): 员工的数量。
    x_max_task_num (int): 每个城市的最大任务数(本函数中未直接使用)。
    H_home_of_server (list[int]): 代表每个员工家所在城市的列表。
    lambd (np.array): 一个二维数组 (I_citys x_max_task_num L_levels)，代表每个城市和等级的任务到达率。

    返回:
    tuple: 包含任务分布矩阵和员工状态列表的元组。

    城市中的任务矩阵，员工状态列表[(员工所在地，距离放假时间)]
    """

    # 生成任务分布矩阵 (n_il)，其维度为 I_citys x_max_task_num L_levels
    n_il = np.zeros((I_citys, L_levels), dtype=int)  # 用零初始化矩阵
    for i_city in range(I_citys):  # 遍历城市
        for l_level in range(L_levels):  # 遍历等级
            # 为每个城市和等级分配一个基于泊松分布的随机数, 表示随机状态生成的任务数量？# TODO
            n_il[i_city, l_level] = np.random.poisson(lambd[i_city, l_level])
    S0_tasks = n_il  # 任务分布矩阵 (I_citys x_max_task_num L_levels), [0, +∞)

    # 初始化一个列表来保存每个员工的状态，给
    S1_servers = []
    for m_server in range(M_servers):  # 遍历所有员工

        # 为员工 'm_server' 随机选择距离放假的工作日数，取值范围 [0, W_workdays]
        w_m = np.random.randint(0, W_workdays + 1)

        # 根据距离放假天数w_m，得到位置i_m，如果距离放假0，则在家里，否则随机一个位置。
        # i_m [1, I_citys]
        if (
            w_m == W_workdays
        ):  # 如果距离放假时间为 0 天，即今天放假，则员工所在城市为家所在城市
            i_m = H_home_of_server[m_server]
        else:  # 否则，为员工随机选择一 个非家乡城市工作
            i_m = np.random.randint(1, I_citys + 1)

        # 将员工的状态作为元组（城市，距离放假的工作日数）添加到列表中
        S1_servers.append((i_m, w_m))

    # 将任务分布矩阵和员工状态列表合并成一个状态元组
    S = (
        S0_tasks,
        S1_servers,
    )  # ((I_citys x_max_task_num L_levels), (M_servers x_max_task_num 1))
    return S  # 返回生成的状态


def aggreg_state(S_t, Z_cluster_num, X, M_servers, I_citys, L_levels):
    # 函数2
    # 定义一个函数，用于根据给定的参数将复杂的状态 S 压缩成一个简化的状态 barS。
    # 输入:
    #   S: 当前状态，一个复杂的结构，包含两部分信息：
    #      - 一个数组，表示每个城市每个等级的数量(n_il)。
    #      - 一个列表，表示服务员和他们服务的城市及工作日(i_m, w_m)
    #   Z_cluster_num: 一个整数，表示将城市分成多少个聚类。
    #   X: 一个整数，用于计算 N 矩阵中的元素值。
    #   M_servers: 服务员的总数。
    #   I_citys: 城市的总数。
    #   L_levels: 等级的总数。
    # 输出:
    #   barS: 一个元组，表示压缩后的状态，包含以下三个部分：
    #         - N: 一个二维数组，表示每个聚类的等级之和。
    #         - g: 一个数组，表示每个聚类的状态。
    #         - w: 一个整数，表示第一个服务员的工作日数。

    # 计算正在工作的服务员的数量, S[1][m_server][1]即S1_servers[m_server][1]即w_m
    barM = np.sum(
        [1 for m_server in range(M_servers) if S_t[1][m_server][1] != 0]
    )  # 距离放假时间不等于0
    # 根据城市数量和设定的簇数，将城市分成Z个簇
    cluster = split_list(I_citys, Z_cluster_num)
    # 计算实际的簇数，考虑到可能会有余数
    # num_cluster = divide_reminder(I_citys, Z_cluster_num) # 这个可以替换为下面的
    num_cluster = np.ceil(I_citys / Z_cluster_num).astype(int)  # 向上取整
    # 初始化表示各簇状态的数组g
    g = np.zeros(num_cluster)

    # 压缩状态的第二部分：计算每个簇的状态
    for z_cluster in range(num_cluster):
        # 统计每个簇中有多少业务员正在工作
        e_z = np.sum(
            [
                1
                for m_server in range(M_servers)
                if S_t[1][m_server][0] in cluster[z_cluster]
            ]
        )
        # 根据工作的业务员数量设置簇的状态
        if e_z == 0:
            g[z_cluster] = 0  # 无业务员工作
        elif e_z <= barM / num_cluster:
            g[z_cluster] = 1  # 工作业务员数量低于或等于平均值
        else:
            g[z_cluster] = 2  # 工作业务员数量高于平均值

    # 获取第一个业务员的工作量
    w = S_t[1][0][1]
    # 压缩状态的第一部分：计算每个簇中各等级的数量总和
    N = np.zeros((num_cluster, L_levels))  # 初始化N矩阵

    i_all = [i for i in cluster[z_cluster]]
    # for z_cluster in range(num_cluster):
    #     for l in range(L_levels):
    #         # 对每个簇的每个等级，计算其数量总和，但不超过X

    #         N[z_cluster][l] = min(X,
    #                               np.sum([S_t[0][i-1][l] for i in cluster[z_cluster]]))
    for z_cluster in range(num_cluster):
        for l in range(L_levels):
            # 对每个簇的每个等级，计算其数量总和，但不超过X
            task_sum = sum(float(S_t[0][i - 1][l]) for i in cluster[z_cluster])
            N[z_cluster][l] = min(X, task_sum)
    # 将计算出的N矩阵、簇的状态数组g和第一个业务员的工作量w组合成新的压缩状态barS
    barS = (N, g, w)
    return barS


# 将城市列表平均分成Z个聚类
def split_list(I_citys, Z_cluster_num) -> list:
    # 创建一个从1到I_citys的城市索引列表
    arr_city_idx = list(range(1, I_citys + 1))
    # 调用函数处理实际的分割
    return split_array_given_array(arr_city_idx, Z_cluster_num)


def split_array_given_array(arr_city_idx, Z_cluster_num) -> list:
    """
    将输入数组分割成长度为 Z_cluster_num 的子数组列表。如果数组不能被 Z_cluster_num 整除，
    那么最后一个子数组将包含所有剩余的元素。

    参数:
    arr_city_idx (list): 需要被分割的输入数组。
    Z_cluster_num (int): 每个子数组的期望长度。

    返回值:
    list: 长度为 Z_cluster_num 的子数组列表，除了可能的最后一个子数组，它包含所有剩余的元素。
    """
    result = []  # 结果列表，用来存储所有的子数组
    quotient = len(arr_city_idx) // Z_cluster_num  # 计算整除的商，即完整子数组的数量
    remainder = (
        len(arr_city_idx) % Z_cluster_num
    )  # 计算余数，即最后一个子数组的元素数量

    # 划分可以整除的数组部分
    for i in range(quotient):
        sub_array = arr_city_idx[
            i * Z_cluster_num : (i + 1) * Z_cluster_num
        ]  # 获取从 i*Z_cluster_num 到 (i+1)*Z_cluster_num 的子数组
        result.append(sub_array)  # 将子数组添加到结果列表中

    # 如果有余数，则处理剩余部分
    if remainder > 0:
        sub_array = arr_city_idx[-remainder:]  # 获取数组最后余数个元素形成的子数组
        result.append(sub_array)  # 将子数组添加到结果列表中

    return result  # 返回结果列表


def divide_reminder(num, divisor):
    """
    将一个整数除以另一个整数，并将结果向上取整。

    参数:
    num (int): 被除数。
    divisor (int): 除数。

    返回值:
    int: 向上取整后的商。
    """
    quotient = num // divisor  # 计算整除的商
    remainder = num % divisor  # 计算余数

    # 如果存在余数，则将商向上取整
    if remainder > 0:
        quotient += 1  # 余数大于0，商加一

    return quotient  # 返回向上取整后的商


def func4(S, mathscr_L):
    # 假设 len(mathscr_L) = 7
    # 假设 S 和 mathscr_L 的数据结构如所描述
    # S 是一个二元组 (S0_tasks, S1_servers)，其中 S0_tasks 是任务矩阵
    # mathscr_L 是业务员的等级集合，例如 [l_1, l_2, ..., l_M]
    # S0 (I_citys x_max_task_num L_levels)
    # 首先计算 N_1 和 N_2
    L_levels = 5
    N_1 = [
        sum(1 for l_m, (i_m, w_m) in zip(mathscr_L, S[1]) if l_m == j and w_m != 0)
        for j in range(1, L_levels + 1)
    ]
    N_2 = [sum(S[0][i][j] for i in range(len(S[0]))) for j in range(L_levels)]
    # 初始化分类后的等级列表
    mathcal_L = []
    current_class = []
    total_N_1 = 0
    total_N_2 = 0

    for j in range(1, L_levels + 1):
        total_N_1 += N_1[j - 1]
        total_N_2 += N_2[j - 1]
        current_class.append(j)

        if total_N_1 <= total_N_2:
            # 当 N_1 总和小于等于 N_2 总和时，终止当前类的添加
            mathcal_L.append(tuple(current_class))
            current_class = []
            total_N_1 = 0
            total_N_2 = 0

    if current_class:
        # 添加最后一个类
        mathcal_L.append(tuple(current_class))

    return mathcal_L, N_1, N_2


def func3(S, L_server, H_home_of_server, r1_reward, c1_city_cost, c2):
    """
    生成状态 S 到决策 A 的函数，通过解决线性规划问题来最大化收益 R_total_reward(S, A)。

    参数:
    S (tuple): 当前状态，包含任务矩阵和服务员信息。
    L_server (list): 服务员的等级列表。
    H_home_of_server (list): 服务员的家位置列表。
    r1_reward (list): 每个等级的收益列表。
    c1_city_cost (list of list): I×I 的成本矩阵。
    c2 (float): 常数成本。

    返回:
    list: 最优决策 A，包含每个服务员的位置和等级。
    """
    n_il, servers_info = S

    # n_il, servers_info = S
    # M_servers = len(servers_info)  # 服务员数量
    # I_citys = len(n_il)          # 城市数量
    # # L_max = len(r1_reward) - 1    # 最大等级

    M_servers = len(servers_info)  # 服务员数量
    I_citys = len(n_il)  # 城市数量
    L_max = max(L_server)  # 最大等级
    # print(f"{M_servers=}40 {I_citys=}26 {L_max=}5 {L_server=}")

    # 创建问题实例
    prob = pulp.LpProblem("Optimal_Server_Assignment", pulp.LpMaximize)

    # 定义决策变量 y_{mil} 为二元变量
    y = pulp.LpVariable.dicts(
        "y",
        (
            (m, i, l)
            for m in range(M_servers)
            for i in range(I_citys)
            for l in range(L_max + 1)
        ),
        cat=pulp.LpBinary,
    )

    # 目标函数
    # 假设 L_max 和 r1_reward 已经定义

    prob += pulp.lpSum(
        r1_reward[l1] * y[m1, i1, l1]
        - c1_city_cost[servers_info[m1][0] - 1][i1] * y[m1, i1, l1]
        for m1 in range(M_servers)
        for i1 in range(I_citys)
        for l1 in range(0, L_max + 1)
    ) - c2 * pulp.lpSum(
        n_il[i][l - 1] - pulp.lpSum(y[m, i, l] for m in range(M_servers))
        for i in range(I_citys)
        for l in range(1, L_max + 1)
    )

    # 添加约束
    # 每个服务员只能分配到一个地点和等级
    for m in range(M_servers):
        prob += (
            pulp.lpSum(y[m, i, l] for i in range(I_citys) for l in range(L_max + 1))
            == 1
        )

    # 服务员不工作时，分配到家乡的等级0
    for m, (im, wm) in enumerate(servers_info):
        if wm == 0:
            prob += y[m, H_home_of_server[m], 0] == 1

    # 服务员工作时，必须分配到合适的等级和城市
    for m, (im, wm) in enumerate(servers_info):
        if wm > 0:
            prob += (
                pulp.lpSum(
                    y[m, i, l]
                    for i in range(I_citys)
                    for l in range(L_server[m], L_max + 1)
                    if n_il[i][l - 1] > 0
                )
                == 1
            )

    # 资源使用不超过可用数量
    for i in range(I_citys):
        for l in range(1, L_max + 1):
            prob += pulp.lpSum(y[m, i, l] for m in range(M_servers)) <= n_il[i][l - 1]

    # 求解问题
    prob.solve()

    # 解析结果
    result = [
        (m, i + 1, l)
        for m in range(M_servers)
        for i in range(I_citys)
        for l in range(L_max + 1)
        if pulp.value(y[m, i, l]) == 1
    ]
    return result


def func5(
    S, mathcal_L, mathscr_L, N_1, N_2, H_home_of_server, r1_reward, c1_city_cost, c2
):
    """
    生成状态 S 到决策 Y_best_allocation 的函数,通过解决线性规划问题来最大化收益 R_total_reward(S, Y_best_allocation)。

    参数:
    S (tuple): 当前状态,包含任务矩阵和服务员信息。
    mathcal_L (list): 分类后的等级列表。
    mathscr_L (list): 所有服务员的等级列表
    N_1 (list): 每个等级的服务员数量。
    N_2 (list): 每个等级的任务数量。
    H_home_of_server (list): 服务员的家位置列表。
    r1_reward (list): 每个等级的收益列表。
    c1_city_cost (list of list): I×I 的成本矩阵。
    c2 (float): 常数成本。

    返回:
    list: 最优决策 Y_best_allocation,包含每个服务员的位置和等级。
    R_total_reward, 总收益
    """
    n_il, servers_info = S
    M_servers = len(servers_info)  # 服务员数量
    I_citys = len(n_il)  # 城市数量
    L_max = [max(l) for l in mathcal_L]  # 最大等级
    # 步骤1:安排放假的员工回家
    C_h = sum(
        c1_city_cost[servers_info[m][0] - 1][H_home_of_server[m]]
        for m in range(M_servers)
        if servers_info[m][1] == 0
    )

    R_total_reward = -C_h  # 初始化总收益为负的回家成本

    Y_best_allocation = [None] * M_servers  # 初始化最优决策 Y_best_allocation

    Y_set = []
    # 步骤2:对每个等级类独立进行员工分配
    for L_set, l_max_L in zip(mathcal_L, L_max):
        M_servers_L = [
            m
            for m in range(M_servers)
            if servers_info[m][1] > 0 and mathscr_L[m] in L_set
        ]  # 该等级类下工作的员工集合
        I_citys_L = [
            i for i in range(I_citys) if any(n_il[i][l - 1] > 0 for l in L_set)
        ]  # 该等级类下有任务需求的城市集合

        # 创建问题实例
        prob = pulp.LpProblem(
            f"Optimal_Server_Assignment_Level_{L_set}", pulp.LpMaximize
        )

        # 定义决策变量 y_{mil} 为二元变量
        y = pulp.LpVariable.dicts(
            "y",
            ((m, i, l) for m in M_servers_L for i in I_citys_L for l in L_set),
            cat=pulp.LpBinary,
        )

        # 定义一个辅助变量,存储服务员分配的收益和城市成本
        # 这个表达式计算了每个服务员分配到每个城市每个等级任务的收益和成本

        reward_cost_expr = pulp.lpSum(
            r1_reward[l1 - 1] * y[m1, i1, l1]
            - c1_city_cost[servers_info[m1][0] - 1][i1] * y[m1, i1, l1]
            for m1 in M_servers_L
            for i1 in I_citys_L
            for l1 in L_set
        )

        # 目标函数
        # 根据等级类型选择
        # 如果每个等级的服务员数量小于等于任务数量,则最大化收益减去未分配任务的惩罚
        if sum(N_1[l - 1] for l in L_set) <= sum(N_2[l - 1] for l in L_set):
            prob += reward_cost_expr - c2 * pulp.lpSum(
                n_il[i][l - 1] - pulp.lpSum(y[m, i, l] for m in M_servers_L)
                for i in I_citys_L
                for l in L_set
            )
        # 否则,只最大化收益
        else:
            prob += reward_cost_expr

        # 添加约束
        for m in M_servers_L:
            # 每个工作中的服务员 m,要求其被分配到城市 i 提供的服务等级 l 必须不低于他自身的服务等级 L_server[m]
            # 且只能被分配到一个城市提供一种等级的服务。
            prob += (
                pulp.lpSum(
                    y[m, i, l] for i in I_citys_L for l in L_set if l >= mathscr_L[m]
                )
                == 1
            )

        # 遍历每个城市和等级
        for i in I_citys_L:
            for l in L_set:
                # 如果每个等级的服务员数量小于等于任务数量,则添加约束:
                # 每个城市每个等级分配的服务员数量小于等于该等级任务数量
                if sum(N_1[l - 1] for l in L_set) <= sum(N_2[l - 1] for l in L_set):
                    prob += (
                        pulp.lpSum(y[m, i, l] for m in M_servers_L) <= n_il[i][l - 1]
                    )
                # 否则,添加约束:
                # 每个城市每个等级分配的服务员数量等于该等级任务数量
                else:
                    prob += (
                        pulp.lpSum(y[m, i, l] for m in M_servers_L) == n_il[i][l - 1]
                    )

        # 求解问题
        prob.solve()

        # 遍历每个服务员、城市和等级
        for m in M_servers_L:
            for i in I_citys_L:
                for l in L_set:
                    # 初始化一个临时列表,用于存储分配结果
                    Y_sub_set = []
                    # 如果当前服务员被分配到某个城市某个等级
                    if pulp.value(y[m, i, l]) == 1:
                        # 将当前服务员、城市(城市编号从1开始)和等级的分配结果存储到Y_best_allocation字典中
                        Y_best_allocation[m] = (m, i + 1, l)
                        # 中断内层循环,开始处理下一个服务员
                        break

        # 更新总收益
        R_total_reward += pulp.value(prob.objective)

        # 提取结果
        # Y_L = [(i, l) for m in M_servers_L for i in I_citys_L for l in L_set if y[m, i, l].reward() == 1]
        # Y_set.append(Y_L)

    # 步骤3:安排放假的员工
    for m in range(M_servers):
        if servers_info[m][1] == 0:
            Y_best_allocation[m] = (m, H_home_of_server[m] + 1, 0)  # 城市编号从1开始
    # # 步骤3:计算总收益
    # R_total_reward = sum(prob.objective.reward() for L in L_set) - C_h
    #     Y_set = []
    # 解析结果

    return Y_best_allocation, R_total_reward


def func6(S, mathcal_L, mathscr_L, N_1, N_2):
    """
    生成状态 S 的决策空间 A,满足约束条件。

    参数:
    S (tuple): 当前状态,包含任务矩阵和服务员信息。
    mathcal_L (list): 分类后的等级列表。
    mathscr_L (list): 所有服务员的等级列表
    N_1 (list): 每个等级的服务员数量。
    N_2 (list): 每个等级的任务数量。

    返回:
    list: 决策空间 A,包含每个服务员的所有可能决策。
    """
    n_il, servers_info = S
    M_servers = len(servers_info)  # 服务员数量
    I_citys = len(n_il)  # 城市数量

    A = [[] for _ in range(M_servers)]  # 初始化决策空间 A
    A_mil = A

    # 对每个服务员生成可能的决策
    for m in range(M_servers):
        if servers_info[m][1] == 0:  # 服务员 m 放假
            A[m].append((servers_info[m][0], 0))  # 放假的服务员只有一个决策,即回家
        else:  # 服务员 m 工作
            for L_set in mathcal_L:  # 遍历每个等级类
                if mathscr_L[m] in L_set:  # 如果服务员 m 的等级属于当前等级类
                    for i in range(I_citys):  # 遍历每个城市
                        for l in L_set:  # 遍历当前等级类的每个等级
                            if l >= mathscr_L[m]:  # 如果当前等级不低于服务员 m 的等级
                                if (
                                    sum(N_1[l - 1] for l in L_set)
                                    <= sum(N_2[l - 1] for l in L_set)
                                    and sum(1 for a in A[m] if a[0] == i + 1)
                                    < sum(n_il[i][l - 1] for l in L_set)
                                ) or (
                                    sum(N_1[l - 1] for l in L_set)
                                    > sum(N_2[l - 1] for l in L_set)
                                    and sum(
                                        1 for a in A[m] if a[0] == i + 1 and a[1] == l
                                    )
                                    < n_il[i][l - 1]
                                ):
                                    # 如果满足约束条件,则将决策添加到服务员 m 的决策空间中
                                    A[m].append((i + 1, l))  # 城市编号从1开始

    return A


def func7(T, x_max_task_num, lambda_il):
    # 生成了每日新到达的任务?
    # T: 表示时间周期，例如天数
    # x_max_task_num: 矩阵元素的最大取值
    # lambda_il: 泊松分布的率参数矩阵 (I_citys x_max_task_num L_levels)

    # 获取 lambda_il 的维度为 I_citys 和 L_levels
    I_citys, L_levels = lambda_il.shape

    # 初始化三维数组
    arriving_taskss = np.zeros((T, I_citys, L_levels), dtype=int)

    # 生成每个时间步的 I_citys x_max_task_num L_levels 矩阵
    for t in range(T):
        for i in range(I_citys):
            for l in range(L_levels):
                # 使用泊松分布生成矩阵元素
                arriving_taskss[t, i, l] = min(
                    np.random.poisson(lambda_il[i, l]), x_max_task_num
                )

    return arriving_taskss


func1 = generate_state
func2 = aggreg_state

func3 = func3
func4 = func4
func5 = func5
func6 = func6
func7 = func7


class ValueFunction:
    def __init__(self, T=7, file_path="./reward_function.pkl"):
        self.rewards = {key: {} for key in range(T + 1)}  # 用字典存储状态-价值对
        self.count = {key: {} for key in range(T + 1)}  # 用字典存储状态出现次数
        self.policies = {key: {} for key in range(T + 1)}  # 用字典存储状态-最优决策对
        self.file_path = file_path

    def set_reward(self, t, S_t, reward):
        """
        存储阶段t和状态state的价值reward
        """
        S_t_str = process_state(S_t)
        self.rewards[t][S_t_str] = reward

    def get_reward(self, t, S_t):
        """
        获取阶段t和状态state的价值,如果状态不存在,则返回默认值0
        """
        S_t_str = process_state(S_t)
        if S_t_str not in self.rewards[t]:
            self.rewards[t][S_t_str] = 0
        else:
            if self.rewards[t][S_t_str] > 0:
                (f"rewards 大于 0 get rewards! {t=} {self.rewards[t][S_t_str]=} ")

        reward = self.rewards[t][S_t_str]
        return reward

    def update_reward(self, t, S_t, current_reward):
        updated = False
        last_reward = self.get_reward(t, S_t)
        if last_reward >= current_reward:
            pass
        else:
            self.set_reward(t, S_t, current_reward)
            updated = True
        return updated

    def set_policy(self, t, S_t, policy_t_s):
        """
        设置阶段t和状态state的最优决策policy_t_s
        """
        S_t_str = process_state(S_t)
        self.policies[t][S_t_str] = policy_t_s

    def get_policy(self, t, S_t):
        """
        获取阶段t和状态state的最优决策,如果状态不存在或没有最优决策,则返回None
        """
        S_t_str = process_state(S_t)
        if S_t_str not in self.policies[t]:
            self.policies[t][S_t_str] = None
        else:
            print(f"get policy! {t=} {self.policies[t][S_t_str]=} {S_t=}")
        return self.policies[t][S_t_str]

    # def update_policy(self, t, S_t, Y_best_allocation, S_t_1, S_last_t_1):
    #     S_t_str = process_state(S_t)
    #     R_s_t = self.get_reward(t + 1, S_t_1)
    #     R_s_t_last = self.get_reward(t + 1, S_last_t_1)
    #     if S_t_str not in self.policies[t] or R_s_t > R_s_t_last:
    #         self.set_policy(self, t, S_t_str, policy_t_s=Y_best_allocation)

    def get_count(self, t, S_t):
        """
        获取阶段t和状态state的出现次数,如果状态不存在,则返回默认值1 表示当前出现一次
        """
        S_t_str = process_state(S_t)
        if S_t_str not in self.count[t]:
            self.set_count(t=t, S_t=S_t, count=1)
        else:
            if self.count[t][S_t_str] > 1:
                print(f"get count! {t} {self.count[t][S_t_str]} {S_t}")
        return self.count[t][S_t_str]

    def set_count(self, t, S_t, count):
        S_t_str = process_state(S_t)
        self.count[t][S_t_str] = count

    def init_or_add_count(self, t, S_t):
        count = self.get_count(t, S_t)
        return count

    def save_model(self, file_path=None):
        """
        保存 ValueFunction 实例到文件
        """
        if file_path is None:
            file_path = self.file_path
        if file_path is None:
            raise ValueError(
                "No file path provided for saving the ValueFunction object."
            )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        """
        从文件中加载 ValueFunction 实例
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)


def process_state(S_t):
    """
    将状态转换为NumPy数组，并返回转换后的状态。
    """
    if type(S_t) == str:
        S_t_str = S_t
    else:
        S_t_str = str(S_t)
    # caller_frame = inspect.currentframe().f_back
    # caller_name = caller_frame.f_code.co_name
    # # print(f"{caller_name=} {S_t_str=}")
    return S_t_str


class Func8:

    pi = {}  # 初始化策略
    I_citys = 3
    L_levels = 5
    W_workdays = 6
    M_servers = 3
    x_max_task_num = 2
    random.seed(42)
    lambd = np.random.rand(I_citys, L_levels)
    r1_reward = [0, 3500, 3000, 2500, 2000, 1500]
    c2 = 100

    max_iter = 1000000
    T = 7

    Z_cluster_num = 3
    X = 3

    def __init__(self):

        self.V = ValueFunction()  # 初始化值函数
        self.H_home_of_server = [
            random.randint(1, self.I_citys) for _ in range(self.M_servers)
        ]  # 随机家的位置
        self.c1_city_cost = [
            [0 if i == j else random.randint(100, 500) for j in range(self.I_citys)]
            for i in range(self.I_citys)
        ]
        self.lambda_il = np.random.rand(self.I_citys, self.L_levels)  # 生成率参数矩阵
        self.arriving_tasks_for_T = func7(self.T, self.x_max_task_num, self.lambda_il)
        self.mathscr_L = [
            random.randint(1, self.L_levels) for _ in range(self.M_servers)
        ]
        print(f"{self.H_home_of_server=} {self.mathscr_L=} {self.c1_city_cost}")

    def func3_transfer(
        self,
        S_t,
        L_server,
        H_home_of_server,
        r1_reward,
        c1_city_cost,
        c2,
        t,
        Xi_arriving_tasks,
    ):

        n_il, servers_info = S_t
        M_servers = self.M_servers  # 服务员数量
        I_citys = self.I_citys  # 城市数量
        L_max = max(L_server)  # 最大等级
        n_next = [
            [pulp.LpVariable(f"x_{i}_{j}", cat="Integer") for j in range(L_max + 1)]
            for i in range(I_citys)
        ]
        ser_info_1 = [
            pulp.LpVariable(f"u_{i}", cat="Integer") for i in range(M_servers)
        ]
        ser_info_2 = [0] * M_servers

        # 创建问题实例
        prob = pulp.LpProblem("Optimal_Server_Assignment", pulp.LpMaximize)

        # 定义决策变量 y_{mil} 为二元变量
        y = pulp.LpVariable.dicts(
            "y",
            (
                (m, i, l)
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(L_max + 1)
            ),
            cat=pulp.LpBinary,
        )

        for i in range(I_citys):
            for l in range(L_max):
                n_next[i][l] = (
                    n_il[i][l]
                    - pulp.lpSum(y[m, i, l + 1] for m in range(M_servers))
                    + Xi_arriving_tasks
                )

        # 每个服务员 m 最多只能被分配到一个城市的一个等级的任务。
        for m in range(M_servers):
            ser_info_1[m] = pulp.lpSum(
                i * pulp.lpSum(y[m, i, l] for l in range(0, L_max + 1))
                for i in range(I_citys)
            )

        for m in range(M_servers):
            if servers_info[m][1] == 0:
                ser_info_2[m] = self.W_workdays
            else:
                ser_info_2[m] = servers_info[m][1] - 1
        ser_info_next = list(zip(ser_info_1, ser_info_2))
        S_next = (n_next, ser_info_next)

        policy_for_S = self.V.get_policy(t, S_t=S_t)
        if policy_for_S:
            S_next_last = self.transition(
                S=S_t,
                Y_best_allocation=policy_for_S,
                Xi_arriving_tasks=Xi_arriving_tasks,
            )
        else:
            S_next_last = S_t  # 这里和数学上可能有所不同
        reward = self.V.get_reward(t=t, S_t=S_next_last)

        prob += (
            pulp.lpSum(
                r1_reward[l] * y[m, i, l]
                - c1_city_cost[servers_info[m][0] - 1][i] * y[m, i, l]
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(0, L_max + 1)
            )
            - c2
            * pulp.lpSum(
                n_il[i][l - 1] - pulp.lpSum(y[m, i, l] for m in range(M_servers))
                for i in range(I_citys)
                for l in range(1, L_max + 1)
            )
            + reward
        )

        # for m in range(M_servers):
        #     prob += pulp.lpSum(y[m, i, l] for i in range(I_citys) for l in range(L_max+1)) == 1

        # 分配给所有服务员的所有任务总数不能超过实际的任务总数。
        total_tasks = sum(
            n_il[i][l - 1] for i in range(I_citys) for l in range(1, L_max + 1)
        )
        prob += (
            pulp.lpSum(
                y[m, i, l]
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(1, L_max + 1)
            )
            <= total_tasks
        )

        # 分配的任务总数不能超过可分配的业务员数量 (非休息日)。
        available_servers = sum(1 for _, (_, wm) in enumerate(servers_info) if wm > 0)
        prob += (
            pulp.lpSum(
                y[m, i, l]
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(1, L_max + 1)
            )
            <= available_servers
        )

        for m, (im, wm) in enumerate(servers_info):
            if wm == 0:
                # 休息日服务员必须待在家中
                prob += y[m, H_home_of_server[m] - 1, 0] == 1
            elif wm > 0:
                # 服务员必须分配任务，且任务等级必须高于服务员等级
                # 1. 允许选择 "待在家中" (l = 0)
                prob += (
                    pulp.lpSum(
                        y[m, i, l]
                        for i in range(I_citys)
                        for l in range(L_max + 1)
                        if (l == 0 or (l >= L_server[m] and n_il[i][l - 1] > 0))
                    )
                    == 1
                )

        # 每个服务员必须且只能被分配到一个任务，无论是在某个城市执行任务还是在家休息。
        for m in range(M_servers):
            prob += (
                pulp.lpSum(y[m, i, l] for i in range(I_citys) for l in range(L_max + 1))
                == 1
            )

        # 限制了服务员只能接受等级不高于自身等级的任务
        for m in range(M_servers):
            for i in range(I_citys):
                for l in range(1, L_max + 1):
                    prob += y[m, i, l] <= (L_server[m] >= l)

        for i in range(I_citys):
            for l in range(1, L_max + 1):
                # 每个城市每个等级分配的服务员数量不能超过任务数量
                prob += (
                    pulp.lpSum(y[m, i, l] for m in range(M_servers)) <= n_il[i][l - 1]
                )

        # 确保每个城市分配的业务员总量不会超过该城市的任务总量
        for i in range(I_citys):
            prob += pulp.lpSum(
                y[m, i, l] for m in range(M_servers) for l in range(1, L_max + 1)
            ) <= sum(n_il[i])

        # 求解问题
        # 选择求解器，例如：CBC
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        # 解析结果
        result = [None] * self.M_servers

        for m in range(self.M_servers):
            # 找到分配给该服务员的城市和任务等级
            assigned = False  # 用于标记服务员是否被分配了任务
            for i in range(self.I_citys):
                for l in range(L_max + 1):
                    if pulp.value(y[m, i, l]) == 1:
                        result[m] = (m, i + 1, l)  # 添加服务员编号 m
                        assigned = True
                        # print(f"{result[m]=}")
                # if assigned:
                #     break

            # 如果服务员没有被分配任务，则设置其在家
            if not assigned:
                result[m] = (m, H_home_of_server[m], 0)
        # 获取目标函数值
        objective_value = pulp.value(prob.objective)

        # print(f"{result=}")
        return result, objective_value

    def run(self):
        """
        带起始探索的强化学习算法,生成最优值函数和最优策略。

        返回:
        tuple: 包含最优状态、最优值函数和最优策略的元组。
        """
        # 初始化值函数和策略
        self.V = ValueFunction()

        for i in range(self.max_iter):
            print(f"-------------iters:{i=}--------------")
            # 生成初始状态
            S_init = func1(
                I_citys=self.I_citys,
                L_levels=self.L_levels,
                W_workdays=self.W_workdays,
                M_servers=self.M_servers,
                x_max_task_num=self.x_max_task_num,
                H_home_of_server=self.H_home_of_server,
                lambd=self.lambd,
            )
            print(f"{S_init=}")
            # 初始化当前轨迹:决策，状态，收益
            trajectory = []
            S_current = S_init
            for t in range(self.T):

                # 更新状态出现次数
                self.V.init_or_add_count(t=t, S_t=S_current)

                Xi_arriving_tasks = self.arriving_tasks_for_T[t]

                Y_best_allocation, R_total_reward = self.func3_transfer(
                    S_t=S_current,
                    t=t,
                    L_server=self.mathscr_L,
                    H_home_of_server=self.H_home_of_server,
                    r1_reward=self.r1_reward,
                    c1_city_cost=self.c1_city_cost,
                    c2=self.c2,
                    Xi_arriving_tasks=Xi_arriving_tasks,
                )
                print(
                    f"{t=} {len(Y_best_allocation)=} {R_total_reward=} {Y_best_allocation=} {S_current[1]=} {Xi_arriving_tasks=} {S_current[0]=} "
                )
                # 记录轨迹:当前决策，状态，收益
                trajectory.append((S_current, Y_best_allocation, R_total_reward))

                # 状态转移
                S_next = self.transition(
                    S_current, Y_best_allocation, Xi_arriving_tasks
                )

                S_current = S_next
                # print(f"{S_current=}")

            # 逆向计算G和更新V 和保存策略
            G = [0] * (self.T + 1)
            reward_diff_sum = 0
            for t in range(self.T - 1, -1, -1):
                Xi_arriving_tasks = self.arriving_tasks_for_T[t]
                S, Y_best_allocation, R_total_reward = trajectory[t]
                # S_current, Y_best_allocation, R_total_reward
                G[t] = G[t + 1] + R_total_reward
                count = self.V.get_count(t, S)
                reward_s_t = self.V.get_reward(t, S)
                reward = reward_s_t + (1 / count) * (G[t] - reward_s_t)
                if count > 1:
                    print(f"{count=} {reward=}")
                self.V.set_reward(t=t, S_t=S, reward=reward)
                self.V.set_policy(t=t, S_t=S, policy_t_s=Y_best_allocation)

                print(
                    f"{t=} {len(self.V.rewards[t])=}\t{len(self.V.count[t])=}\t {len(self.V.policies[t])=}"
                )
                reward_diff_sum += abs(reward_s_t - reward)

            print(f"{reward_diff_sum=}")
            # 更新策略

        # 找到最优状态
        S_opt = max(self.V.rewards[0], key=self.V.get_reward)

        return S_opt, self.V

    def transition(self, S, Y_best_allocation, Xi_arriving_tasks):
        # print("transiting")
        """
        状态转移函数。

        参数:
        S (tuple): 当前状态。
        Y_best_allocation (list): 当前决策。 # m, i[1,I_citys] l[0,L_max]
        Xi_arriving_tasks (np.array): 每个时间段新增任务的数量矩阵。

        返回:
        tuple: 下一个状态。
        """
        n_il, servers_info = S
        task_count = self.count_tasks(Y_best_allocation)
        n_il_new = n_il + Xi_arriving_tasks - task_count
        servers_info_new = []
        M_servers = self.M_servers

        for m in range(len(servers_info)):
            i, w = servers_info[m]
            if w > 0:
                servers_info_new.append(
                    (Y_best_allocation[m][1], w - 1)
                )  # 0 1 2  m i l
            else:
                servers_info_new.append((self.H_home_of_server[m], self.W_workdays))
        return (n_il_new, servers_info_new)

    def count_tasks(self, Y_best_allocation):
        """
        统计决策Y中完成的任务数量。

        参数:
        Y_best_allocation (list): 决策。tuple (m, i, l) # m, i[1,I_citys] l[0,L_max]

        返回:
        np.array: 每个城市每个等级完成的任务数量。
        """
        # 将 Y_best_allocation 转换为 NumPy 数组
        Y_array = np.array(Y_best_allocation)

        # 从 Y_array 中提取城市和等级信息
        cities = Y_array[:, 1] - 1  # 城市索引从 0 开始
        levels = Y_array[:, 2]  # 等级索引从 0 开始

        # 创建一个零数组，形状为 (self.I_citys, self.L_levels)
        counts = np.zeros((self.I_citys, self.L_levels), dtype=int)

        # 统计任务数量
        for i in range(len(Y_best_allocation)):
            city = cities[i]
            level = levels[i]
            if level != 0:  # 去除0 表示 业务员不分配的情况
                counts[city, level - 1] += 1
        return counts


# 创建 Func8 类的实例
func8_instance = Func8()

S_opt, valueFunction = func8_instance.run()
