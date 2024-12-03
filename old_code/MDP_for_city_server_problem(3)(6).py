#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random

import numpy as np


def generate_state(
    I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd
):

    n_il = np.zeros((I_citys, L_levels), dtype=int)
    for i_city in range(I_citys):
        for l_level in range(L_levels):
            n_il[i_city, l_level] = np.random.poisson(lambd[i_city, l_level])
    S0_tasks = n_il

    S1_servers = []
    for m_server in range(M_servers):
        w_m = np.random.randint(0, W_workdays + 1)
        if w_m == W_workdays:
            i_m = H_home_of_server[m_server]
        else:
            i_m = np.random.randint(1, I_citys + 1)

        S1_servers.append((i_m, w_m))

    S = (S0_tasks, S1_servers)

    return S


# In[3]:


I_citys = 26
L_levels = 5
W_workdays = 6
M_servers = 40
x_max_task_num = 2
H_home_of_server = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]
lambd = np.random.rand(I_citys, L_levels)
func1 = generate_state
state = func1(
    I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd
)
print(state)


# In[4]:


def aggreg_state(S, Z_cluster_num, X, M_servers, I_citys, L_levels):
    barM = np.sum([1 for m_server in range(M_servers) if S[1][m_server][1] != 0])
    cluster = split_list(I_citys, Z_cluster_num)
    num_cluster = np.ceil(I_citys / Z_cluster_num).astype(int)
    g = np.zeros(num_cluster)

    for z_cluster in range(num_cluster):
        e_z = np.sum(
            [
                1
                for m_server in range(M_servers)
                if S[1][m_server][0] in cluster[z_cluster]
            ]
        )
        if e_z == 0:
            g[z_cluster] = 0
        elif e_z <= barM / num_cluster:
            g[z_cluster] = 1
        else:
            g[z_cluster] = 2

    w = S[1][0][1]

    N = np.zeros((num_cluster, L_levels))
    for z_cluster in range(num_cluster):
        for l in range(L_levels):
            N[z_cluster][l] = min(
                X, np.sum([S[0][i - 1][l] for i in cluster[z_cluster]])
            )

    barS = (N, g, w)
    return barS


def split_list(I_citys, Z_cluster_num) -> list:
    arr_city_idx = list(range(1, I_citys + 1))
    return split_array_given_array(arr_city_idx, Z_cluster_num)


def split_array_given_array(arr_city_idx, Z_cluster_num) -> list:

    result = []
    quotient = len(arr_city_idx) // Z_cluster_num
    remainder = len(arr_city_idx) % Z_cluster_num

    for i in range(quotient):
        sub_array = arr_city_idx[i * Z_cluster_num : (i + 1) * Z_cluster_num]
        result.append(sub_array)

    if remainder > 0:
        sub_array = arr_city_idx[-remainder:]
        result.append(sub_array)

    return result


def divide_reminder(num, divisor):

    quotient = num // divisor  # 计算整除的商
    remainder = num % divisor

    if remainder > 0:
        quotient += 1  # 余数大于0，商加一

    return quotient


# In[5]:


Z_cluster_num = 3
X = 3
M_servers = 40
I_citys = 26
L_levels = 5
barS = aggreg_state(state, Z_cluster_num, X, M_servers, I_citys, L_levels)
print(barS)


# In[6]:


def func4(S, L_server):
    L_levels = 5
    N_1 = [
        sum(1 for l_m, (i_m, w_m) in zip(L_server, S[1]) if l_m == j and w_m != 0)
        for j in range(1, L_levels + 1)
    ]
    N_2 = [sum(S[0][i][j] for i in range(len(S[0]))) for j in range(L_levels)]
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


# In[7]:


S = state
L_server = [
    1,
    1,
    1,
    2,
    3,
    4,
    5,
    2,
    3,
    5,
    1,
    1,
    1,
    2,
    3,
    4,
    5,
    2,
    3,
    5,
    1,
    1,
    1,
    2,
    3,
    4,
    5,
    2,
    3,
    5,
    1,
    1,
    1,
    2,
    3,
    4,
    5,
    2,
    3,
    5,
]
mathcal_L, N_1, N_2 = func4(S, L_server)
print(mathcal_L, N_1, N_2)


# In[8]:


len(S[0]), len(S[1])


# In[9]:


import pulp


def func3(S, L_server, H_home_of_server, r1, c1, c2):

    n_il, servers_info = S

    M_servers = len(servers_info)  # 服务员数量
    I_citys = len(n_il)  # 城市数量
    L_max = max(L_server)  # 最大等级

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

    prob += pulp.lpSum(
        r1[l] * y[m, i, l] - c1[servers_info[m][0] - 1][i] * y[m, i, l]
        for m in range(M_servers)
        for i in range(I_citys)
        for l in range(0, L_max + 1)
    ) - c2 * pulp.lpSum(
        n_il[i][l - 1] - pulp.lpSum(y[m, i, l] for m in range(M_servers))
        for i in range(I_citys)
        for l in range(1, L_max + 1)
    )

    for m in range(M_servers):
        prob += (
            pulp.lpSum(y[m, i, l] for i in range(I_citys) for l in range(L_max + 1))
            == 1
        )

    for m, (im, wm) in enumerate(servers_info):
        if wm == 0:
            prob += y[m, H_home_of_server[m], 0] == 1

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


# In[10]:


S = state
n_il, servers_info = S
H_home_of_server = H_home_of_server
r1 = [0, 3500, 3000, 2500, 2000, 1500]
I_citys = len(n_il)

c1 = [
    [0 if i == j else random.randint(100, 500) for j in range(I_citys)]
    for i in range(I_citys)
]
c2 = 100
A = func3(S, L_server, H_home_of_server, r1, c1, c2)


# In[11]:


len(A), A


# In[12]:


import pulp


def func5(S, mathcal_L, L_server, N_1, N_2, H_home_of_server, r1, c1, c2):
    """
    生成状态 S 到决策 Y 的函数,通过解决线性规划问题来最大化收益 R(S, Y)。

    参数:
    S (tuple): 当前状态,包含任务矩阵和服务员信息。
    mathcal_L (list): 分类后的等级列表。
    mathscr_L (list): 所有服务员的等级列表
    N_1 (list): 每个等级的服务员数量。
    N_2 (list): 每个等级的任务数量。
    H_home_of_server (list): 服务员的家位置列表。
    r1 (list): 每个等级的收益列表。
    c1 (list of list): I×I 的成本矩阵。
    c2 (float): 常数成本。

    返回:
    list: 最优决策 Y,包含每个服务员的位置和等级。
    total_revenue, 总收益
    """
    n_il, servers_info = S
    M_servers = len(servers_info)  # 服务员数量
    I_citys = len(n_il)  # 城市数量
    L_max = [max(l) for l in mathcal_L]  # 最大等级
    # print("M_servers, I_citys, L_max, N_1, N_2, mathcal_L, mathscr_L", M_servers, I_citys, L_max, N_1, N_2, mathcal_L, mathscr_L)
    # 步骤1:安排放假的员工回家
    C_h = sum(
        c1[servers_info[m][0] - 1][H_home_of_server[m]]
        for m in range(M_servers)
        if servers_info[m][1] == 0
    )

    total_revenue = -C_h  # 初始化总收益为负的回家成本

    Y = [None] * M_servers  # 初始化最优决策 Y

    Y_set = []
    # 步骤2:对每个等级类独立进行员工分配
    for L_set, l_max_L in zip(mathcal_L, L_max):
        # print("L_set ", L_set)
        M_servers_L = [
            m
            for m in range(M_servers)
            if servers_info[m][1] > 0 and mathscr_L[m] in L_set
        ]  # 该等级类下工作的员工集合
        I_citys_L = [
            i for i in range(I_citys) if any(n_il[i][l - 1] > 0 for l in L_set)
        ]  # 该等级类下有任务需求的城市集合
        # print("M_servers_L, I_citys_L", M_servers_L, I_citys_L)
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

        # 目标函数
        if sum(N_1[l - 1] for l in L_set) <= sum(
            N_2[l - 1] for l in L_set
        ):  # 等级类型为"≤"
            prob += pulp.lpSum(
                r1[l1 - 1] * y[m1, i1, l1]
                - c1[servers_info[m1][0] - 1][i1] * y[m1, i1, l1]
                for m1 in M_servers_L
                for i1 in I_citys_L
                for l1 in L_set
            ) - c2 * pulp.lpSum(
                n_il[i][l - 1] - pulp.lpSum(y[m, i, l] for m in M_servers_L)
                for i in I_citys_L
                for l in L_set
            )
        else:  # 等级类型为">"
            prob += pulp.lpSum(
                r1[l1 - 1] * y[m1, i1, l1]
                - c1[servers_info[m1][0] - 1][i1] * y[m1, i1, l1]
                for m1 in M_servers_L
                for i1 in I_citys_L
                for l1 in L_set
            )

        # 添加约束
        for m in M_servers_L:
            # 每个工作中的服务员 m,要求其被分配到城市 i 提供的服务等级 l 必须不低于他自身的服务等级 L_mathscr[m]
            # 且只能被分配到一个城市提供一种等级的服务。
            prob += (
                pulp.lpSum(
                    y[m, i, l] for i in I_citys_L for l in L_set if l >= mathscr_L[m]
                )
                == 1
            )

        for i in I_citys_L:
            for l in L_set:
                if sum(N_1[l - 1] for l in L_set) <= sum(
                    N_2[l - 1] for l in L_set
                ):  # 等级类型为"≤"
                    prob += (
                        pulp.lpSum(y[m, i, l] for m in M_servers_L) <= n_il[i][l - 1]
                    )
                else:  # 等级类型为">"
                    prob += (
                        pulp.lpSum(y[m, i, l] for m in M_servers_L) == n_il[i][l - 1]
                    )

        # 求解问题
        prob.solve()

        for m in M_servers_L:
            for i in I_citys_L:
                for l in L_set:
                    Y_sub_set = []
                    if pulp.value(y[m, i, l]) == 1:
                        Y[m] = (m, i + 1, l)  # 城市编号从1开始
                        break

        # 更新总收益
        total_revenue += pulp.value(prob.objective)

        # 提取结果
        # Y_L = [(i, l) for m in M_servers_L for i in I_citys_L for l in L_set if y[m, i, l].value() == 1]
        # Y_set.append(Y_L)

    # 步骤3:安排放假的员工
    for m in range(M_servers):
        if servers_info[m][1] == 0:
            Y[m] = (m, H_home_of_server[m] + 1, 0)  # 城市编号从1开始
    # # 步骤3:计算总收益
    # R = sum(prob.objective.value() for L in L_set) - C_h
    #     Y_set = []
    # 解析结果

    return Y, total_revenue


#

# In[13]:


Y, total_revenue = func5(S, mathcal_L, L_server, N_1, N_2, H_home_of_server, r1, c1, c2)
print(Y, total_revenue)


# In[14]:


def func6(S, mathcal_L, L_server, N_1, N_2):
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
                if L_server[m] in L_set:  # 如果服务员 m 的等级属于当前等级类
                    for i in range(I_citys):  # 遍历每个城市
                        for l in L_set:  # 遍历当前等级类的每个等级
                            if l >= L_server[m]:  # 如果当前等级不低于服务员 m 的等级
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


# In[15]:


A_all = func6(S, mathcal_L, L_server, N_1, N_2)
print(A_all)


# In[16]:


import numpy as np


def func7(T, x_max_task_num, lambda_il):
    # 生成了每日新到达的任务?
    # T: 表示时间周期，例如天数
    # x_max_task_num: 矩阵元素的最大取值
    # lambda_il: 泊松分布的率参数矩阵 (I_citys x_max_task_num L_levels)

    # 获取 lambda_il 的维度为 I_citys 和 L_levels
    I_citys, L_levels = lambda_il.shape

    # 初始化三维数组
    arriving_tasks_i = np.zeros((T, I_citys, L_levels), dtype=int)

    # 生成每个时间步的 I_citys x_max_task_num L_levels 矩阵
    for t in range(T):
        for i in range(I_citys):
            for l in range(L_levels):
                # 使用泊松分布生成矩阵元素
                arriving_tasks_i[t, i, l] = min(
                    np.random.poisson(lambda_il[i, l]), x_max_task_num
                )

    return arriving_tasks_i


# In[17]:


# 示例参数
T = 7  # 时间步数量
x_max_task_num = 3  # 最大值
I_citys = 40  # 城市数量
L_levels = 5  # 等级数量
lambda_il = np.random.rand(I_citys, L_levels)  # 生成率参数矩阵

# 生成arriving_tasks_i
arriving_tasks_i = func7(T, x_max_task_num, lambda_il)
print(arriving_tasks_i.shape, arriving_tasks_i)


# In[ ]:


# In[18]:


func1 = generate_state
func2 = aggreg_state
func3 = func3
func4 = func4
func5 = func5
func6 = func6
func7 = func7


# In[62]:


# 示例参数
T = 7  # 时间步数量
x_max_task_num = 3  # 最大值
I_citys = 26  # 城市数量
L_levels = 5  # 等级数量
lambda_il = np.random.rand(I_citys, L_levels)  # 生成率参数矩阵

# 生成arriving_tasks_i
arriving_tasks_i = func7(T, x_max_task_num, lambda_il)
print(arriving_tasks_i.shape, arriving_tasks_i)


# In[112]:


import pulp


def lp_aggreg(S, Z_cluster_num, X, M_servers, I_citys, L_levels):
    barM = np.sum([1 for m_server in range(M_servers) if S[1][m_server][1] != 0])
    cluster = split_list(I_citys, Z_cluster_num)
    num_cluster = np.ceil(I_citys / Z_cluster_num)
    g = [pulp.LpVariable(f"v_{i}", cat="Integer") for i in range(num_cluster)]

    for z_cluster in range(num_cluster):
        e_z = pulp.lpSum(
            [
                1
                for m_server in range(M_servers)
                if S[1][m_server][0] in cluster[z_cluster]
            ]
        )
        if e_z == 0:
            g[z_cluster] = 0
        elif e_z <= barM / num_cluster:
            g[z_cluster] = 1
        else:
            g[z_cluster] = 2

    w = S[1][0][1]

    N = [
        [pulp.LpVariable(f"k_{i}", cat="Integer") for i in range(num_cluster)]
        for j in range(L_levels)
    ]

    for z_cluster in range(num_cluster):
        for l in range(L_levels):
            N[z_cluster][l] = min(
                X, pulp.lpSum([S[0][i - 1][l] for i in cluster[z_cluster]])
            )

    barS = (N, g, w)
    return barS


def split_list(I_citys, Z_cluster_num) -> list:
    # split_list(10, 3)=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    arr_city_idx = list(range(1, I_citys + 1))
    return split_array_given_array(arr_city_idx, Z_cluster_num)


def split_array_given_array(arr_city_idx, Z_cluster_num) -> list:
    # split_array_given_array([1:10],3)= [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    result = []
    quotient = len(arr_city_idx) // Z_cluster_num
    remainder = len(arr_city_idx) % Z_cluster_num

    for i in range(quotient):
        sub_array = arr_city_idx[i * Z_cluster_num : (i + 1) * Z_cluster_num]
        result.append(sub_array)

    if remainder > 0:
        sub_array = arr_city_idx[-remainder:]
        result.append(sub_array)

    return result


def divide_reminder(num, divisor):  # divide_reminder(10, 3) = 10/3+1 =4

    quotient = num // divisor  # 计算整除的商
    remainder = num % divisor

    if remainder > 0:
        quotient += 1  # 余数大于0，商加一

    return quotient


# In[113]:


num_cluster = np.ceil(9 / 3)
num_cluster


# In[114]:


Z_cluster_num = 3
X = 3
M_servers = 40
I_citys = 26
L_levels = 5
barS = aggreg_state(state, Z_cluster_num, X, M_servers, I_citys, L_levels)
print(barS)


# In[121]:


def V(S):
    one = S[1][1][1]
    return one


# In[122]:


V(S)


# In[123]:


import pulp


def func3_transfer(S, L_server, H_home_of_server, r1, c1, c2):

    n_il, servers_info = S

    M_servers = len(servers_info)  # 服务员数量
    I_citys = len(n_il)  # 城市数量
    L_max = max(L_server)  # 最大等级
    n_next = [
        [pulp.LpVariable(f"x_{i}_{j}", cat="Integer") for j in range(L_max + 1)]
        for i in range(I_citys)
    ]
    ser_info_1 = [pulp.LpVariable(f"u_{i}", cat="Integer") for i in range(M_servers)]
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
                + func7(T, x_max_task_num, lambda_il)
            )
    for m in range(M_servers):
        ser_info_1[m] = pulp.lpSum(
            i * pulp.lpSum(y[m, i, l] for l in range(0, L_max + 1))
            for i in range(I_citys)
        )
    for m in range(M_servers):
        if servers_info[m][1] == 0:
            ser_info_2[m] = W_workdays
        else:
            ser_info_2[m] = servers_info[m][1] - 1
    ser_info_next = list(zip(ser_info_1, ser_info_2))
    S_next = (n_next, ser_info_next)

    # print(S_next[1][39][1])
    # S_next_agg = lp_aggreg(S_next, Z_cluster_num, X, M_servers, I_citys, L_levels)

    prob += (
        pulp.lpSum(
            r1[l] * y[m, i, l] - c1[servers_info[m][0] - 1][i] * y[m, i, l]
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
        + V(S_next)
    )

    for m in range(M_servers):
        prob += (
            pulp.lpSum(y[m, i, l] for i in range(I_citys) for l in range(L_max + 1))
            == 1
        )

    for m, (im, wm) in enumerate(servers_info):
        if wm == 0:
            prob += y[m, H_home_of_server[m], 0] == 1

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


# In[124]:


S = state
n_il, servers_info = S
H_home_of_server = H_home_of_server
r1 = [0, 3500, 3000, 2500, 2000, 1500]
I_citys = len(n_il)


c1 = [
    [0 if i == j else random.randint(100, 500) for j in range(I_citys)]
    for i in range(I_citys)
]
c2 = 100
A = func3_transfer(S, L_server, H_home_of_server, r1, c1, c2)


# In[125]:


A


# In[83]:


# 函数8
import numpy as np

for j in range(J):
    s0 = func1(
        I_citys,
        L_levels,
        W_workdays,
        M_servers,
        x_max_task_num,
        H_home_of_server,
        lambd,
    )
    xi = func7(T, x_max_task_num, lambda_il)
    R = [0] * T
    for t in range(T):
        action = func3(s0, L_mathscr, H_home_of_server, r1, c1, c2)
        R[t] = revenue(s0, action)
        s1 = state_trans(s0, action, xi[t])


def revenue(st, at):
    pass


def state_trans(S, act, xi, home, W):  # 状态转移
    dic1 = {}
    for i, row in enumerate(S[0]):
        for j, value in enumerate(row):
            dic1[(i + 1, j + 1)] = value
    dic2 = {(x[1], x[2]): 1 for x in act if x[2] != 0}
    S_A_cell = {}
    for key in dic1:
        if key in dic2:
            S_A_cell[key] = dic1[key] - dic2[key]
        else:
            S_A_cell[key] = dic1[key]
    S_A = [[0] * len(S[0][0]) for _ in range(len(S0))]
    for key, value in S_A_cell.items():
        S_A[key[0] - 1][key[1] - 1] = value

    result = np.add(S_A, xi)
    return result


# In[95]:


import numpy as np


def state_trans(S, act, xi, home, W):
    dic1 = {}
    for i, row in enumerate(S[0]):
        for j, value in enumerate(row):
            dic1[(i + 1, j + 1)] = value  # dic1是将任务矩阵写成字典形式
    dic2 = {(x[1], x[2]): 1 for x in act if x[2] != 0}  # dic2是将决策A=(m,i,l)写成字典
    S_A_cell = {}
    for key in dic1:
        if key in dic2:
            S_A_cell[key] = dic1[key] - dic2[key]
        else:
            S_A_cell[key] = dic1[key]
    S_A = [[0] * len(S[0][0]) for _ in range(len(S[0]))]
    for key, value in S_A_cell.items():
        S_A[key[0] - 1][key[1] - 1] = value  # S_A 是S-A之后的值
    S_A_XI_1 = np.add(S_A, xi)  # S_A_XI_1 是S-A+XI之后的值S[0]

    m_loc = [0] * len(act)
    m_w = [0] * len(act)
    for x in act:
        if x[2] == 0:
            m_loc[x[0]] = home[x[0]]
        else:
            m_loc[x[0]] = x[1]

    for j in range(len(S[1])):
        if S[1][j][1] == 0:
            m_w[j] = W
        else:
            m_w[j] = S[1][j][1] - 1

    S_A_XI_2 = list(zip(m_loc, m_w))  # S_A_XI_2 是S-A+XI之后的值S[1]

    S_A_XI = (S_A_XI_1, S_A_XI_2)

    return S_A_XI


# In[116]:


S = state
act = A
xi = arriving_tasks_i[0]
home = H_home_of_server
W = W_workdays
tra = state_trans(S, act, xi, home, W)
# print(state)
# print(A)
print(tra)
s_2 = aggreg_state(tra, Z_cluster_num, X, M_servers, I_citys, L_levels)
s_2


# In[120]:


import pulp


def func8_add_v(S, L_server, home, r1, c1, c2):

    n_il, servers_info = S
    M_servers = len(servers_info)  # 服务员数量
    I_citys = len(n_il)  # 城市数量
    L_max = max(L_server)  # 最大等级

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

    prob += pulp.lpSum(
        r1[l] * y[m, i, l] - c1[servers_info[m][0] - 1][i] * y[m, i, l]
        for m in range(M_servers)
        for i in range(I_citys)
        for l in range(0, L_max + 1)
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
                    for l in range(L_mathscr[m], L_max + 1)
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
    print(y)
    return result


# In[121]:


func8_add_v(S, L_server, home, r1, c1, c2)


# In[122]:


print(y)


# In[ ]:
