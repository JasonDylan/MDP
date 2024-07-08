# %%
import logging
import copy
import csv
import json
import pickle
import random
import time
from functools import wraps

import numpy as np
import pandas as pd
import pulp

# 装饰器用于计算函数调用时间


class SValue:
    def __init__(self, T):
        self.s_values = {}
        self.T = T

    def _convert_to_tuple(self, S_t):
        """
        将状态转换为NumPy数组，并返回转换后的状态。
        """
        if type(S_t) == str:
            S_t_str = S_t
        else:
            S_t_str = str(S_t)
        # caller_frame = inspect.currentframe().f_back
        # caller_name = caller_frame.f_code.co_name
        # # logging.info(f"{caller_name=} {S_t_str=}")
        return S_t_str

    def init_s_value_t(self, t, S_agg):
        if t not in self.s_values:
            self.s_values[t] = {}
        tuple_s_agg = self._convert_to_tuple(S_agg)
        self.s_values[t][tuple_s_agg] = {"count": 0, "total_reward": 0}

    def get_s_count(self, t, S_agg):
        if t in self.s_values:
            tuple_s_agg = self._convert_to_tuple(S_agg)
            if tuple_s_agg in self.s_values[t]:
                return self.s_values[t][tuple_s_agg]["count"]
            else:
                self.init_s_value_t(t, S_agg)
        else:
            self.init_s_value_t(t, S_agg)
        return 0

    def get_total_reward(self, t, S_agg):
        if t in self.s_values:
            tuple_s_agg = self._convert_to_tuple(S_agg)
            if tuple_s_agg in self.s_values[t]:
                return self.s_values[t][tuple_s_agg]["total_reward"]
            else:
                self.init_s_value_t(t, S_agg)
        else:
            self.init_s_value_t(t, S_agg)
        return 0

    def update_total_rewards(self, total_reward):
        for t in self.s_values:
            for S_agg in self.s_values[t]:
                if S_agg in self.s_values[t]:
                    count = self.get_s_count(t, S_agg)
                    old_value = self.get_total_reward(t, S_agg)
                    weight = 1 / (count + 1)
                    new_value = old_value + weight * (total_reward[t] - old_value)
                    self.s_values[t][S_agg]["count"] = count + 1
                    self.s_values[t][S_agg]["total_reward"] = new_value
                else:
                    self.init_s_value_t(t, S_agg)

    def get_s_values(self):
        return self.s_values


class TaskAllocationProblem:
    def __init__(
        self,
        I_citys,
        L_levels,
        W_workdays,
        M_servers,
        x_max_task_num,
        H_home_of_server,
        lambd,
        T,
        lambda_il,
        L_server,
        r1,
        c1,
        c2,
    ):
        self.I_citys = I_citys
        self.L_levels = L_levels
        self.W_workdays = W_workdays
        self.M_servers = M_servers
        self.x_max_task_num = x_max_task_num
        self.H_home_of_server = H_home_of_server
        self.lambd = lambd
        self.T = T
        self.lambda_il = lambda_il
        self.L_server = L_server
        self.r1 = r1
        self.c1 = c1
        self.c2 = c2

    
        
    def func1(self):
        # generate_state(I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd):

        n_il = np.zeros((self.I_citys, self.L_levels), dtype=int)
        for i_city in range(self.I_citys):
            for l_level in range(self.L_levels):
                n_il[i_city, l_level] = np.random.poisson(self.lambd[i_city, l_level])
        S0_tasks = n_il

        S1_servers = []
        for m_server in range(
            self.M_servers
        ):  # W_workdays =6 ? 主要看哪里用到了W_workdays在判断是否要+1 或者改为7
            w_m = np.random.randint(0, self.W_workdays)
            if w_m == self.W_workdays:
                i_m = self.H_home_of_server[m_server]
            else:
                i_m = np.random.randint(1, self.I_citys) - 1

            S1_servers.append((i_m, w_m))

        S = (S0_tasks, S1_servers)

        return S

    def save_to_csv(self, S, tasks_csv_path, servers_csv_path):
        # Save S0_tasks to tasks_csv_path
        df_tasks = pd.DataFrame(S[0])
        df_tasks.to_csv(tasks_csv_path, index=False, header=False)

        # Save S1_servers to servers_csv_path
        df_servers = pd.DataFrame(S[1], columns=["City", "Workday_left"])
        df_servers.to_csv(servers_csv_path, index=False)

    def save_to_one_csv(self, S, csv_path):
        # Convert S0_tasks to DataFrame
        df_tasks = pd.DataFrame(
            S[0], columns=[f"Level_{i}" for i in range(S[0].shape[1])]
        )

        # Convert S1_servers to DataFrame
        df_servers = pd.DataFrame(S[1], columns=["City", "Workday_left"])

        # Create an empty DataFrame with the same number of rows as the larger DataFrame
        max_len = max(len(df_tasks), len(df_servers))
        empty_df = pd.DataFrame(np.nan, index=range(max_len), columns=[""])

        # Padding df_tasks and df_servers with NaN to match the length if needed
        df_tasks = df_tasks.reindex(range(max_len))
        df_servers = df_servers.reindex(range(max_len))

        # Combine DataFrames with an empty column in between
        df_combined = pd.concat([df_tasks, empty_df, df_servers], axis=1)

        # Save combined DataFrame to CSV
        df_combined.to_csv(csv_path, index=False)

    def split_list(self, I_citys, Z_cluster_num) -> list:
        arr_city_idx = list(range(1, I_citys + 1))  # [1, I_citys]
        return self.split_array_given_array(arr_city_idx, Z_cluster_num)

    def split_array_given_array(self, arr_city_idx, Z_cluster_num) -> list:

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

    def func2(self, S, Z_cluster_num, X):
        # aggreg_state(S, Z_cluster_num, X, M_servers, I_citys, L_levels):
        barM = np.sum(
            [1 for m_server in range(self.M_servers) if S[1][m_server][1] != 0]
        )
        cluster = self.split_list(self.I_citys, Z_cluster_num)
        num_cluster = np.ceil(self.I_citys / Z_cluster_num).astype(int)
        g = np.zeros(num_cluster)

        for z_cluster in range(num_cluster):
            e_z = np.sum(
                [
                    1
                    for m_server in range(self.M_servers)
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

        N = np.zeros((num_cluster, self.L_levels))
        for z_cluster in range(num_cluster):
            for l in range(self.L_levels):
                N[z_cluster][l] = min(
                    X,
                    np.sum(
                        [S[0][i - 1][l] for i in cluster[z_cluster]]
                    ),  # todo 这里i是否应该-1？
                )

        barS = (N, g, w)
        return barS

    def func3_transfer(self, S, V):
        # def func3_transfer(S, L_server, H_home_of_server, r1, c1, c2, V):

        n_il, servers_info = S

        # M_servers = len(servers_info)  # 服务员数量
        # I_citys = len(n_il)  # 城市数量
        M_servers = self.M_servers  # 服务员数量
        I_citys = self.I_citys  # 城市数量
        L_server = self.L_server
        n_next = [
            [
                pulp.LpVariable(f"x_{i}_{j}", cat="Integer")
                for j in range(self.L_levels + 1)
            ]
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
                for l in range(self.L_levels + 1)
            ),
            cat=pulp.LpBinary,
        )

        for i in range(I_citys):
            for l in range(self.L_levels):
                n_next[i][l] = (
                    n_il[i][l]
                    - pulp.lpSum(y[m, i, l + 1] for m in range(M_servers))
                    + self.task_arr
                )
        for m in range(M_servers):
            ser_info_1[m] = pulp.lpSum(
                i * pulp.lpSum(y[m, i, l] for l in range(0, self.L_levels + 1))
                for i in range(I_citys)
            )
        for m in range(M_servers):
            if servers_info[m][1] == 0:
                ser_info_2[m] = self.W_workdays
            else:
                ser_info_2[m] = servers_info[m][1] - 1
        ser_info_next = list(zip(ser_info_1, ser_info_2))
        S_next = (n_next, ser_info_next)

        # logging.info(S_next[1][39][1])
        # S_next_agg = lp_aggreg(S_next, Z_cluster_num, X, M_servers, I_citys, L_levels)

        prob += (
            pulp.lpSum(
                self.r1[l] * y[m, i, l] - self.c1[servers_info[m][0]][i] * y[m, i, l]
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(0, self.L_levels + 1)
            )
            - self.c2
            * pulp.lpSum(
                n_il[i][l - 1] - pulp.lpSum(y[m, i, l] for m in range(M_servers))
                for i in range(I_citys)
                for l in range(1, self.L_levels + 1)
            )
            + V
        )

        # 每个服务员必须且只能被分配到一个任务，无论是在某个城市执行任务还是在家休息。 l 等于0 表示在家休息
        for m in range(M_servers):
            prob += (
                pulp.lpSum(
                    y[m, i, l] for i in range(I_citys) for l in range(self.L_levels + 1)
                )
                == 1
            )

        # 每个城市 i 每个等级 l 被分配的服务员数量不能超过该等级的任务数量
        for i in range(self.I_citys):
            for l in range(1, self.L_levels + 1):
                prob += (
                    pulp.lpSum(y[m, i, l] for m in range(self.M_servers))
                    <= n_il[i][l - 1]
                )

        # 分配给所有服务员的所有任务总数不能超过实际的任务总数。
        total_tasks = sum(
            n_il[i][l - 1] for i in range(I_citys) for l in range(1, self.L_levels + 1)
        )
        prob += (
            pulp.lpSum(
                y[m, i, l]
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(1, self.L_levels + 1)
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
                for l in range(1, self.L_levels + 1)
            )
            <= available_servers
        )

        for m, (im, wm) in enumerate(servers_info):
            if wm == 0:
                try:
                    prob += y[m, self.H_home_of_server[m], 0] == 1
                except Exception as ex:
                    logging.info(f"{ex=} {m=} {self.H_home_of_server=} {self.L_server=} {S=}")
                    raise ex
            elif wm > 0:

                # 服务员必须分配任务，且任务等级必须高于服务员等级
                # 1. 允许选择 "待在家中" (l = 0)
                prob += (
                    pulp.lpSum(
                        y[m, i, l]
                        for i in range(I_citys)
                        for l in range(self.L_levels + 1)
                        if (l == 0 and i == self.H_home_of_server[m])
                        or (l > 0 and l >= L_server[m] and n_il[i][l - 1] > 0)
                    )
                    == 1
                )

        # 限制了服务员只能接受等级不高于自身等级的任务
        for m in range(M_servers):
            for i in range(I_citys):
                for l in range(1, self.L_levels + 1):
                    prob += y[m, i, l] <= (L_server[m] >= l)

        # 求解问题
        # 选择求解器，例如：CBC
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        # 解析结果
        A = [
            (m, i, l)
            for m in range(M_servers)
            for i in range(I_citys)
            for l in range(self.L_levels + 1)
            if pulp.value(y[m, i, l]) == 1
        ]

        return A

    def func4(self, S):
        # func4(S, L_server):
        N_1 = [
            sum(
                1
                for l_m, (i_m, w_m) in zip(self.L_server, S[1])
                if l_m == j and w_m != 0
            )
            for j in range(1, self.L_levels + 1)
        ]
        N_2 = [sum(S[0][i][j] for i in range(len(S[0]))) for j in range(self.L_levels)]
        mathcal_L = []
        current_class = []
        total_N_1 = 0
        total_N_2 = 0

        for j in range(1, self.L_levels + 1):
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

    def func5(self, S, mathcal_L, N_1, N_2):
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
        total_reward, 总收益
        """
        n_il, servers_info = S

        M_servers = len(servers_info)  # 服务员数量
        I_citys = len(n_il)  # 城市数量
        # M_servers = self.M_servers  # 服务员数量
        # I_citys = self.I_citys  # 城市数量
        H_home_of_server = self.H_home_of_server
        L_server = self.L_server
        c1 = self.c1
        c2 = self.c2
        r1 = self.r1
        L_max_set = [max(l) for l in mathcal_L]  # 最大等级
        # logging.info("M_servers, I_citys, self.L_levels, N_1, N_2, mathcal_L, mathscr_L", M_servers, I_citys, L_max_set, N_1, N_2, mathcal_L, mathscr_L)
        # 步骤1:安排放假的员工回家
        C_h = sum(
            c1[servers_info[m][0]][H_home_of_server[m] - 1]
            for m in range(M_servers)
            if servers_info[m][1] == 0
        )

        total_reward = -C_h  # 初始化总收益为负的回家成本

        Y = [None] * M_servers  # 初始化最优决策 Y

        Y_set = []
        # 步骤2:对每个等级类独立进行员工分配
        for L_set, l_max_L in zip(mathcal_L, L_max_set):
            # logging.info("L_set ", L_set)
            M_servers_L = [
                m
                for m in range(M_servers)
                if servers_info[m][1] > 0 and L_server[m] in L_set
            ]  # 该等级类下工作的员工集合
            I_citys_L = [
                i for i in range(I_citys) if any(n_il[i][l - 1] > 0 for l in L_set)
            ]  # 该等级类下有任务需求的城市集合
            # logging.info("M_servers_L, I_citys_L", M_servers_L, I_citys_L)
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
                    - c1[servers_info[m1][0]][i1] * y[m1, i1, l1]
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
                    - c1[servers_info[m1][0]][i1] * y[m1, i1, l1]
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
                        y[m, i, l] for i in I_citys_L for l in L_set if l >= L_server[m]
                    )
                    == 1
                )

            for i in I_citys_L:
                for l in L_set:
                    if sum(N_1[l - 1] for l in L_set) <= sum(
                        N_2[l - 1] for l in L_set
                    ):  # 等级类型为"≤"
                        prob += (
                            pulp.lpSum(y[m, i, l] for m in M_servers_L)
                            <= n_il[i][l - 1]
                        )
                    else:  # 等级类型为">"
                        prob += (
                            pulp.lpSum(y[m, i, l] for m in M_servers_L)
                            == n_il[i][l - 1]
                        )

            # 求解问题
            status = prob.solve()
            if status != pulp.LpStatusOptimal:
                raise ValueError(
                    f"Linear programming problem did not find an optimal solution. for{L_set=} {l_max_L}"
                )

            for m in M_servers_L:
                for i in I_citys_L:
                    for l in L_set:
                        Y_sub_set = []
                        if pulp.value(y[m, i, l]) == 1:
                            Y[m] = (m, i, l)  # 城市编号从1开始
                            # break

            # 提取结果
            # Y_L = [(i, l) for m in M_servers_L for i in I_citys_L for l in L_set if y[m, i, l].value() == 1]
            # Y_set.append(Y_L)

        # 步骤3:安排放假的员工
        for m in range(M_servers):
            if servers_info[m][1] == 0:
                Y[m] = (m, H_home_of_server[m], 0)  # 城市编号从1开始
        # # 步骤3:计算总收益
        # R = sum(prob.objective.value() for L in L_set) - C_h
        #     Y_set = []
        # 解析结果

        return Y

    def func6(self, S, mathcal_L, N_1, N_2):
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

        L_server = self.L_server
        A = [[] for _ in range(M_servers)]  # 初始化决策空间 A
        A_mil = A

        # 对每个服务员生成可能的决策
        for m in range(M_servers):
            if servers_info[m][1] == 0:  # 服务员 m 放假
                A[m].append(
                    (m, servers_info[m][0], 0)
                )  # 放假的服务员只有一个决策,即回家
            else:  # 服务员 m 工作
                for L_set in mathcal_L:  # 遍历每个等级类
                    if L_server[m] in L_set:  # 如果服务员 m 的等级属于当前等级类
                        for i in range(I_citys):  # 遍历每个城市
                            for l in L_set:  # 遍历当前等级类的每个等级
                                if (
                                    l >= L_server[m]
                                ):  # 如果当前等级不低于服务员 m 的等级
                                    if (
                                        sum(N_1[l - 1] for l in L_set)
                                        <= sum(N_2[l - 1] for l in L_set)
                                        and sum(1 for a in A[m] if a[0] == i + 1)
                                        < sum(n_il[i][l - 1] for l in L_set)
                                    ) or (
                                        sum(N_1[l - 1] for l in L_set)
                                        > sum(N_2[l - 1] for l in L_set)
                                        and sum(
                                            1
                                            for a in A[m]
                                            if a[0] == i + 1 and a[1] == l
                                        )
                                        < n_il[i][l - 1]
                                    ):
                                        # 如果满足约束条件,则将决策添加到服务员 m 的决策空间中
                                        A[m].append((m, i + 1, l))  # 城市编号从1开始

        return A

    def state_trans(self, S, act, xi):
        dic1 = {}
        n_il = S[0]
        for i, a_city_level in enumerate(n_il):
            for level_1, task_count in enumerate(a_city_level):
                dic1[(i, level_1 + 1)] = (
                    task_count  # dic1是将任务矩阵写成字典形式 level_1[0, 4]
                )
        dic2 = {
            (x[1], x[2]): 1 for x in act if x[2] != 0
        }  # dic2是将决策A=(m,i,l)写成字典 m业务员分配到城市i完成任务l x[1] = i x[2] = l [1, 5]
        S_A_cell = {}
        for key in dic1:  # 对于有某个等级的任务的城市
            if key in dic2:  # 如果在分配的决策里，则去完成任务
                S_A_cell[key] = dic1[key] - dic2[key]
            else:
                S_A_cell[key] = dic1[key]
        S_A = [[0] * len(S[0][0]) for _ in range(len(S[0]))]
        for key, value in S_A_cell.items():
            S_A[key[0]][key[1] - 1] = value  # S_A 是S-A之后的值 key[0] 是 i key[1] 是 l
        S_A_XI_1 = np.add(S_A, xi)  # S_A_XI_1 是S-A+XI之后的值S[0]

        m_loc = [0] * len(act)
        m_w = [0] * len(act)
        for x in act:
            if x[2] == 0:
                m_loc[x[0]] = self.H_home_of_server[x[0]]
            else:
                m_loc[x[0]] = x[1]

        for j in range(len(S[1])):
            if S[1][j][1] == 0:
                m_w[j] = self.W_workdays
            else:
                m_w[j] = S[1][j][1] - 1

        S_A_XI_2 = list(zip(m_loc, m_w))  # S_A_XI_2 是S-A+XI之后的值S[1]

        S_A_XI = (S_A_XI_1, S_A_XI_2)

        return S_A_XI

    def func7(self, T):
        # 生成了每日新到达的任务?
        # T: 表示时间周期，例如天数
        # x_max_task_num: 矩阵元素的最大取值
        # lambda_il: 泊松分布的率参数矩阵 (I_citys x_max_task_num L_levels)

        # 获取 lambda_il 的维度为 I_citys 和 L_levels
        I_citys, L_levels = self.lambda_il.shape

        # 初始化三维数组
        arriving_tasks_i = np.zeros((T, I_citys, L_levels), dtype=int)

        # 生成每个时间步的 I_citys x_max_task_num L_levels 矩阵
        for t in range(T):
            for i in range(I_citys):
                for l in range(L_levels):
                    # 使用泊松分布生成矩阵元素
                    arriving_tasks_i[t, i, l] = min(
                        np.random.poisson(self.lambda_il[i, l]), self.x_max_task_num
                    )

        return arriving_tasks_i

    def func8_org(self, T=7, J=1000, Z_cluster_num=3):
        X = self.x_max_task_num
        # 目前适配了自动生成每个benchmark相同的到达任务和，初始化J个初始S
        # 初始化s_value列表，用于存储状态的信息
        s_value = []
        # 生成每日新到达的任务
        # self.task_arr = self.func7()
        self.task_arr = self.task_arr

        for j in range(J):
            # 初始化每日的收益列表
            pr = T * [0]
            # 初始化位置列表和转移位置列表
            location = []
            tran_loc = []
            for t in range(T):
                if t == 0:
                    # 生成初始状态S
                    # S = self.func1()
                    S = self.init_S_J[j]
                else:
                    # 使用上一轮的状态作为当前状态S
                    S = S_next
                # 聚合当前状态S，生成聚合状态S_agg
                S_agg = self.func2(S, Z_cluster_num, X)
                if j == 0:
                    # 对于第一轮迭代，将初始状态S_agg添加到s_value列表中
                    s_value.append([t, 1, S_agg, 0])
                    location.append(t)
                    V = 0
                else:
                    len_s_v = len(s_value)
                    for i in range(len_s_v):
                        # 判断当前状态S_agg是否已经存在于s_value列表中
                        if t == s_value[i][0] and all(
                            np.array_equal(a, b) for a, b in zip(S_agg, s_value[i][2])
                        ):
                            # 如果存在，则将该位置添加到位置列表和转移位置列表中，并获取对应的总收益V
                            location.append(i)
                            tran_loc.append(i)
                            V = s_value[i][3]
                            break
                    else:
                        # 如果不存在，则将当前状态S_agg添加到s_value列表中
                        s_value.append([t, 1, S_agg, 0])
                        loc = len(s_value) - 1
                        location.append(loc)
                        V = 0
                # 使用转移决策函数func3_transfer生成决策A
                A = self.func3_transfer(S, V)
                # 计算当前状态S和决策A的收益pr
                pr[t] = self.Profit(S, A)
                # 获取每日的任务到达量
                xi = self.task_arr[t]
                # 根据当前状态S、决策A和任务到达量xi生成下一状态S_next
                S_next = self.state_trans(S, A, xi)
            # 初始化每日的总收益列表
            total_reward = [0] * T
            # 逆序计算每日的总收益
            for t in range(T - 1, -1, -1):
                total_reward[t] += sum(pr[t:])
            logging.info(f"{location=}")
            # 更新s_value中的总收益值
            for i in location:
                i_t = location.index(i)
                if i in tran_loc:
                    # 更新已存在的位置的总收益值
                    count = s_value[i][1]
                    old_value = s_value[i][3]
                    weight = 1 / (count + 1)
                    new_value = old_value + weight * (total_reward[i_t] - old_value)
                    s_value[i][1] = count + 1
                    s_value[i][3] = new_value
                else:
                    # 更新新添加的位置的总收益值
                    s_value[i][3] = total_reward[i_t]

        return s_value

    def func8(self, T=7, J=10000, Z_cluster_num=3):
        X = self.x_max_task_num
        T = T
        s_value = SValue(T)
        Z = Z_cluster_num
        logging.info(f"{(T,Z)=} {len(self.task_arr)=}")
        for j in range(J):
            pr = T * [0]
            for t in range(T):
                if t == 0:
                    # 生成初始状态S
                    S = self.init_S_J[j]
                else:
                    S = S_next
                # 聚合当前状态S，生成聚合状态S_agg
                S_agg = self.func2(S, Z_cluster_num, X)
                logging.info(f"{j=} {t=} {S_agg=}")
                if j == 0:
                    # 对于第一轮迭代，将初始状态S_agg添加到s_value列表中
                    s_value.init_s_value_t(t, S_agg)
                    V = 0
                else:
                    # 判断当前状态S_agg是否已经存在于s_value列表中
                    # 如果存在，则将该位置添加到位置列表和转移位置列表中，并获取对应的总收益V
                    # 如果不存在，则将当前状态S_agg添加到s_value列表中
                    V = s_value.get_total_reward(t, S_agg)
                A = self.func3_transfer(S, V)
                pr[t] = self.Profit(S, A)
                xi = self.task_arr[t]
                S_next = self.state_trans(S, A, xi)
            total_reward = [0] * T
            for t in range(T - 1, -1, -1):
                total_reward[t] = sum(pr[t:])
            s_value.update_total_rewards(total_reward)
            logging.info(f"{s_value.s_values=}")
            len_state = len(s_value.s_values[0].keys())
            logging.info(f"-------{(T,Z)=} {j=} {t=} {len_state=} {total_reward=}------")

        return s_value

    def all_task_init(self, J=10000, T=7):

        # 生成每日新到达的任务
        random.seed(42)
        np.random.seed(42)  # 生成初始状态S
        self.task_arr = self.func7(T)
        random.seed(42)
        np.random.seed(42)  # 生成初始状态S
        self.init_S_J = [self.func1() for j in range(J)]
        logging.info(f"init {self.init_S_J[:5]}")
        for i in range(min(5, J)):
            self.save_to_one_csv(self.init_S_J[i], csv_path=f"init/init_state_{i}_{T}_{J}.csv")

    def calc_total_reward_for_init_S_by_rnd(self, init_S, T=7):
        """
        随机分配策略下的总收益计算。

        参数:
        Xi_arriving_tasks (np.array): 每个时间段新增任务的数量矩阵。
        S_t (tuple): 初始状态。

        返回:
        float: 总收益。
        """
        S_current = init_S
        total_reward = 0
        task_arr = self.task_arr

        # 初始化收益列表
        pr = T * [0]
        save_S = init_S
        for t in range(T):
            Y_allocation = self.generate_random_allocation(
                S_current, self.H_home_of_server, L_server=self.L_server
            )

            R_t = self.Profit(S=S_current, A=Y_allocation)
            total_reward += R_t
            pr[t] = R_t
            
            # 获取当前时间步的任务到达量
            xi = task_arr[t]
            S_next = self.state_trans(S=S_current, act=Y_allocation, xi=xi)
            S_current = S_next
        return save_S, pr

    def generate_random_allocation(self, S, H_home_of_server, L_server):
        """
        生成满足约束条件的随机决策。

        参数:
        S (tuple): 当前状态。
        H_home_of_server (list): 服务员的家位置列表。
        L_server (list): 服务员的等级列表。

        返回:
        list: 随机决策,包含每个服务员的位置和等级。
        """
        n_il, servers_info = copy.deepcopy(S)
        M_servers = len(servers_info)

        allocation = [None] * M_servers

        for m in range(self.M_servers):
            im, wm = servers_info[m]

            if wm == 0:
                allocation[m] = (m, H_home_of_server[m], 0)
            else:
                available_tasks = [
                    (i, l)
                    for i in range(self.I_citys)
                    for l in range(1, self.L_levels + 1)
                    if n_il[i][l - 1] > 0 and l >= L_server[m]
                ]

                if available_tasks:
                    i, l = random.choice(available_tasks)
                    allocation[m] = (m, i, l)
                    n_il[i][l - 1] -= 1
                else:
                    allocation[m] = (m, H_home_of_server[m], 0)

        return allocation

    def math_program_near(self, S, L_server):
        """
        生成最近分配决策。

        参数:
        S (tuple): 当前状态。

        返回:
        list: 最近分配决策,包含每个服务员的位置和等级。
        """
        n_il, servers_info = copy.deepcopy(S)
        M_servers = len(servers_info)
        I_citys = len(n_il)
        L_levels = len(n_il[0])

        allocation = [None] * M_servers

        # 计算每个服务员与每个任务之间的距离
        distances = {
            (m, i, l): self.c1[servers_info[m][0]][i]
            for m in range(M_servers)
            for i in range(I_citys)
            for l in range(1, 1+L_levels)
        }
        for m in range(M_servers):
            im, wm = servers_info[m]

            if wm == 0:
                allocation[m] = (m, self.H_home_of_server[m], 0)
            else:
                # todo 这里其实也是随机，因为业务员时按顺序分配最近的，而不是全局最近的，如果要真的求最近的还是得用pulp改
                available_tasks = [
                    (i, l+1)
                    for i in range(I_citys)
                    for l in range(L_levels)
                    if n_il[i][l] > 0 and l+1 >= L_server[m]
                ]
                # logging.info(f"{available_tasks=}")
                if available_tasks:
                    # 找到距离最近的任务
                    nearest_task = min(
                        available_tasks,
                        key=lambda task: distances[(m, task[0], task[1])],
                    )
                    
                    i_nearest, l_nearest = nearest_task
                    allocation[m] = (m, i_nearest, l_nearest)
                    # logging.info(f"{m=} {nearest_task=}")    
                    # logging.info(f"{allocation[m]=} {n_il[i_nearest]=} {servers_info[m]=}")
                
                    n_il[i_nearest][l_nearest-1] -= 1
                else:
                    allocation[m] = (m, self.H_home_of_server[m], 0)
           
        return allocation

    def nearest_distance(self, init_S, T=7):
        # 获取任务到达量的数组
        task_arr = self.task_arr

        # 初始化收益列表
        pr = T * [0]

        for t in range(T):
            if t == 0:
                # 生成初始状态
                S = init_S
                save_S = S
            else:
                # 使用上一状态的结果作为当前状态
                S = S_next
            # 使用最近距离策略求解决策和收益
            A = self.math_program_near(S, L_server=self.L_server)
            profit = self.Profit(S=S, A=A)
            pr[t] = profit

            # 获取当前时间步的任务到达量
            xi = task_arr[t]

            # 根据当前状态、决策和任务到达量计算下一状态
            S_next = self.state_trans(S, A, xi)

        # 返回初始状态和收益列表
        return save_S, pr

    def math_program_static(self, S, L_server):

        n_il, servers_info = S

        # M_servers = len(servers_info)  # 服务员数量
        # I_citys = len(n_il)  # 城市数量
        M_servers = self.M_servers  # 服务员数量
        I_citys = self.I_citys  # 城市数量

        # 创建问题实例
        prob = pulp.LpProblem("Optimal_Server_Assignment", pulp.LpMaximize)

        # 定义决策变量 y_{mil} 为二元变量
        y = pulp.LpVariable.dicts(
            "y",
            (
                (m, i, l)
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(self.L_levels + 1)
            ),
            cat=pulp.LpBinary,
        )

        prob += pulp.lpSum(
            self.r1[l] * y[m, i, l] - self.c1[servers_info[m][0]][i] * y[m, i, l]
            for m in range(M_servers)
            for i in range(I_citys)
            for l in range(0, self.L_levels + 1)
        ) - self.c2 * pulp.lpSum(
            n_il[i][l - 1] - pulp.lpSum(y[m, i, l] for m in range(M_servers))
            for i in range(I_citys)
            for l in range(1, self.L_levels + 1)
        )

        # 每个服务员必须且只能被分配到一个任务，无论是在某个城市执行任务还是在家休息。 l 等于0 表示在家休息
        for m in range(M_servers):
            prob += (
                pulp.lpSum(
                    y[m, i, l] for i in range(I_citys) for l in range(self.L_levels + 1)
                )
                == 1
            )

        # 每个城市 i 每个等级 l 被分配的服务员数量不能超过该等级的任务数量
        for i in range(self.I_citys):
            for l in range(1, self.L_levels + 1):
                prob += (
                    pulp.lpSum(y[m, i, l] for m in range(self.M_servers))
                    <= n_il[i][l - 1]
                )

        # 分配给所有服务员的所有任务总数不能超过实际的任务总数。
        total_tasks = sum(
            n_il[i][l - 1] for i in range(I_citys) for l in range(1, self.L_levels + 1)
        )
        prob += (
            pulp.lpSum(
                y[m, i, l]
                for m in range(M_servers)
                for i in range(I_citys)
                for l in range(1, self.L_levels + 1)
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
                for l in range(1, self.L_levels + 1)
            )
            <= available_servers
        )

        for m, (im, wm) in enumerate(servers_info):
            if wm == 0:
                try:
                    prob += y[m, self.H_home_of_server[m], 0] == 1
                except Exception as ex:
                    logging.info(f"{ex=} {m=} {self.H_home_of_server=} {self.L_server=} {S=}")
                    raise ex
            elif wm > 0:

                # 服务员必须分配任务，且任务等级必须高于服务员等级
                # 1. 允许选择 "待在家中" (l = 0)
                prob += (
                    pulp.lpSum(
                        y[m, i, l]
                        for i in range(I_citys)
                        for l in range(self.L_levels + 1)
                        if (l == 0 and i == self.H_home_of_server[m])
                        or (l > 0 and l >= L_server[m] and n_il[i][l - 1] > 0)
                    )
                    == 1
                )

        # 限制了服务员只能接受等级不高于自身等级的任务
        for m in range(M_servers):
            for i in range(I_citys):
                for l in range(1, self.L_levels + 1):
                    prob += y[m, i, l] <= (L_server[m] >= l)

        # 求解问题
        prob.solve()
        obj = pulp.value(prob.objective)

        # 解析结果
        result = [
            (m, i, l)
            for m in range(M_servers)
            for i in range(I_citys)
            for l in range(self.L_levels + 1)
            if pulp.value(y[m, i, l]) == 1
        ]
        return obj, result

    def static_optimal(self, init_S, T=7):
        # static_optimal(I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd,\
        #             T, lambda_il, L_server, r1, c1, c2):
        task_arr = self.task_arr
        pr = T * [0]
        for t in range(T):
            if t == 0:
                S = init_S
                save_S = S
            else:
                S = S_next
            obj, A = self.math_program_static(
                S,
                L_server=self.L_server,
            )
            pr[t] = obj
            xi = task_arr[t]
            S_next = self.state_trans(
                S,
                A,
                xi,
            )
        return save_S, pr

    def Profit(self, S, A):
        # def Profit(S, A, r1, c1, c2):
        n_il, servers_info = copy.deepcopy(S)
        M_servers = len(servers_info)

        # 计算奖励
        reward = 0
        for m in range(M_servers):
            reward += self.r1[A[m][2]]
        # 计算第一个成本， 去完成任务的成本
        cost1 = 0

        for m in range(M_servers):
            # logging.info(f"{servers_info[m][0]=} {A[m][1]=}")
            cost1 += self.c1[servers_info[m][0]][A[m][1]] # c1 是服务器在不同任务和等级上的成本矩阵
        # 计算第二个成本
        cost2 = 0
        n_il_format = np.array([[0] * self.L_levels 
                                for _ in range(self.I_citys)])
        
        for item in A:
            m, i, l =  item
            if l > 0 :
                n_il_format[i][l-1] += 1
        S_A = n_il-n_il_format
        cost2 = self.c2 * np.sum(S_A)
        profit = reward - cost1 - cost2
        if cost2<0:
            logging.info(f"cost2<0: {cost2=}{A=}\n{n_il=}\n{n_il_format=}\n{S_A=}")
        
        logging.info(f"{reward=}-{cost1=}-{cost2=}={profit=}")
        if int(reward) == 0:
            logging.info(f" int(reward) == 0 {S=}\n{A=}")
        return profit

    def single_stage(self, init_S):
        pr = self.T * [0]
        for t in range(self.T):
            if t == 0:
                # 获取生成的初始状态
                S = init_S
                save_S = S
            else:
                # 使用上一轮的状态作为当前状态S
                S = S_next
            # 使用转移决策函数func3_transfer生成决策A
            A = self.func3_transfer(S, V=0)
            # 计算当前状态S和决策A的收益pr
            pr[t] = self.Profit(S, A)
            # 获取每日的任务到达量
            xi = self.task_arr[t]
            # 根据当前状态S、决策A和任务到达量xi生成下一状态S_next
            S_next = self.state_trans(S, A, xi)
        return save_S, pr


# %%
