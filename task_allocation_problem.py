# %%
import random

import numpy as np
import pandas as pd
import pulp


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
        # # print(f"{caller_name=} {S_t_str=}")
        return S_t_str

    def init_s_value_t(self, t, S_agg):
        if t not in self.s_values:
            self.s_values[t] = {}
        tuple_s_agg = self._convert_to_tuple(S_agg)
        self.s_values[t][tuple_s_agg] = {"count": 1, "total_reward": 0}

    def update_total_reward(self, t, S_agg, total_reward):
        for t in self.s_values:
            tuple_s_agg = self._convert_to_tuple(S_agg)
            if tuple_s_agg in self.s_values[t]:
                count = self.s_values[t][tuple_s_agg]["count"]
                old_value = self.s_values[t][tuple_s_agg]["total_reward"]
                weight = 1 / (count + 1)
                new_value = old_value + weight * (total_reward[t] - old_value)
                self.s_values[t][tuple_s_agg]["count"] = count + 1
                self.s_values[t][tuple_s_agg]["total_reward"] = new_value
                print(f"{new_value=}")
            else:
                self.init_s_value_t(t, S_agg)

    def get_total_reward(self, t, S_agg):
        if t in self.s_values:
            tuple_s_agg = self._convert_to_tuple(S_agg)
            if tuple_s_agg in self.s_values[t]:
                return self.s_values[t][tuple_s_agg]["total_reward"]
        return 0

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

        # print(S_next[1][39][1])
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
                    print(f"{ex=} {m=} {self.H_home_of_server=} {self.L_server=} {S=}")
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
        # print("M_servers, I_citys, self.L_levels, N_1, N_2, mathcal_L, mathscr_L", M_servers, I_citys, L_max_set, N_1, N_2, mathcal_L, mathscr_L)
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
            # print("L_set ", L_set)
            M_servers_L = [
                m
                for m in range(M_servers)
                if servers_info[m][1] > 0 and L_server[m] in L_set
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

    def func7(self):
        # 生成了每日新到达的任务?
        # T: 表示时间周期，例如天数
        # x_max_task_num: 矩阵元素的最大取值
        # lambda_il: 泊松分布的率参数矩阵 (I_citys x_max_task_num L_levels)

        # 获取 lambda_il 的维度为 I_citys 和 L_levels
        I_citys, L_levels = self.lambda_il.shape

        # 初始化三维数组
        arriving_tasks_i = np.zeros((self.T, I_citys, L_levels), dtype=int)

        # 生成每个时间步的 I_citys x_max_task_num L_levels 矩阵
        for t in range(self.T):
            for i in range(I_citys):
                for l in range(L_levels):
                    # 使用泊松分布生成矩阵元素
                    arriving_tasks_i[t, i, l] = min(
                        np.random.poisson(self.lambda_il[i, l]), self.x_max_task_num
                    )

        return arriving_tasks_i

    def func8_org(self, J, Z_cluster_num, X):
        # 目前适配了自动生成每个benchmark相同的到达任务和，初始化J个初始S
        # 初始化s_value列表，用于存储状态的信息
        s_value = []
        # 生成每日新到达的任务
        # self.task_arr = self.func7()
        self.task_arr = self.task_arr

        for j in range(J):
            # 初始化每日的收益列表
            pr = self.T * [0]
            # 初始化位置列表和转移位置列表
            location = []
            tran_loc = []
            for t in range(self.T):
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
            total_reward = [0] * self.T
            # 逆序计算每日的总收益
            for t in range(self.T - 1, -1, -1):
                total_reward[t] += sum(pr[t:])
            print(f"{location=}")
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

    def func8(self, J, Z_cluster_num, X):
        s_value = SValue(self.T)
        for j in range(J):
            pr = self.T * [0]
            for t in range(self.T):
                if t == 0:
                    S = self.init_S_J[j]
                else:
                    S = S_next
                S_agg = self.func2(S, Z_cluster_num, X)
                print(f"{S_agg=}")
                if j == 0:
                    s_value.init_s_value_t(t, S_agg)
                    V = 0
                else:
                    V = s_value.get_total_reward(t, S_agg)
                    if V == 0:
                        s_value.init_s_value_t(t, S_agg)
                A = self.func3_transfer(S, V)
                pr[t] = self.Profit(S, A)
                xi = self.task_arr[t]
                S_next = self.state_trans(S, A, xi)
            total_reward = [0] * self.T
            for t in range(self.T - 1, -1, -1):
                total_reward[t] == sum(pr[t:])
            s_value.update_total_reward(t, S_agg, total_reward)
            print(f"{S_agg=}")

            print(f"-------{j=}-----{s_value.s_values[0].keys()=}-------")
            break

        return s_value.get_s_values()

    def all_task_init(self, J):

        # 生成每日新到达的任务
        random.seed(42)
        np.random.seed(42)  # 生成初始状态S
        self.task_arr = self.func7()
        random.seed(42)
        np.random.seed(42)  # 生成初始状态S
        self.init_S_J = [self.func1() for j in range(J)]
        for i in range(5):
            self.save_to_one_csv(self.init_S_J[i], csv_path=f"init/init_state_{i}.csv")

    def math_program_near(self, S, L_server):
        n_il, servers_info = S
        M_servers = len(servers_info)  # 服务员数量
        I_citys = len(n_il)  # 城市数量

        M_servers = self.M_servers  # 服务员数量
        I_citys = self.I_citys  # 城市数量
        # 创建问题实例
        prob = pulp.LpProblem("Optimal_Server_Assignment", pulp.LpMinimize)

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
            self.c1[servers_info[m][0]][i] * y[m, i, l]
            for m in range(M_servers)
            for i in range(I_citys)
            for l in range(0, self.L_levels + 1)
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
                    print(f"{ex=} {m=} {self.H_home_of_server=} {self.L_server=} {S=}")
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

    def nearest_distance(self, init_S):
        # 获取任务到达量的数组
        task_arr = self.task_arr

        # 初始化收益列表
        pr = self.T * [0]

        for t in range(self.T):
            if t == 0:
                # 生成初始状态
                S = init_S
                save_S = S
            else:
                # 使用上一状态的结果作为当前状态
                S = S_next
            # 使用最近距离策略求解决策和收益
            obj, A = self.math_program_near(S, L_server=self.L_server)
            pr[t] = obj

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
                    print(f"{ex=} {m=} {self.H_home_of_server=} {self.L_server=} {S=}")
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

    def static_optimal(self, init_S):
        # static_optimal(I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd,\
        #             T, lambda_il, L_server, r1, c1, c2):
        task_arr = self.task_arr
        pr = self.T * [0]
        for t in range(self.T):
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
        n_il, servers_info = S
        M_servers = len(servers_info)

        reward = 0
        for m in range(M_servers):
            reward += self.r1[A[m][2]]

        cost1 = 0
        for m in range(M_servers):
            cost1 += self.c1[servers_info[m][0]][A[m][1] - 1]

        cost2 = 0
        dic1 = {}
        for i, row in enumerate(n_il):  # i = i
            for j, value in enumerate(row):  # j = l
                dic1[(i, j + 1)] = (
                    value  # dic1是将任务矩阵写成字典形式 value = level task count
                )
        dic2 = {
            (x[1], x[2]): 1 for x in A if x[2] != 0
        }  # dic2是将决策A=(m,i,l)写成字典
        S_A_cell = {}
        for (
            key
        ) in (
            dic1
        ):  # 对于每个任务 i l的id 如果存在在决策里，表示有分配给这个任务的 dic2[key] 必然为1
            if key in dic2:
                S_A_cell[key] = dic1[key] - dic2[key]  # 减少分配
            else:
                S_A_cell[key] = dic1[key]
        S_A = [[0] * len(n_il[0]) for _ in range(len(n_il))]
        for key, value in S_A_cell.items():
            S_A[key[0]][key[1] - 1] = value  # S_A 是S-A之后的值
        cost2 = self.c2 * np.sum(S_A)

        profit = reward - cost1 - cost2
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

I_citys = 26
L_levels = 5
W_workdays = 6
M_servers = 40
x_max_task_num = 2
random.seed(42)
np.random.seed(42)
H_home_of_server = [
    random.randint(1, I_citys) - 1 for _ in range(M_servers)
]  # 随机家的位置
lambd = np.random.rand(I_citys, L_levels)  # 生成率参数矩阵,
T = 7
lambda_il = np.random.rand(I_citys, L_levels)
L_server = [random.randint(1, L_levels) for _ in range(M_servers)]
r1 = [0, 3500, 3000, 2500, 2000, 1500]
c1 = [
    [0 if i == j else random.randint(100, 500) for j in range(I_citys)]
    for i in range(I_citys)
]
c2 = 100
J = 10000
Z_cluster_num = 3
X = 3

# 创建问题实例
problem = TaskAllocationProblem(
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
)

print(f"{L_server=}")
print(f"{H_home_of_server=}")
print(f"{c1=}")


# %% [markdown]
# ## 测试定义的函数

# %%
# func1 测试

random.seed(42)
np.random.seed(42)
for i in range(5):
    S = problem.func1()
    if not i:
        print(f"{S[0]=}\n {S[1]=}")
    # problem.save_to_csv(S, tasks_csv_path=f"init/init_tasks_rnd_{i}.csv", servers_csv_path=f"init/init_servers_rnd_{i}.csv")
    problem.save_to_one_csv(S, csv_path=f"init/init_state_{i}.csv")

print("New")

random.seed(42)
np.random.seed(42)
S = problem.func1()
print(f"{S[0]=}\n {S[1]=}")

# %%
# func2 测试
barS = problem.func2(S=S, Z_cluster_num=Z_cluster_num, X=X)
print(f"{barS=}")

# %%
problem.task_arr = problem.func7()
# func3 测试
A = problem.func3_transfer(S=S, V=0)
print(f"{A=}")

# %%

# func4 测试
mathcal_L, N_1, N_2 = problem.func4(S=S)
print(f"{(mathcal_L, N_1, N_2, S)=}")

# func5 测试
Y = problem.func5(S, mathcal_L, N_1, N_2)
print(f"{Y=}")

# func6 测试
A_per_L_set = problem.func6(S, mathcal_L, N_1, N_2)
print(f"{A_per_L_set=}")


# %%
# 测试收益函数
problem.Profit(S, A)

# %%
# func7 测试
arriving_tasks_i = problem.func7()
print(f"{arriving_tasks_i=}")

# %%

# func8 测试
# 调用方法
problem.all_task_init(J=J)
s_value = problem.func8(J, Z_cluster_num, X)
print(f"{s_value=}")

# %%

# func11 测试 random
# func12 测试 nearest
problem.all_task_init(J=10000)
save_S, pr = problem.nearest_distance(init_S=problem.init_S_J[0])
print(f"{save_S}=, {pr=}")

# %%

# func13 测试 single stage
save_S, pr = problem.static_optimal(init_S=problem.init_S_J[0])
print(f"{save_S}=, {pr=}")

# %%
# from numpy import array
# problem.func3_transfer(S=(array([[2, 0, 0, 2, 0],
#        [1, 0, 0, 0, 1],
#        [0, 1, 0, 0, 0]]), [(1, 4), (2, 2), (2, 4)]), V=0)

# %%
import csv


class TaskRunner:
    # 本类的目标

    def run(self):
        T_values = [3, 14, 21]
        Z_values = [3, 5, 9]
        S_n = 5
        problem_n = 4
        VFA_state_values = {}
        # 初始化 result 四维矩阵
        result = np.zeros((S_n, problem_n, len(T_values), len(Z_values)))

        try:
            # for T_idx, T in enumerate(T_values):
            #     for Z_idx, Z in enumerate(Z_values):

            #         print(f"-------{(T,Z)=}--------")
            #         self.init_a_problem(T=T)

            #         for S_idx, s in enumerate(problem.init_S_J[0:5]):
            #             save_S, pr1= self.problem.static_optimal(init_S=s)
            #             save_S, pr2 = self.problem.nearest_distance(init_S=s) # pr 是T个阶段的收益
            #             save_S, pr3 = self.problem.single_stage(init_S=s)
            #             print(S_idx, pr1)
            #              # 解包元组并求和
            #             result[S_idx][1][T_idx][Z_idx] = sum(pr1)
            #             result[S_idx][2][T_idx][Z_idx] = sum(pr2)
            #             result[S_idx][3][T_idx][Z_idx] = sum(pr3)

            #             print(f"{(S_idx,1,T_idx,Z_idx)=} {result[S_idx][1][T_idx][Z_idx]=}")
            #             print(f"{(S_idx,2,T_idx,Z_idx)=} {result[S_idx][2][T_idx][Z_idx]=}")
            #             print(f"{(S_idx,3,T_idx,Z_idx)=} {result[S_idx][3][T_idx][Z_idx]=}")

            # self.save_to_csv(result, T_values, Z_values, S_n, file_name = "./data/benchmark_results.csv")
            print(f"----------------------finished benchmark---------------------")
            for T_idx, T in enumerate(T_values):
                for Z_idx, Z in enumerate(Z_values):
                    print(f"-------{(T,Z)=}--------")
                    self.init_a_problem(T=T)
                    s_value = self.run_VFA_task(T, Z)
                    VFA_state_values.update({(T, Z): s_value})

                    for S_idx, s in enumerate(problem.init_S_J[0:5]):
                        s_agg = self.func2(s, Z_cluster_num, X)
                        for value in s_value:
                            if all(
                                np.array_equal(a, b)
                                for a, b in zip(s_agg, s_value[0][2])
                            ):
                                V = s_value[0][3]
                        else:
                            V = 0
                        result[S_idx][0][T][Z] = V
                        print(
                            f"{(S_idx,0,T_idx,Z_idx)=} {result[S_idx][0][T_idx][Z_idx]=}"
                        )
        finally:
            # 将结果保存到文件
            np.save("result.npy", result)

            # 将结果保存到 CSV 文件
            self.save_to_csv(
                result, T_values, Z_values, S_n, file_name="./data/results.csv"
            )

    def init_a_problem(
        self,
        T,
    ):
        print("problem init")
        I_citys = 26
        L_levels = 5
        W_workdays = 6
        M_servers = 40
        x_max_task_num = 2
        random.seed(42)
        np.random.seed(42)
        H_home_of_server = [random.randint(1, I_citys) - 1 for _ in range(M_servers)]
        L_server = [random.randint(1, L_levels) for _ in range(M_servers)]
        r1 = [0, 3500, 3000, 2500, 2000, 1500]
        lambd = np.random.rand(I_citys, L_levels)  # 生成率参数矩阵
        c1 = [
            [0 if i == j else random.randint(100, 500) for j in range(I_citys)]
            for i in range(I_citys)
        ]
        # 创建问题实例
        self.problem = TaskAllocationProblem(
            I_citys=I_citys,
            L_levels=L_levels,
            W_workdays=W_workdays,
            M_servers=M_servers,
            x_max_task_num=x_max_task_num,
            H_home_of_server=H_home_of_server,
            lambd=lambd,
            T=T,
            lambda_il=lambd,
            L_server=L_server,
            r1=r1,
            c1=c1,
            c2=200,
        )

        self.problem.all_task_init(J=10000)
        print("problem init done")

    def run_VFA_task(self, T, Z):
        s_value = self.problem.func8(J=10000, Z_cluster_num=Z, X=x_max_task_num)
        return s_value

    def save_to_csv(
        self, result, T_values, Z_values, S_n, file_name="./data/results.csv"
    ):
        """
        将结果保存到 CSV 文件中。
        :param result: 四维矩阵，包含不同参数组合下的收益。
        :param T_values: T 的取值列表。
        :param Z_values: Z 的取值列表。
        :param S_n: S 的数量。
        """
        headers = [""] + [f"T={T},Z={Z}" for T in T_values for Z in Z_values]

        rows = []

        for S_idx in range(S_n):
            for problem_idx, problem_name in enumerate(["VFA", "RA", "RDA", "MA"]):
                row = [f"Sj{S_idx + 1}_{problem_name}"]
                for T_idx in range(len(T_values)):
                    for Z_idx in range(len(Z_values)):
                        row.append(result[S_idx][problem_idx][T_idx][Z_idx])
                rows.append(row)

        with open(file_name, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)


# %%
task = TaskRunner()
task.run()

# %%
