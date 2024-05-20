

    # def func3_with_V(self, S, t, L_server, H_home_of_server, r1_reward, c1_city_cost, c2, V, Xi_arriving_tasks):
    #     """
    #     生成状态 S 到决策 A 的函数,通过解决线性规划问题来最大化收益 R_total_reward(S, A) + V(t+1, S')。

    #     参数:
    #     S (tuple): 当前状态,包含任务矩阵和服务员信息。
    #     L_server (list): 服务员的等级列表。
    #     H_home_of_server (list): 服务员的家位置列表。
    #     r1_reward (list): 每个等级的收益列表。
    #     c1_city_cost (list of list): I×I 的成本矩阵。
    #     c2 (float): 常数成本。
    #     V (ValueFunction): 值函数对象。
    #     Xi_arriving_tasks (list of list): 在当前时间步到达的任务矩阵。

    #     返回:
    #     list: 最优决策 A,包含每个服务员的位置和等级。
    #     R_total_reward, 总收益
    #     """
    #     n_il, servers_info = S
    #     M_servers = len(servers_info)  # 服务员数量
    #     I_citys = len(n_il)          # 城市数量
    #     L_max = len(r1_reward) - 1    # 最大等级
    #     # 创建问题实例
    #     prob = pulp.LpProblem("Optimal_Server_Assignment", pulp.LpMaximize)

    #     # 定义决策变量 y_{mil} 为二元变量
    #     y = pulp.LpVariable.dicts("y", 
    #                             ((m, i, l) for m in range(M_servers) for i in range(I_citys) for l in range(L_max+1)), 
    #                             cat=pulp.LpBinary)
        
    #     # 定义一个辅助函数,计算下一个状态的价值函数
    #     def calculate_next_state_value(servers_info, arriving_tasks):
    #         """
    #         计算下一个状态的价值函数
    #         Args:
    #             arriving_tasks (list): 当前时间步到达的任务
    #         Returns:
    #             float: 下一个状态的价值函数
    #         """
    #         new_servers_info1 = self.update_servers_info(servers_info, y, M_servers, I_citys, L_max)
    #         # 计算下一个状态, 减去根据决策变量 y 分配到下一个等级 l+1 的服务员数量 pulp.lpSum(y[m, i, l+1] for m in range(M_servers))
    # #         new_servers_info2 = [(i+1, servers_info[m][1]-1) if pulp.sum(y[m, i, l]) == 1 else (servers_info[m][0], servers_info[m][1])
    # #   for m in range(M_servers) for i in range(I_citys) for l in range(L_max+1)]
    #         new_servers_info2 = [(i+1, servers_info[m][1]-1) if pulp.value(pulp.lpSum(y[m, i, l] for l in range(L_max+1))) == 1 else (servers_info[m][0], servers_info[m][1])
    #                  for m in range(M_servers) for i in range(I_citys)]
    #         print(f"{new_servers_info1=}")
    #         print(f"{new_servers_info2=}")
    #         print(f"{len(servers_info)=} {len(new_servers_info1)=}{len(new_servers_info2)=} ")
    #         new_state = [
    #             [[n_il[i][l] - pulp.lpSum(y[m, i, l] for m in range(M_servers)) \
    #               + arriving_tasks[i][l] for l in range(L_max)] for i in range(I_citys)]
    #             ,new_servers_info2
    #         ]
    #         # 压缩状态
    #         S_bar_next = self.aggreg_state(new_state, self.Z_cluster_num, self.X, M_servers, I_citys, self.L_levels)
    #         # 更新价值函数的计数
    #         count = V.get_count(t+1, S_bar_next)
    #         # 返回下一个状态的价值函数
    #         return V.get(t+1, S_bar_next) * (1 - 1 / count)


    #     # prob += pulp.lpSum(
    #     # r1_reward[l1] * y[m1, i1, l1] - c1_city_cost[servers_info[m1][0]-1][i1] * y[m1, i1, l1]
    #     #                for m1 in range(M_servers) for i1 in range(I_citys)
    #     #                for l1 in range(0, L_max+1))\
    #     #                 - c2 * pulp.lpSum(n_il[i][l-1] - pulp.lpSum(y[m, i, l] for m in range(M_servers)) for i in range(I_citys)
    #     #                for l in range(1, L_max+1))


    #     # 目标函数
        
    #     prob += pulp.lpSum(
    #         r1_reward[l1] * y[m1, i1, l1] - c1_city_cost[servers_info[m1][0]-1][i1] * y[m1, i1, l1]
    #                     for m1 in range(M_servers) for i1 in range(I_citys)
    #                     for l1 in range(L_max+1)) \
    #             - c2 * pulp.lpSum(n_il[i][l-1] - pulp.lpSum(y[m, i, l] for m in range(M_servers)) for i in range(I_citys)
    #                     for l in range(1, L_max+1)) \
    #             +  calculate_next_state_value(servers_info, Xi_arriving_tasks)
    #             # + pulp.lpSum(calculate_next_state_value(Xi_arriving_tasks)) / len(Xi_arriving_tasks)

        
    #     # 添加约束
    #     # 每个服务员只能分配到一个地点和等级
    #     for m in range(M_servers):
    #         prob += pulp.lpSum(y[m, i, l] for i in range(I_citys) for l in range(L_max+1)) == 1

    #     # 服务员不工作时，分配到家乡的等级0
    #     for m, (im, wm) in enumerate(servers_info):
    #         if wm == 0:
    #             prob += y[m, H_home_of_server[m]-1, 0] == 1

    #     # 服务员工作时，必须分配到合适的等级和城市
    #     for m, (im, wm) in enumerate(servers_info):
    #         if wm > 0:
    #             prob += pulp.lpSum(y[m, i, l] for i in range(I_citys) for l in range(L_server[m], L_max+1) if n_il[i][l-1] > 0) == 1

    #     # 资源使用不超过可用数量
    #     for i in range(I_citys):
    #         for l in range(1, L_max+1):
    #             prob += pulp.lpSum(y[m, i, l] for m in range(M_servers)) <= n_il[i][l-1]


    #     # 求解问题
    #     prob.solve()

    #     # 解析结果
    #     result = [(m, i+1, l) for m in range(M_servers) for i in range(I_citys) for l in range(L_max+1) if pulp.value(y[m, i, l]) == 1]
    #     return result, pulp.value(prob.objective) # i[1,I_citys] l[0,L_max]


#  def update_servers_info(self, servers_info, y, M_servers, I_citys, L_max):
#         new_servers_info = []
#         for m in range(M_servers):
#             for i in range(I_citys):
#                 for l in range(L_max+1):
#                     # print(f"{y[m, i, l]=}")
#                     # city_assigned = pulp.reward(y[m, i, l])
#                     city_assigned = pulp.reward(y[m, i, l])
#                     if city_assigned == 1:
#                         new_work_days  = servers_info[m][1] - 1
#                         if l == 0:
#                             if new_work_days >= 0:
#                                 new_servers_info.append((H_home_of_server[m], new_work_days))
#                             else:
#                                 new_servers_info.append((H_home_of_server[m], W_workdays))
#                         elif l != 0:
#                             if new_work_days >= 0:
#                                 new_servers_info.append((i+1, new_work_days))
#                             else:
#                                 new_servers_info.append((i+1, W_workdays))

#                     # else:
#                     #     new_servers_info.append(servers_info[m])
        
#         return new_servers_info





    # def aggreg_state(self, S_t, Z_cluster_num, X, M_servers, I_citys, L_levels):
    #     '''
    #     函数2
    #     定义一个函数，用于根据给定的参数将复杂的状态 S 压缩成一个简化的状态 barS。
    #     输入:
    #       S: 当前状态，一个复杂的结构，包含两部分信息：
    #          - 一个数组，表示每个城市每个等级的数量(n_il)。
    #          - 一个列表，表示服务员和他们服务的城市及工作日(i_m, w_m)
    #       Z_cluster_num: 一个整数，表示将城市分成多少个聚类。
    #       X: 一个整数，用于计算 N 矩阵中的元素值。
    #       M_servers: 服务员的总数。
    #       I_citys: 城市的总数。
    #       L_levels: 等级的总数。
    #     输出:
    #       barS: 一个元组，表示压缩后的状态，包含以下三个部分：
    #             - N: 一个二维数组，表示每个聚类的等级之和。
    #             - g: 一个数组，表示每个聚类的状态。
    #             - w: 一个整数，表示第一个服务员的工作日数。
    #     '''

    #     # 计算正在工作的服务员的数量, S[1][m_server][1]即S1_servers[m_server][1]即w_m
    #     barM = np.sum([1 for m_server in range(M_servers) if S_t[1][m_server][1] != 0]) # 距离放假时间不等于0
    #     # 根据城市数量和设定的簇数，将城市分成Z个簇
    #     cluster = split_list(I_citys, Z_cluster_num)
    #     # 计算实际的簇数，考虑到可能会有余数
    #     # num_cluster = divide_reminder(I_citys, Z_cluster_num) # 这个可以替换为下面的
    #     num_cluster = np.ceil(I_citys / Z_cluster_num).astype(int) # 向上取整
    #     # 初始化表示各簇状态的数组g
    #     g = np.zeros(num_cluster)
        
    #     # 压缩状态的第二部分：计算每个簇的状态
    #     for z_cluster in range(num_cluster):
    #         # 统计每个簇中有多少业务员正在工作 
    #         e_z = np.sum([1 for m_server in range(M_servers) if S_t[1][m_server][0] in cluster[z_cluster]])
    #         # 根据工作的业务员数量设置簇的状态
    #         if e_z == 0:
    #             g[z_cluster] = 0  # 无业务员工作
    #         elif e_z <= barM / num_cluster:
    #             g[z_cluster] = 1  # 工作业务员数量低于或等于平均值
    #         else:
    #             g[z_cluster] = 2  # 工作业务员数量高于平均值
        
    #     # 获取第一个业务员的工作量
    #     w = S_t[1][0][1]
    #     # 压缩状态的第一部分：计算每个簇中各等级的数量总和
    #     N = np.zeros((num_cluster, L_levels))  # 初始化N矩阵
        
    #     i_all = [i for i in cluster[z_cluster]]

    #     for z_cluster in range(num_cluster):
    #         for l in range(L_levels):
    #             # 对每个簇的每个等级，计算其数量总和，但不超过X
    #             state_sum = sum(S_t[0][i-1][l] for i in cluster[z_cluster])
    #             # print(f"{state_sum=}")
    #             task_sum = pulp.reward(state_sum)
    #             if task_sum is None:
    #                 task_sum = 0
    #             else:
    #                 print(f"{task_sum=} is not None")
    #             N[z_cluster][l] = min(X, task_sum)
    #             # print(f"{N[z_cluster][l]=}")
    #     # 将 N 和 g 转换为元组
    #     N = tuple(map(tuple, N))
    #     g = tuple(g)
    #     # 将计算出的N矩阵、簇的状态数组g和第一个业务员的工作量w组合成新的压缩状态barS
    #     barS = (N, g, w)        
    #     return barS






        # # 压缩后，参与计算调用func5_bar(), 
        # # 1. 通过规划算法计算得到一个决策，根据决策获取下一个状态
        # # 2. 将下一个状态进行压缩，重新参与规划算法的决策（这个是迭代过程实现的还是在单步过程计算两次规划算法?)
        # # S00->V00->S10->V10->S20
        # # S01->V01->S11->V11->S21
        # # S01计算时使用与S11状态相同的历史迭代遇到过的决策收益V10?

        # barS=aggreg_state(S1, Z_cluster_num, X, M_servers, I_citys, L_levels)

        # for j in range(max_iter):
        #     S = func1(I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd)  # 随机生成初始状态
        #     SS, AA, RR = [], [], []  # 记录状态、决策和收益的序列
        #     for t in range(1, W_workdays+1):  # 对每个时间段
        #         A = None
        #         max_Q = float('-inf')
                
        #         for Y_best_allocation in func6(S, mathcal_L, mathscr_L, N_1, N_2):  # 遍历决策空间
        #             R_total_reward = func5(S, mathcal_L, mathscr_L, N_1, N_2, H_home_of_server, r1_reward, c1_city_cost, c2)[1]  # 计算即时收益
        #             next_S = self.transition(S, Y_best_allocation, func7)  # 计算下一个状态
        #             next_S_bar = func2(next_S, Z_cluster_num, X, M_servers, I_citys, L_levels)  # 状态压缩
        #             Q = R_total_reward + V_reward_func[t][hash(next_S_bar)]  # 计算动作值函数
        #             if Q > max_Q:
        #                 max_Q = Q
        #                 A = Y_best_allocation

        #         SS.append(S)
        #         AA.append(A)
        #         RR.append(func5(S, mathcal_L, mathscr_L, N_1, N_2, H_home_of_server, r1_reward, c1_city_cost, c2)[1])

        #         S = self.transition(S, A, func7)  # 状态转移

        #     # 更新值函数
        #     G = 0
        #     for t in range(W_workdays-1, -1, -1):
        #         G += RR[t]  # 计算累积收益
        #         S_bar = func2(SS[t], Z_cluster_num, X, M_servers, I_citys, L_levels)  # 状态压缩
        #         n = self.count_accur(S_bar, SS[:t])  # 计算状态出现次数
        #         V_reward_func[t][hash(S_bar)] = (G + (n-1) * V_reward_func[t][hash(S_bar)]) / n  # 更新值函数

        # # 计算最优策略
        # for S in self.get_all_states(I_citys, L_levels, W_workdays, M_servers, x_max_task_num):
        #     pi[hash(S)] = None
        #     max_Q = float('-inf')
        #     for Y_best_allocation in func6(S, mathcal_L, mathscr_L, N_1, N_2):
        #         R_total_reward = func5(S, mathcal_L, mathscr_L, N_1, N_2, H_home_of_server, r1_reward, c1_city_cost, c2)[1]
        #         next_S = self.transition(S, Y_best_allocation, func7)
        #         next_S_bar = func2(next_S, Z_cluster_num, X, M_servers, I_citys, L_levels)
        #         Q = R_total_reward + V_reward_func[W_workdays-1][hash(next_S_bar)]
        #         if Q > max_Q:
        #             max_Q = Q
        #             pi[hash(S)] = Y_best_allocation

        # return V_reward_func, pi