import pulp
import numpy as np
import random

class TaskAllocationProblem:
    def __init__(self, I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd, T, lambda_il, L_server, r1, c1, c2):
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
        # ...
        pass

    def func2(self, S, Z_cluster_num, X):
        # ...
        pass

    def func3_transfer(self, S, V):
        # ...
        pass

    def state_trans(self, S, act, xi):
        # ...
        pass

    def func7(self):
        # ...
        pass

    def func8(self, J, Z_cluster_num, X):
        s_value = []
        for j in range(J):
            task_arr = self.func7()
            pr = self.T * [0]
            location = []
            tran_loc = []
            for t in range(self.T):
                if t == 0:
                    S = self.func1()
                else:
                    S = S_next
                S_agg = self.func2(S, Z_cluster_num, X)
                if j == 0:
                    s_value.append([t, 1, S_agg, 0])
                    location.append(t)
                    V = 0
                else:
                    len_s_v = len(s_value)
                    for i in range(len_s_v):
                        if t == s_value[i][0] and all(np.array_equal(a, b) for a, b in zip(S_agg, s_value[i][2])):
                            location.append(i)
                            tran_loc.append(i)
                            V = s_value[i][3]
                            break
                    else:
                        s_value.append([t, 1, S_agg, 0])
                        loc = len(s_value) - 1
                        location.append(loc)
                        V = 0
                A = self.func3_transfer(S, V)
                pr[t] = self.Profit(S, A)
                xi = task_arr[t]
                S_next = self.state_trans(S, A, xi)

            total_reward = [0] * self.T
            for t in range(self.T - 1, -1, -1):
                total_reward[t] += sum(pr[t:])
            print(location)
            for i in location:
                i_t = location.index(i)
                if i in tran_loc:
                    new_value = (1 - 1 / (s_value[i][1] + 1)) * s_value[i][3] + 1 / (s_value[i][1] + 1) * total_reward[i_t]
                    s_value[i][1] += 1
                    s_value[i][3] = new_value
                else:
                    s_value[i][3] = total_reward[i_t]

        return s_value

    def math_program_near(self, S):
        # ...
        pass

    def nearest_distance(self):
        # ...
        pass

    def math_program_static(self, S):
        # ...
        pass

    def static_optimal(self):
        # ...
        pass

    def Profit(self, S, A):
        # ...
        pass

# 创建问题实例
problem = TaskAllocationProblem(I_citys, L_levels, W_workdays, M_servers, x_max_task_num, H_home_of_server, lambd, T, lambda_il, L_server, r1, c1, c2)

# 调用方法
s_value = problem.func8(J, Z_cluster_num, X)
save_S, pr = problem.nearest_distance()
save_S, pr = problem.static_optimal()