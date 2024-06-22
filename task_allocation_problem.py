# %%
import csv
import json
import pickle
import random
import os
import numpy as np

from my_wraps import timeit
from task_alloacate_problem import SValue, TaskAllocationProblem

# 装饰器用于计算函数调用时间


class TaskRunner:
    # 本类的目标
    def __init__(self) -> None:
        pass

    def debug_profit(self):
        
        J = 1
        T = 21
        Z = 3

        self.init_a_problem(T=T, Z=Z, J=J)

        from numpy import array 
        S=(array([[3, 8, 1, 3, 3],
       [0, 0, 5, 3, 2],
       [0, 6, 3, 0, 1],
       [0, 2, 5, 2, 0],
       [3, 0, 2, 0, 8],
       [1, 3, 3, 1, 1],
       [3, 2, 0, 3, 6],
       [2, 1, 0, 8, 0],
       [1, 2, 0, 3, 2],
       [5, 0, 1, 1, 0],
       [3, 3, 1, 5, 2],
       [5, 0, 1, 1, 0],
       [2, 0, 2, 1, 0],
       [1, 1, 1, 1, 3],
       [1, 0, 0, 4, 4],
       [1, 4, 0, 3, 0],
       [0, 3, 1, 0, 3],
       [0, 1, 3, 3, 0],
       [1, 4, 4, 1, 2],
       [2, 3, 3, 0, 0],
       [0, 2, 2, 2, 4],
       [2, 0, 3, 1, 0],
       [0, 0, 7, 3, 2],
       [3, 8, 0, 6, 1],
       [2, 6, 0, 0, 1],
       [1, 5, 3, 0, 1]]), [(9, 0), (3, 4), (0, 5), (23, 3), (8, 2), (7, 5), (7, 3), (4, 0), (23, 6), (3, 4), (21, 2), (23, 4), (17, 4), (17, 0), (18, 2), (13, 5), (1, 2), (0, 5), (2, 5), (0, 0), (7, 4), (16, 6), (13, 0), (0, 3), (17, 4), (6, 6), (22, 5), (20, 5), (17, 0), (17, 5), (13, 4), (7, 2), (14, 5), (22, 0), (13, 0), (1, 0), (12, 0), (24, 4), (25, 4), (5, 6)])
        A=[(0, 20, 0), (1, 3, 0), (2, 0, 0), (3, 23, 0), (4, 8, 0), (5, 7, 0), (6, 7, 0), (7, 4, 0), (8, 23, 0), (9, 3, 0), (10, 21, 0), (11, 23, 0), (12, 17, 0), (13, 2, 0), (14, 18, 0), (15, 13, 0), (16, 1, 0), (17, 0, 0), (18, 2, 0), (19, 6, 0), (20, 7, 0), (21, 16, 0), (22, 19, 0), (23, 0, 0), (24, 17, 0), (25, 6, 0), (26, 22, 0), (27, 20, 0), (28, 22, 0), (29, 17, 0), (30, 13, 0), (31, 7, 0), (32, 14, 0), (33, 18, 0), (34, 8, 0), (35, 25, 0), (36, 0, 0), (37, 24, 0), (38, 25, 0), (39, 5, 0)]

        A = self.problem.generate_random_allocation(
                S, self.problem.H_home_of_server, L_server=self.problem.L_server
            )
        
        profit = self.problem.Profit(S,A)
        print(f"1.{profit=} {A=}")

        A = self.problem.math_program_near(
            S,  L_server=self.problem.L_server
        )
        
        profit = self.problem.Profit(S,A)
        print(f"2.{profit=} {A=}")

        obj, A = self.problem.math_program_static(
            S,  L_server=self.problem.L_server
        )
        
        profit = self.problem.Profit(S,A)
        print(f"3.{profit=} {A=}")
        
    def run_test(self):

        J = 1
        T = 21
        Z = 3

        self.init_a_problem(T=T, Z=Z, J=J)
        s_value = self.run_VFA_task(T=T, Z=Z, J=J)

    def run_benchmark(self):
        T_values = [7, 14, 21]
        Z_values = [3, 5, 9]
        S_n = 5
        problem_n = 4
        result = np.zeros((S_n, problem_n, len(T_values), len(Z_values)))
        print("sp")
        
        self.init_a_problem(T=7, J=10000)
        self.problem.nearest_distance(init_S=self.problem.init_S_J[0], T=7) 
        print("sp done")
        for T_idx, T in enumerate(T_values):
            for Z_idx, Z in enumerate(Z_values):

                print(f"-------{(T,Z)=}--------")
                self.init_a_problem(T=T, J=10000)

                for S_idx, s in enumerate(self.problem.init_S_J[0:5]):
                    # RA RDA MA
                    save_S, pr1= self.problem.calc_total_reward_for_init_S_by_rnd(init_S=s, T=T)
                    save_S, pr2 = self.problem.nearest_distance(init_S=s, T=T) # pr 是T个阶段的收益
                    save_S, pr3 = self.problem.static_optimal(init_S=s, T=T)
        #             print(S_idx, pr1)
        #              # 解包元组并求和
                    result[S_idx][1][T_idx][Z_idx] = sum(pr1)
                    result[S_idx][2][T_idx][Z_idx] = sum(pr2)
                    result[S_idx][3][T_idx][Z_idx] = sum(pr3)

                    print(f"{(S_idx,1,T_idx,Z_idx)=} {result[S_idx][1][T_idx][Z_idx]=}")
                    print(f"{(S_idx,2,T_idx,Z_idx)=} {result[S_idx][2][T_idx][Z_idx]=}")
                    print(f"{(S_idx,3,T_idx,Z_idx)=} {result[S_idx][3][T_idx][Z_idx]=}")

        self.save_to_csv(result, T_values, Z_values, S_n, file_name = "./data/benchmark_results.csv")
        
    def run(self):
        J = 10000
        T_values = [7, 14, 21]
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
                    x_max_task_num = self.problem.x_max_task_num
                    s_value = self.run_VFA_task(T, Z, J=J)
                    VFA_state_values.update({(T, Z): s_value})

                    # 计算矩阵
                    for S_idx, s in enumerate(self.problem.init_S_J[0 : min(J, 5)]):
                        s_agg = self.problem.func2(s, Z_cluster_num=Z, X=x_max_task_num)
                        result[S_idx][0][T_idx][Z_idx] = s_value.get_total_reward(
                            t=0, S_agg=s_agg
                        )
                        print(
                            f"{(S_idx,0,T_idx,Z_idx)=} {result[S_idx][0][T_idx][Z_idx]=}"
                        )

                    # 保存 result 到 CSV 文件
                    self.save_to_csv(
                        result,
                        T_values,
                        Z_values,
                        S_n,
                        file_name=f"./data/result_per_iter/result_{J}_{Z}_{T}.csv",
                    )
        finally:

            # 将结果保存到 CSV 文件
            self.save_to_csv(
                result, T_values, Z_values, S_n, file_name="./data/final/results.csv"
            )

            # 获取文件夹路径
            folder_path = os.path.dirname("./data/save_params/s_value.pkl")

            # 检查文件夹是否存在，如果不存在则创建它
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open("./data/save_params/s_value.pkl", "wb") as file:
                pickle.dump(s_value, file)
            
            # 获取文件夹路径
            folder_path = os.path.dirname("./data/save_params/s_value_s_values.json")

            # 检查文件夹是否存在，如果不存在则创建它
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 单独保存 s_value.s_values 字典为 JSON
            with open("./data/save_params/s_value_s_values.json", "w") as file:
                json.dump(s_value.s_values, file, indent=4)

    def run_org(self):
        T_values = [7, 14, 21]
        Z_values = [3, 5, 9]
        S_n = 5
        problem_n = 4
        VFA_state_values = {}
        # 初始化 result 四维矩阵
        result = np.zeros((S_n, problem_n, len(T_values), len(Z_values)))

        try:
            for T_idx, T in enumerate(T_values):
                for Z_idx, Z in enumerate(Z_values):
                    print(f"-------{(T,Z)=}--------")
                    self.init_a_problem(T=T)
                    s_value = self.run_VFA_task_org(T, Z, J=5)
                    VFA_state_values.update({(T, Z): s_value})

                    for S_idx, s in enumerate(self.problem.init_S_J[0:5]):
                        s_agg = self.problem.func2(s, Z_cluster_num=Z, X=self.problem.x_max_task_num)
                        for value in s_value:
                            if value[0] == 0 and all(
                                np.array_equal(a, b) for a, b in zip(s_agg, value[2])
                            ):
                                V = value[3]

                                print(f"same {V=}")
                                break
                        else:
                            V = 0
                        result[S_idx][0][T_idx][Z_idx] = V
                        print(
                            f"{(S_idx,0,T_idx,Z_idx)=} {result[S_idx][0][T_idx][Z_idx]=}"
                        )
                    # 保存 result 到 CSV 文件
                    csv_filename = f"result_{self.problem.J}_{Z}_{T}_org.csv"

                    self.save_to_csv(
                        result, T_values, Z_values, S_n, file_name=csv_filename
                    )
        finally:
            # 将结果保存到文件
            np.save("result_org.npy", result)

            # 将结果保存到 CSV 文件
            self.save_to_csv(
                result, T_values, Z_values, S_n, file_name="./data/results_org.csv"
            )

            with open("./data/s_value_org.pkl", "wb") as file:
                pickle.dump(s_value, file)
            # 单独保存 s_value.s_values 字典为 JSON
            with open("./data/s_value_s_values_org.json", "w") as file:
                json.dump(s_value.s_values, file, indent=4)

    def init_a_problem(self, T=7, Z=3, J=10000):
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
            c2=20,
        )

        self.problem.all_task_init(J=J, T=T)
        print("problem init done")

    @timeit
    def run_VFA_task(self, T=7, Z=3, J=10000) -> SValue:
        s_value = self.problem.func8(T=T, J=J, Z_cluster_num=Z)
        return s_value

    @timeit
    def run_VFA_task_org(self, T, Z, J=10000) -> SValue:
        s_value = self.problem.func8_org(
            T=T,
            J=J,
            Z_cluster_num=Z,
        )
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

        # 获取文件夹路径
        folder_path = os.path.dirname(file_name)

        # 检查文件夹是否存在，如果不存在则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(file_name, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)


# %%
def test():
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
    c2 = 20
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
    print(f"func1 {S[0]=}\n {S[1]=}")

    barS = problem.func2(S=S, Z_cluster_num=Z_cluster_num, X=X)
    print(f"func2 {barS=}")

    problem.task_arr = problem.func7()
    A = problem.func3_transfer(S=S, V=0)
    print(f"func3_transfer {A=}")

    # func4 测试
    mathcal_L, N_1, N_2 = problem.func4(S=S)
    print(f"func4 {(mathcal_L, N_1, N_2, S)=}")

    # func5 测试
    Y = problem.func5(S, mathcal_L, N_1, N_2)
    print(f"func5 {Y=}")

    # func6 测试
    A_per_L_set = problem.func6(S, mathcal_L, N_1, N_2)
    print(f"func6 {A_per_L_set=}")

    profit = problem.Profit(S, A)
    print(f"测试收益函数 Profit {profit=}")

    arriving_tasks_i = problem.func7()
    print(f"func7 {arriving_tasks_i=}")

    problem.all_task_init(J=10000, T=7)
    save_S, pr = problem.nearest_distance(init_S=problem.init_S_J[0])
    print(f"{save_S}=, {pr=}")

    # func13 测试 single stage
    save_S, pr = problem.static_optimal(init_S=problem.init_S_J[0])
    print(f"{save_S}=, {pr=}")


# %%
# test()

# %%
task = TaskRunner()
# task.run()

# # %%
# task.run_org()

# %%

# task.run_test()

# %%
task.debug_profit()
# %%
# task.run_benchmark()
# %%
