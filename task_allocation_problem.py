# %%
import logging
from util.logging_config import setup_logging
import csv
import json
import os
import pickle
import random
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import thread_map

from my_wraps import timeit
from task_alloacate_problem import SValue, TaskAllocationProblem

# 装饰器用于计算函数调用时间

logger = setup_logging(__name__)
def save_to_csv(result, T_values, Z_values, S_n, file_name="./data/results.csv"):
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


class TaskRunner:

    city_num = 26
    DIST_MATRIX_PATH = r"./data/中国各城市空间权重矩阵(1).xlsx"
    PATH_CITY_DIST = f"./data/final/city_{city_num}.xlsx"

    # 本类的目标
    def __init__(self) -> None:
        pass

    def run_benchmark(self):
        T_values = [7, 14, 21]
        Z_values = [3, 5, 9]
        S_n = 5
        problem_n = 4
        result = np.zeros((S_n, problem_n, len(T_values), len(Z_values)))
        self.init_a_problem(T=7, J=10000)
        logging.info("sp done")
        for T_idx, T in enumerate(T_values):
            for Z_idx, Z in enumerate(Z_values):

                logging.info(f"-------{(T,Z)=}--------")
                self.init_a_problem(T=T, J=10000)
                for S_idx, s in enumerate(self.problem.init_S_J[0:5]):
                    # RA RDA MA
                    save_S, pr1 = self.problem.calc_total_reward_for_init_S_by_rnd(
                        init_S=s, T=T
                    )
                    save_S, pr2 = self.problem.nearest_distance(
                        init_S=s, T=T
                    )  # pr 是T个阶段的收益
                    save_S, pr3 = self.problem.static_optimal(init_S=s, T=T)
                    result[S_idx][1][T_idx][Z_idx] = sum(pr1)
                    result[S_idx][2][T_idx][Z_idx] = sum(pr2)
                    result[S_idx][3][T_idx][Z_idx] = sum(pr3)

                    logging.info(f"{(S_idx,1,T_idx,Z_idx)=} {result[S_idx][1][T_idx][Z_idx]=}")
                    logging.info(f"{(S_idx,2,T_idx,Z_idx)=} {result[S_idx][2][T_idx][Z_idx]=}")
                    logging.info(f"{(S_idx,3,T_idx,Z_idx)=} {result[S_idx][3][T_idx][Z_idx]=}")

        save_to_csv(
            result, T_values, Z_values, S_n, file_name="./data/benchmark_results.csv"
        )
        logging.info(f"----------------------finished benchmark---------------------")

    def run(self, T, Z, J, S_n, problem_n) -> SValue:
        # 初始化 result 四维矩阵
        try:
            logging.info(f"-------{(T,Z)=}--------")
            self.init_a_problem(T=T, Z=Z, J=J)
            s_value = self.run_VFA_task(T=T, Z=Z, J=J)

            result = np.zeros((S_n, problem_n, 1, 1))
            logging.info(f"done {self.problem.init_S_J[:5]}")
            for S_idx, s in enumerate(self.problem.init_S_J[0 : min(J, S_n)]):
                s_agg = self.problem.func2(s, Z_cluster_num=Z, X_max_task_num=self.problem.X_max_task_num)
                logging.info(f"{s_agg=}")
                result[S_idx][0][0][0] = s_value.get_total_reward(t=0, S_agg=s_agg)
                logging.info(f"{(T, Z, J)=} {(S_idx, 0, 0, 0)=} {result[S_idx][0][0][0]=}")

            # 保存 result 到 CSV 文件
            save_to_csv(
                result,
                [T],
                [Z],
                S_n,
                file_name=f"./data/result_per_iter/result_{J}_{Z}_{T}.csv",
            )

            s_value_memory = sys.getsizeof(s_value)
            logging.info(f"{(T,Z)=} s_value memory usage: {s_value_memory} bytes")
        finally:
            # 获取文件夹路径
            folder_path = os.path.dirname(f"./data/save_params/s_value_{T}_{Z}_{J}.pkl")

            # 检查文件夹是否存在，如果不存在则创建它
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(f"./data/save_params/s_value_{T}_{Z}_{J}.pkl", "wb") as file:
                pickle.dump(s_value, file)

            # 获取文件夹路径
            folder_path = os.path.dirname(
                f"./data/save_params/s_value_s_values_{T}_{Z}_{J}.json"
            )

            # 检查文件夹是否存在，如果不存在则创建它
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 单独保存 s_value.s_values 字典为 JSON
            with open(
                f"./data/save_params/s_value_s_values_{T}_{Z}_{J}.json", "w"
            ) as file:
                json.dump(s_value.s_values, file, indent=4)
        return s_value, result

    def init_a_problem(self, T=7, Z=3, J=10000):
        logging.info("problem init")
        I_citys = 26
        L_levels = 5
        W_workdays = 6
        M_servers = 40
        X_max_task_num = 3
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
            X_max_task_num=X_max_task_num,
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
        logging.info("problem init done")

    @timeit
    def run_VFA_task(self, T=7, Z=3, J=10000) -> SValue:
        s_value = self.problem.func8(T=T, J=J, Z_cluster_num=Z)
        return s_value


# %%


def process_task(args):
    T, Z, J, S_n, problem_n = args
    VFA_state_values = {}
    task = TaskRunner()
    s_value, result = task.run(
        T=T, Z=Z, J=J, S_n=S_n, problem_n=problem_n
    )
    VFA_state_values.update({(T, Z): s_value})
    return result

def test():
    J = 5
    S_n = 5
    problem_n = 4
    T_values = [7, 14, 21]
    Z_values = [3, 5, 9]
    result = np.zeros((S_n, problem_n, len(T_values), len(Z_values)))

    task_args = [
        (T, Z, J, S_n, problem_n,) for T in T_values for Z in Z_values
    ]
    from numpy import array
    
    task = TaskRunner()
    task.init_a_problem(T=7, Z=3, J=5)
    S1 = (array([[0, 3, 0, 0, 1],
       [0, 0, 2, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [1, 0, 0, 0, 3],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 0, 1],
       [0, 2, 0, 2, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 1, 1],
       [0, 0, 3, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 2, 0, 0],
       [1, 0, 1, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0],
       [0, 0, 2, 0, 1],
       [1, 2, 0, 0, 0],
       [0, 1, 1, 0, 0],
       [0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 2, 2, 1],
       [1, 1, 1, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 1, 1, 0, 0]]), [(9, 5), (6, 2), (15, 3), (15, 1), (19, 0), (16, 3), (0, 1), (11, 5), (4, 4), (22, 2), (8, 0), (18, 2), (19, 2), (23, 5), (23, 0), (10, 3), (7, 0), (5, 3), (2, 3), (24, 5), (24, 2), (17, 4), (17, 5), (21, 1), (15, 2), (8, 4), (0, 3), (0, 3), (20, 5), (23, 3), (16, 2), (15, 0), (18, 3), (22, 5), (13, 5), (5, 5), (12, 5), (21, 2), (1, 2), (0, 4)])
    S_agg = task.problem.func2(S1, Z_cluster_num=3, X_max_task_num=task.problem.X_max_task_num)
    print(f"final {S_agg=}")
    print("(T, Z, J, S_n, problem_n,)", task_args[0])
    process_task(task_args[0])
    

def main():
    J = 5
    S_n = 5
    problem_n = 4
    T_values = [7, 14, 21]
    Z_values = [3, 5, 9]
    result = np.zeros((S_n, problem_n, len(T_values), len(Z_values)))

    task_args = [
        (T, Z, J, S_n, problem_n) for T in T_values for Z in Z_values
    ]

    try:
        results = thread_map(
            process_task, task_args, max_workers=1
        )  # 调整 max_workers 根据你的 CPU 核数

        for i, (T, Z, _, _, _, _) in enumerate(task_args):
            T_idx = T_values.index(T)
            Z_idx = Z_values.index(Z)
            result[:, :, T_idx, Z_idx] = results[i][:, :, 0, 0]

    finally:
        # 将结果保存到 CSV 文件
        save_to_csv(
            result, T_values, Z_values, S_n, file_name="./data/final/results.csv"
        )


if __name__ == "__main__":
    test()