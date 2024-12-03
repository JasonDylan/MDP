import itertools
from collections import Counter

import numpy as np
import pandas as pd
from allocate_cls import allocate
import random



class CityDistanceManager:
    # 本类是城市距离管理类，用于获取将城市对应的其他城市距离排序
    def __init__(self, city_id_distance_df):
        self.city_id_distance_df = city_id_distance_df
        self.city_ids = city_id_distance_df.columns.tolist()
        self.city_id_2_sorted_cities = [self.get_sorted_cities(city_id=city_id) for city_id in self.city_ids] 

    def get_sorted_cities(self, city_id):
        # 本函数用于获取一个城市对于其他城市距离的排序
        distance = self.city_id_distance_df[city_id]
        sorted_indices = np.argsort(distance)
        sorted_cities = [(distance.index[idx], distance.iloc[idx]) for idx in sorted_indices]
        return sorted_cities

    def get_nearest_cities(self, city_id, n=3):
        # 本函数用于获取最近的3个城市
        nearest_cities = self.city_id_2_sorted_cities[city_id][1:n+1]
        return nearest_cities
    
    def get_nearest_cities_for_select_cities(self, city_id, task_city_ids, n=3):
        # 本函数用于获取指定的城市中，最近的几个城市
        sorted_cities = self.city_id_2_sorted_cities[city_id]
        nearest_cities = [city for city in sorted_cities if city[0] in task_city_ids][:n]
        return nearest_cities
    
def get_combinations(
    server_city_id_list,
    task_city_id_list,
    current_combination,
    min_cost_sum=float("inf"),
    allocate_else=0,
    server_and_task_combination_list=[],
    need_comb_num=None,
    a_city_distance_df=None,
):
    # 本函数用于递归获取，对于server_list和city_list的所有业务员和城市的匹配方式，复杂度太高，已经停止使用
    if (
        len(server_city_id_list) == 0
        or len(task_city_id_list) == 0
        # or allocate_else == 0
    ):
        assert current_combination != None
        # TODO 取消存储防止爆内存
        current_sum = sum(
            [a_city_distance_df[server][city] for server, city in current_combination]
        )
        if current_sum < min_cost_sum:
            min_cost_sum = current_sum
            server_and_task_combination_list = [current_combination]
        # elif current_sum == min_cost_sum:
        #     server_and_task_combination_list.append(
        #         current_combination
        #     )
        if need_comb_num is not None:
            # Filter min_cost_city_id_of_server_and_task_combination to contain only the first need_comb_num combinations
            pairs = itertools.combinations(
                min_cost_city_id_of_server_and_task_combination, need_comb_num
            )
            new_min_cost_sum = float("inf")
            new_min_cost_city_id_of_server_and_task_combination = None
            for pair in pairs:
                current_sum = sum(
                    [a_city_distance_df[server][city] for server, city in pair]
                )
                if current_sum < new_min_cost_sum:
                    new_min_cost_sum = current_sum
                    new_min_cost_city_id_of_server_and_task_combination = pair
            min_cost_sum = new_min_cost_sum
            min_cost_city_id_of_server_and_task_combination = (
                new_min_cost_city_id_of_server_and_task_combination
            )
        return min_cost_sum, server_and_task_combination_list

    for i, city in enumerate(task_city_id_list):
        remaining_cities = task_city_id_list[:i] + task_city_id_list[i + 1 :]
        new_current_combination = current_combination + [(server_city_id_list[0], city)]
        (
            min_cost_sum,
            server_and_task_combination_list,
        ) = get_combinations(
            server_city_id_list[1:],
            remaining_cities,
            new_current_combination,
            min_cost_sum,
            allocate_else - 1,
            server_and_task_combination_list,
            need_comb_num,
            a_city_distance_df,
        )

    return min_cost_sum, server_and_task_combination_list


def get_revenue_by_task_and_num(task_lv, num):
    # 本函数用于根据任务数量和等级，计算收益
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    return revenue_for_lv[task_lv - 1] * num


def get_selected_task_lv_city_id_list(task_lv, a_task_df):
    """
    获取指定等级为task_lv的任务所在城市id列表,并根据任务数量重复城市id

    输入:
    - task_lv: int,指定的任务等级
    - a_task_df: DataFrame,任务数据框,包含每个城市不同等级的任务数量

    输出:
    - selected_task_city_id_list_repeated: list,指定等级的任务所在城市id列表,根据任务数量重复城市id

    注意:
    - 函数通过筛选a_task_df中对应等级(level task_lv)的任务数量大于等于1的城市
    - 从筛选后的数据框的索引中提取城市id,并转换为整数列表
    - 根据筛选后的任务数量,重复对应的城市id,生成最终的城市id列表
    """
    # 本函数用于获取指定level为task_lv的task的城市id
    # remain_servers_df
    # 对于一个decisions 内的单个allocation，进行分配
    # 输入为指定task_lv的城市位置，有指定server_lv对应的server位置, 和分配数量
    # 然后得到城市的位置range和server的位置range， 随机匹配allocate_num的个数量，约束是如果以及分配完的目的地不应该再次分配
    # task_lv = 3  # 指定的level
    this_lv = f"level {task_lv}"
    selected_task_city_df = a_task_df.loc[a_task_df[this_lv] >= 1][
        this_lv
    ]  # 选择出当前 lv 满足task_lv要求的城市

    selected_task_city_idx = selected_task_city_df.index  # task 城市idx
    selected_task_city_id_list = list(
        selected_task_city_idx.str.replace("city ", "").astype(int)
    )  # task 城市id
    # 根据值大于2的城市生成对应数量的重复元素，用于计算分配
    selected_task_city_id_list_repeated = list(
        np.repeat(selected_task_city_id_list, selected_task_city_df.values)
    )
    return selected_task_city_id_list_repeated


def get_selected_selected_server_city_n_id_list(server_lv, remain_servers_df):
    """
    获取指定等级为server_lv的业务员的城市id列表和业务员id列表

    输入:
    - server_lv: int,指定的业务员等级
    - remain_servers_df: DataFrame,剩余业务员数据框,包含业务员等级(lv)和当前所在城市(current_city)

    输出:
    - selected_server_id_list: list,指定等级的业务员id列表
    - selected_server_city_id_list: list,指定等级的业务员所在城市id列表

    注意:
    - 函数通过筛选remain_servers_df中lv等于server_lv的行来获取指定等级的业务员信息
    - 从筛选后的数据框中提取current_city列获得业务员所在城市id列表
    - 使用筛选后的数据框的索引并进行字符串处理获得业务员id列表
    """
    # 本函数用于获取指定level为server_lv的server的城市id_list
    # server_lv = 1
    selected_server_city_df = remain_servers_df[remain_servers_df["lv"] == server_lv][
        "current_city"
    ]  # 选择 lv 满足 server_lv 的server
    selected_server_city_id_list = list(selected_server_city_df)  # server 城市 id
    selected_server_idx = selected_server_city_df.index  # server idx
    selected_server_id_list = list(
        selected_server_idx.str.replace("server ", "").astype(int)
    )  # server id

    return selected_server_id_list, selected_server_city_id_list

def combine_cities(cities, path=[], all_paths=[]):
    """
    根据给定的分配元组,将业务员分配到最近的3个城市,并返回最大收益和分配方案

    输入:
    - allocate_tuple: tuple,分配元组,包含任务等级、业务员等级和分配数量(task_lv, server_lv, allocate_num)
    - a_task_df: DataFrame,任务数据框,包含每个城市不同等级的任务数量
    - remain_servers_df: DataFrame,剩余业务员数据框,包含每个城市不同等级的业务员数量
    - a_city_distance_df: DataFrame,城市距离数据框,包含城市之间的距离信息

    输出:
    - final_revenue: float,最大收益,即完成任务获得的奖励减去业务员出差的费用
    - new_server_and_task_combination_list: list,最优分配方案列表,每个元素为一个列表,包含业务员编号、业务员城市、分配去的城市编号、城市等级和业务员等级

    注意:
    - 函数根据分配元组中的任务等级和业务员等级,从任务数据框和剩余业务员数据框中获取满足条件的城市列表
    - 使用CityDistanceManager获取每个业务员城市最近的3个任务城市
    - 使用combine_cities函数生成所有可能的分配方案
    - 最终返回收益和分配方案列表
    - 该函数通过选择最近的3个城市来降低计算复杂度
    """
    if not cities:
        all_paths.append(path)
        return all_paths

    for city_id in cities[0]:
        if city_id not in path:
            # 继续递归，选择下一个城市ID
            combine_cities(cities[1:], path + [city_id], all_paths)

    return all_paths

# 任务等级，业务员等级，数量
# task_lv, server_lv, allocate_num
# 分配指定等级对应的城市和业务员位置
def allocate_servers_2_cities_for_a_decision_nearest_n_city(
    allocate_tuple, a_task_df, remain_servers_df, a_city_distance_df
):
    
    # 本函数用于将一个组合的分配方式，获取最近的3个城市的方式，来降低复杂度，
    # 并组合成 [业务员id，业务员城市id，分配去的城市id，分配去的城市的任务的等级，业务员任务等级]

    task_lv, server_lv, allocate_num = allocate_tuple
    # 完成tasklv的任务可以获得的奖励
    revenue = get_revenue_by_task_and_num(task_lv, allocate_num)
    min_cost_sum = 0

    # 获取指定level为task_lv的task的城市id
    selected_task_city_id_list_repeated = get_selected_task_lv_city_id_list(
        task_lv=task_lv, a_task_df=a_task_df
    )
    # 获取指定level为server_lv的server的城市id
    (
        selected_server_id_list,
        selected_server_city_id_list,
    ) = get_selected_selected_server_city_n_id_list(
        server_lv=server_lv, remain_servers_df=remain_servers_df
    )

    server_city_id_2_server_id = {
        city: idx
        for city, idx in zip(selected_server_city_id_list, selected_server_id_list)
    }  # 城市id转业务员id ， 用于生成分配策略

    # 初始化 CityDistanceManager
    manager = CityDistanceManager(a_city_distance_df, )
    assignment_combinations = []
    nearest_cities_2_task_city_id_list = []
    # nearest_cities_2_task_city_id_list:[[13, 6, 20], [16, 6, 13], [16, 6, 14], [16, 6, 14]]
    nearest_cities_2_task_city_list = [manager.get_nearest_cities(server_city_id, selected_task_city_id_list_repeated, n=3) for server_city_id in selected_server_city_id_list]
    nearest_cities_2_task_city_id_list = [[id for id, dist in nearest_cities ] for nearest_cities in nearest_cities_2_task_city_list]
    
    all_combinations = combine_cities(nearest_cities_2_task_city_id_list)
    server_and_task_combination_list = [ (server, city) for cities in all_combinations for server,city in zip(selected_server_city_id_list, nearest_cities_2_task_city_id_list)]
    final_revenue = revenue - min_cost_sum
    
    # 分配后应该更新server 位置 为城市
    # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
    server_and_task_combination_list = assignment_combinations
    # 存到内存
    new_server_and_task_combination_list = []
    for (
        min_cost_city_id_of_server_and_task_combination
    ) in server_and_task_combination_list:
        new_min_cost_city_id_of_server_and_task_combination = []
        for server_city_id, city_id in min_cost_city_id_of_server_and_task_combination:
            new_min_cost_city_id_of_server_and_task_combination.append(
                [
                    server_city_id_2_server_id[server_city_id],
                    server_city_id,
                    city_id,
                    task_lv,
                    server_lv,
                ]
            )
            # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
        new_server_and_task_combination_list.append(
            new_min_cost_city_id_of_server_and_task_combination
        )

    # todo ，现在修改的后一个list 是很多的组合，revenue不是对应一个
    return final_revenue, new_server_and_task_combination_list 

# 任务等级，业务员等级，数量
# task_lv, server_lv, allocate_num
# 分配指定等级对应的城市和业务员位置
def allocate_servers_2_cities_for_a_decision(
    allocate_tuple, a_task_df, remain_servers_df, a_city_distance_df
):
    """
    根据给定的分配元组,将业务员分配到对应的城市,并返回最大收益和分配方案

    输入:
    - allocate_tuple: tuple,分配元组,包含任务等级、业务员等级和分配数量(task_lv, server_lv, allocate_num)
    - a_task_df: DataFrame,任务数据框,包含每个城市不同等级的任务数量
    - remain_servers_df: DataFrame,剩余业务员数据框,包含每个城市不同等级的业务员数量
    - a_city_distance_df: DataFrame,城市距离数据框,包含城市之间的距离信息

    输出:
    - final_revenue: float,最大收益,即完成任务获得的奖励减去业务员出差的费用
    - new_server_and_task_combination_list: list,最优分配方案列表,每个元素为一个列表,包含业务员编号、业务员城市、分配去的城市编号、城市等级和业务员等级

    注意:
    - 函数根据分配元组中的任务等级和业务员等级,从任务数据框和剩余业务员数据框中获取满足条件的城市列表
    - 使用get_combinations函数生成所有可能的分配方案,并计算每个方案的最小成本
    - 最终返回收益最大的分配方案和对应的收益值
    - 该函数已废弃,仅供参考
    """
    # 本函数用于将一个组合的分配方式，递归方式获取所有分配，
    # 并组合成 [业务员id，业务员城市id，分配去的城市id，分配去的城市的任务的等级，业务员任务等级]
    # 已废弃

    task_lv, server_lv, allocate_num = allocate_tuple
    # if allocate_num > 1:
    # 完成tasklv的任务可以获得的奖励
    revenue = get_revenue_by_task_and_num(task_lv, allocate_num)
    min_cost_sum = 0

    # 获取指定level为task_lv的task的城市id
    selected_task_city_id_list_repeated = get_selected_task_lv_city_id_list(
        task_lv=task_lv, a_task_df=a_task_df
    )
    # 获取指定level为server_lv的server的城市id
    (
        selected_server_id_list,
        selected_server_city_id_list,
    ) = get_selected_selected_server_city_n_id_list(
        server_lv=server_lv, remain_servers_df=remain_servers_df
    )
    #
    server_city_id_2_server_id = {
        city: idx
        for city, idx in zip(selected_server_city_id_list, selected_server_id_list)
    }  # 城市id转业务员id，用于生成分配策略

    # 获取业务员所在城市的列表
    # 生成所有城市-业务员位置的排列组合
    # 获取 满足task_lv要求的城市 和 满足 server_lv 的server 条件下的所有分配，
    (
        min_cost_sum,
        server_and_task_combination_list,
    ) = get_combinations(
        selected_server_city_id_list,
        selected_task_city_id_list_repeated,
        current_combination=[],
        a_city_distance_df=a_city_distance_df,
        allocate_else=allocate_num,
    )
    # get_nearest_combinations()
    # else:
    #     min_cost_city_id_of_server_and_task_combination = [(task_lv, server_lv)]
    #     min_cost_sum  =
    # print("min Combination:", min_cost_city_id_of_server_and_task_combination) # cost最小的组合
    # print("min Sum:", min_cost_sum)
    final_revenue = revenue - min_cost_sum

    # 分配后应该更新server 位置 为城市
    # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']

    # 存到内存
    new_server_and_task_combination_list = []
    # if min_cost_city_id_of_server_and_task_combination:
    #     print(min_cost_city_id_of_server_and_task_combination)
    # else:
    #     print(f"{min_cost_city_id_of_server_and_task_combination=}")
    for (
        min_cost_city_id_of_server_and_task_combination
    ) in server_and_task_combination_list:
        new_min_cost_city_id_of_server_and_task_combination = []
        for server_city_id, city_id in min_cost_city_id_of_server_and_task_combination:
            new_min_cost_city_id_of_server_and_task_combination.append(
                [
                    server_city_id_2_server_id[server_city_id],
                    server_city_id,
                    city_id,
                    task_lv,
                    server_lv,
                ]
            )
            # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
        new_server_and_task_combination_list.append(
            new_min_cost_city_id_of_server_and_task_combination
        )

    return final_revenue, new_server_and_task_combination_list


def allocate_servers_2_cities(
    a_decision: list[tuple],
    initial_task_df,
    remain_servers_df,
    a_city_distance_df,
) -> dict:
    """
    本函数用于根据decisions集合分组情况/城市状态df/用户状态df,分配所有情况

    输入:
    - a_decision: list[tuple],一个决策列表,每个元素为一个元组(任务等级,业务员等级,数量)
    - initial_task_df: DataFrame,初始任务数据框
    - remain_servers_df: DataFrame,剩余业务员数据框
    - a_city_distance_df: DataFrame,城市距离数据框

    输出:
    - revenue_and_combination_for_decision: dict,包含收益和组合的字典
      - 'revenue': float,总收益
      - 'combination': list[tuple],所有分配的组合,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)
    """
    revenue_and_combination_for_decision = {}

    # 对于组内的decision 返回的应该是多种分配
    revenue_sum = 0
    all_combination_for_a_decision = []
    a_task_df = initial_task_df.copy()

    allocate_servers_2_cities_for_a_decision_list = [
        allocate_servers_2_cities_for_a_decision_nearest_n_city(
            a_allocate,
            # (4, 2, 1),
            a_task_df,
            remain_servers_df,
            a_city_distance_df,
        )
        for a_allocate in a_decision
    ]
    # todo 这里现在是选择了0，但是需要笛卡尔积获得所有情况
    # 然后获取所有分配情况后，生成分配获取收益
    revenue_sum = sum(
        [item[0] for item in allocate_servers_2_cities_for_a_decision_list]
    )  # 对首个加和
    all_combination_for_a_decision = []
    for (
        a_revenue_for_a_allocate,
        combination_for_a_allocate_list,
    ) in allocate_servers_2_cities_for_a_decision_list:
        combination_for_a_allocate = combination_for_a_allocate_list[0]
        if len(combination_for_a_allocate_list) > 1:
            print(combination_for_a_allocate_list)
        all_combination_for_a_decision += combination_for_a_allocate

    # for a_allocate in a_decision:
    #     (
    #         a_revenue_for_a_allocate,
    #         combination_for_a_allocate,
    #     ) = allocate_servers_2_cities_for_a_decision_nearest_n_city(
    #         a_allocate, a_task_df, remain_servers_df, a_city_distance_df
    #     )
    #     print(a_allocate, a_revenue_for_a_allocate, combination_for_a_allocate)
    #     revenue_sum += a_revenue_for_a_allocate
    #     all_combination_for_a_decision.extend(combination_for_a_allocate)

    revenue_and_combination_for_decision = {
        "revenue": revenue_sum,
        "combination": all_combination_for_a_decision,
    }

    return revenue_and_combination_for_decision


def get_all_allocations_for_decisions(allocations_for_decisions: dict) -> list[dict]:
    """
    有了每个组合内的分配,开始合并不同组之间的决策,并reduce并获取V历史记录的对应的收益。

    输入:
    - allocations_for_decisions: dict,每个组合内的分配情况
      - 键: int,分组的索引
      - 值: dict,包含收益和组合的字典
        - 'revenue': float,总收益
        - 'combination': list[tuple],所有分配的组合,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)

    输出:
    - final_revenue_n_combination_list: list[dict],最终的收益和组合列表
      - 每个元素为一个字典,包含:
        - 'final_revenue': float,最终的总收益
        - 'final_combination': list[tuple],最终的分配组合,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)
    """
    num_combinations = 1
    for key, value in allocations_for_decisions.items():
        num_combinations *= len(value)

    # Print the number of possible combinations
    # print(f"{num_combinations=}")

    # Get all possible combinations of values from the dictionary
    join_combination = list(itertools.product(*allocations_for_decisions.values()))
    #
    final_revenue_n_combination_list = []
    for a_join_comb in join_combination:
        final_revenue = 0
        final_combination = []

        for idx, join_idx in enumerate(a_join_comb):
            final_revenue += allocations_for_decisions[idx][join_idx]["revenue"]
            final_combination.extend(
                allocations_for_decisions[idx][join_idx]["combination"]
            )
        final_revenue_n_combination_list.append(
            {"final_revenue": final_revenue, "final_combination": final_combination}
        )
    return final_revenue_n_combination_list


def generate_tasks(arriving_rate):
    """
    根据给定的到达率生成随机的任务数量

    输入:
    - arriving_rate: float,任务的到达率,表示平均每个时间单位到达的任务数量

    输出:
    - tasks: int,生成的随机任务数量

    注意:
    - 函数使用泊松分布(Poisson distribution)生成随机的任务数量
    - 泊松分布适用于描述单位时间内随机事件发生的次数,这里用于模拟任务的到达过程
    """
    # 使用numpy的random.poisson函数生成一个随机的任务数量
    # 泊松分布的参数为arriving_rate,表示平均每个时间单位到达的任务数量
    tasks = np.random.poisson(arriving_rate)
    return tasks


def update_state(state_df, tasks):
    """
    更新状态数据框,将新的任务数量添加到原有的状态中

    输入:
    - state_df: DataFrame,原有的状态数据框,行索引为城市,列索引为任务等级
    - tasks: list,新的任务列表,每个元素为一个元组(city, level, num)

    输出:
    - updated_task_df: DataFrame,更新后的状态数据框

    注意:
    - 函数会创建一个新的数据框updated_task_df,复制原有状态数据框的内容
    - 根据新的任务列表tasks创建一个临时的数据框tasks_df,与原有状态数据框具有相同的行列索引
    - 遍历原有状态数据框的每个城市和任务等级,将对应的新任务数量累加到updated_task_df中
    """
    updated_task_df = state_df.copy()

    tasks_df = pd.DataFrame(tasks, index=state_df.index, columns=state_df.columns)
    for city in state_df.index:
        for lv in state_df.columns:
            updated_task_df.loc[city, lv] += tasks_df.loc[city, lv]
    return updated_task_df


def reduce_task_df(a_task_df, proveng_dict, city_num_2_name):
    # 默认是有两列index，一列省份一列市的，缩减为省份的
    """
    将任务数据框缩减为省份级别

    输入:
    - a_task_df: DataFrame,原始的任务数据框
    - proveng_dict: dict,城市名称到省份的映射字典
    - city_num_2_name: dict,城市编号到城市名称的映射字典

    输出:
    - final_reduced_task_df: DataFrame,缩减后的任务数据框,按省份聚合

    注意:
    - 函数会创建一个新的数据框reduced_task_df,并根据城市编号和城市名称映射获取对应的省份作为新的索引
    - 最终返回一个按省份聚合后的数据框final_reduced_task_df
    """
    new_index = [
        proveng_dict[city_num_2_name[int(city[5:])][0]] for city in a_task_df.index
    ]

    # 将新的索引列表赋值给a_task_df的索引
    reduced_task_df = a_task_df.copy()
    reduced_task_df.index = new_index

    # print("New a_task_df:")
    # print(f"{reduced_task_df=}")
    final_reduced_task_df = reduced_task_df.groupby(reduced_task_df.index).sum()

    return final_reduced_task_df


def reduce_server_df(a_servers_df, proveng_dict, city_num_2_name):
    # 默认是有两列index，一列省份一列市的，缩减为省份的
    
    """
    将业务员数据框缩减为省份级别

    输入:
    - a_servers_df: DataFrame,原始的业务员数据框
    - proveng_dict: dict,城市名称到省份的映射字典
    - city_num_2_name: dict,城市编号到城市名称的映射字典

    输出:
    - city_prov_tuples: list,缩减后的业务员省份列表,每个元素为一个元组(server_id, province)

    注意:
    - 函数会创建一个新的数据框reduced_servers_df,并根据城市编号和城市名称映射获取对应的省份
    - 最终返回一个列表,其中每个元素为一个元组(server_id, province)
    """
    reduced_servers_df = a_servers_df.copy()
    reduced_servers_df["current_prov"] = (
        reduced_servers_df["current_city"]
        .map(city_num_2_name)
        .map(lambda x: proveng_dict[x[0]])
    )
    city_prov_tuples = [
        (int(idx[7:]), row["current_prov"])
        for idx, row in reduced_servers_df.iterrows()
    ]

    return city_prov_tuples


# 根据最优解的combination 求得业务员之后的位置
# columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
def update_server_cities(servers_df, allocation_for_a_day):
    """
    根据当天的业务员分配情况更新业务员所在城市

    输入:
    - servers_df: DataFrame,原始的业务员数据框
    - allocation_for_a_day: list,当天的业务员分配情况,每个元素为一个元组(server_id, server_org_city, server_to_city, city_lv, server_lv)

    输出:
    - new_servers_df: DataFrame,更新后的业务员数据框

    注意:
    - 函数会创建一个新的数据框new_servers_df,并更新其中的"current_city"列
    - 对于每个分配,函数会根据server_id找到对应的业务员,并将其"current_city"更新为分配的目标城市server_to_city
    """
    new_servers_df = servers_df.copy()
    new_servers_df.columns = ["current_city", "lv", "day off"]
    for allocation in allocation_for_a_day:
        server_id, server_org_city, server_to_city, city_lv, server_lv = (
            allocation[0],
            allocation[1],
            allocation[2],
            allocation[3],
            allocation[4],
        )
        new_servers_df["current_city"][f"server {server_id}"] = server_to_city

    return new_servers_df


def get_allocation_for_a_day(
    final_revenue_n_combination_list,
    remain_servers_df,
    a_task_df,
    arriving_rate_df,
    weekday,
    proveng_dict,
    city_num_2_name,
    reduce_V,
    reduce_V_iter,
    reduce_V_actual,
    a_iter,
) -> (float, list[tuple]):
    """
    本函数用于获取最好的分配

    输入:
    - final_revenue_n_combination_list: list[dict],最终的收益和组合列表
      - 每个元素为一个字典,包含:
        - 'final_revenue': float,最终的总收益
        - 'final_combination': list[tuple],最终的分配组合,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)
    - remain_servers_df: DataFrame,剩余业务员数据框
    - a_task_df: DataFrame,任务数据框
    - arriving_rate_df: DataFrame,到达率数据框
    - weekday: int,星期几(0-6)
    - proveng_dict: dict,省份字典
    - city_num_2_name: dict,城市编号到名称的映射
    - reduce_V: dict,缩减后的状态值函数
    - reduce_V_iter: int,缩减的迭代次数
    - reduce_V_actual: dict,实际的缩减状态值函数
    - a_iter: int,当前的迭代次数

    输出:
    - max_revenue: float,最大的总收益
    - allocation_for_a_day: list[tuple],一天的最优分配,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)
    """
    allocation_for_a_day = []
    # 做决策了这里就是要，挑一个最大的，同时要把state的值压缩了之后做保存，当前state情况的最优决策
    max_revenue = 0
    max_combination = []
    for revenue_n_combination in final_revenue_n_combination_list:
        revenue = revenue_n_combination["final_revenue"]
        combination = revenue_n_combination["final_combination"]

        a_servers_df = remain_servers_df
        final_allocation_for_a_day = combination

        new_task_df, new_servers_df, allocate_task_df = allocate_reduce_df(
            final_allocation_for_a_day, a_task_df, arriving_rate_df, a_servers_df
        )
        reduceV_revenue = get_a_state_revenue(
            a_servers_df=a_servers_df,
            new_servers_df=new_servers_df,
            new_task_df=new_task_df,
            a_task_df=a_task_df,
            allocate_task_df=allocate_task_df,
            reduce_V=reduce_V,
            reduce_V_iter=reduce_V_iter,
            reduce_V_actual=reduce_V_actual,
            weekday=weekday,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
            a_iter=a_iter,
        )
        if reduceV_revenue != 0:
            print(f"{reduceV_revenue=}")

        if max_revenue < revenue + reduceV_revenue:
            max_revenue = revenue + reduceV_revenue
            max_combination = combination

    allocation_for_a_day = max_combination
    return max_revenue, allocation_for_a_day


def allcocate_comb_2_allocate_task_df(
    final_allocation_for_a_day: list[tuple], org_task_df: pd.DataFrame
) -> pd.DataFrame:
    """
    获取指定状态的收益值

    输入:
    - a_servers_df: DataFrame,原始业务员数据框
    - new_servers_df: DataFrame,更新后的业务员数据框(业务员分配后的位置)
    - new_task_df: DataFrame,更新后的任务数据框(分配后新增任务)
    - a_task_df: DataFrame,原始任务数据框
    - allocate_task_df: DataFrame,分配的任务数据框
    - reduce_V: dict,缩减后的状态值函数
    - reduce_V_iter: dict,缩减后的状态值函数(按迭代次数)
    - reduce_V_actual: dict,实际的缩减状态值函数
    - weekday: int,星期几(1-7)
    - proveng_dict: dict,城市编号到省份的映射
    - city_num_2_name: dict,城市编号到名称的映射
    - a_iter: int,当前的迭代次数

    输出:
    - revenue: float,指定状态的收益值

    注意:
    - 状态由缩减后的业务员状态和新任务状态组成
    - 如果状态在实际的缩减状态值函数中存在,则返回对应的收益值,否则返回0
    """
    
    new_task_df = org_task_df.copy()
    new_task_df.fillna(0, inplace=True)
    # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
    for allocation in final_allocation_for_a_day:
        city_num = allocation[2]
        city_level = allocation[3]
        new_task_df.loc[f"city {city_num}", f"level {city_level}"] = 1

    return new_task_df


def allocate_servers_2_citys_MDP(
    remain_servers_df,
    a_task_df,
    a_city_distance_df,
    arriving_rate_df,
    weekday,
    proveng_dict,
    city_num_2_name,
    reduce_V,
    reduce_V_iter,
    reduce_V_actual,
    a_iter,
):
    '''
    将业务员根据算法分配到城市里

    输入:
    - remain_servers_df: DataFrame,剩余的业务员数据
    - a_task_df: DataFrame,任务数据
    - a_city_distance_df: DataFrame,城市距离数据
    - arriving_rate_df: DataFrame,到达率数据
    - weekday: int,星期几(0-6)
    - proveng_dict: dict,省份字典
    - city_num_2_name: dict,城市编号到名称的映射
    - reduce_V: dict,缩减后的状态值函数
    - reduce_V_iter: int,缩减的迭代次数
    - reduce_V_actual: dict,实际的缩减状态值函数
    - a_iter: int,当前的迭代次数

    输出:
    - final_revenue: float,最终的收益
    - final_allocation_for_a_day: list,一天的最终分配方案,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)
   
    '''
    # 根据 lv 来做分组
    join_idx_2_task_lv_server_lv_num = generate_idx_2_joins(
        a_task_df, remain_servers_df
    )
    # print(f"{join_idx_2_task_lv_server_lv_num=}")
    cnt = 0
    # print("（业务员编号id，业务员城市，分配去的城市编号，城市等级，业务员等级）")

    join_idx_2_revenue_and_combination_list_dict = {}
    revenue_sum = 0
    # 对于每一个集合，进行集合内分配
    for (
        join_idx,
        decisions_for_each_joins,
    ) in join_idx_2_task_lv_server_lv_num.items():
        cnt += 1
        # 根据集合分组后的，集合内level的数量分配
        # print("----join_idx, decisions_for_each_join----", join_idx, decisions_for_each_joins)
        # decisions_for_each_join = [[(1, 1, 4)]]
        revenue_and_combination_list_for_a_join = {}

        for idx, a_decision in enumerate(decisions_for_each_joins):
            # 集合分组后，level的分配方式可能很多, 一个decision里面是一个分配方案[(1, 1, 4)]
            # 可能是这样的 [(1, 1, 4)] / [(1, 1, 2), (2, 2, 2)]
            # 根据数量分配，进行实际的城市分配方案 list下的一个list是
            # print(f"{idx=} {a_decision=}")
            revenue_and_combination_for_decision = allocate_servers_2_cities(
                a_decision, a_task_df, remain_servers_df, a_city_distance_df
            )
            # revenue_and_combination_for_decision = {'revenue':final_revenue, 'combination':min_cost_city_id_of_server_and_task_combination}
            revenue_and_combination_list_for_a_join.update(
                {idx: revenue_and_combination_for_decision}
            )

        join_idx_2_revenue_and_combination_list_dict.update(
            {join_idx: revenue_and_combination_list_for_a_join}
        )

    # 有了每个组合内的分配，开始合并不同组之间的决策，并reduce 并获取V历史记录的对应的收益。todo 可能需要修改计算revenue的
    final_revenue_n_combination_list = get_all_allocations_for_decisions(
        join_idx_2_revenue_and_combination_list_dict
    )

    # 获取最好的分配
    final_revenue, final_allocation_for_a_day = get_allocation_for_a_day(
        final_revenue_n_combination_list=final_revenue_n_combination_list,
        remain_servers_df=remain_servers_df,
        a_task_df=a_task_df,
        arriving_rate_df=arriving_rate_df,
        weekday=weekday,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
        reduce_V=reduce_V,
        reduce_V_iter=reduce_V_iter,
        reduce_V_actual=reduce_V_actual,
        a_iter=a_iter,
    )
    return final_revenue, final_allocation_for_a_day


# def get_revenue_by_task_and_num(task_lv, num):
#     revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
#     return revenue_for_lv[task_lv - 1] * num
def rnd_allocate_servers_2_citys(remain_servers_df, a_task_df, a_city_distance_df):
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    allocation_data = []
    cities_remain = a_task_df.index.tolist()
    max_level = int(a_task_df.columns.max()[6:])
    final_allocation_for_a_day = []
    final_revenue = 0
    for server in remain_servers_df.index:
        if len(cities_remain) == 0:
            break
        server_id = int(server[7:])
        server_level = remain_servers_df.loc[server, "lv"]
        server_city = remain_servers_df.loc[server, "current_city"]
        available_cities = [
            (city, level)
            for city in cities_remain
            for level in range(server_level, max_level + 1)
            if a_task_df.loc[city, f"level {level}"] == 1
        ]
        if len(available_cities) == 0:
            continue

        # print(f"{server_level=} {available_cities=}")
        city = random.choice(available_cities)
        task_level = city[1]
        task_city = int(city[0][5:])
        revenue = revenue_for_lv[task_level - 1]
        cost = a_city_distance_df[server_city][task_city]

        final_revenue += revenue - cost
        final_allocation_for_a_day.append(
            [server_id, server_city, task_city, task_level, server_level]
        )
        # print(f"Server: {server}, Level: {server_level}, City: {city}, City Level: {city[1]}")
        # ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
        allocation_data.append({"server": server, "city": city})
        cities_remain.remove(city[0])

    allocation_df = pd.DataFrame(allocation_data, columns=["server", "city"])

    return final_revenue, final_allocation_for_a_day


def nearest_distance_allocate_servers_2_citys(
    remain_servers_df, a_task_df, a_city_distance_df
):
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    allocation_data = []
    cities_remain = a_task_df.index.tolist()
    max_level = int(a_task_df.columns.max()[6:])
    final_allocation_for_a_day = []
    final_revenue = 0
    for server in remain_servers_df.index:
        if len(cities_remain) == 0:
            break

        server_id = int(server[7:])
        server_level = remain_servers_df.loc[server, "lv"]
        server_city = remain_servers_df.loc[server, "current_city"]
        available_cities = [
            (city, level)
            for city in cities_remain
            for level in range(server_level, max_level + 1)
            if a_task_df.loc[city, f"level {level}"] == 1
        ]
        if len(available_cities) == 0:
            continue

        # print(f"{server_level=} {available_cities=}")
        # city = find_nearest_distance()

        min_dist = float("inf")
        min_city = None
        for eval_city in available_cities:
            eval_city_id = int(eval_city[0][5:])
            if a_city_distance_df[server_city][eval_city_id] < min_dist:
                min_city = eval_city
                min_dist = a_city_distance_df[server_city][eval_city_id]

        city = min_city
        task_level = city[1]
        task_city = int(city[0][5:])
        revenue = revenue_for_lv[task_level - 1]
        cost = a_city_distance_df[server_city][task_city]

        final_revenue += revenue - cost
        final_allocation_for_a_day.append(
            [server_id, server_city, task_city, task_level, server_level]
        )
        # print(f"Server: {server}, Level: {server_level}, City: {city}, City Level: {city[1]}")
        # ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
        allocation_data.append({"server": server, "city": city})
        cities_remain.remove(city[0])

    allocation_df = pd.DataFrame(allocation_data, columns=["server", "city"])

    return final_revenue, final_allocation_for_a_day


def single_stage_opt_allocate_servers_2_citys(
    remain_servers_df, a_task_df, a_city_distance_df
):
    # 根据 lv 来做分组, join 分组后的结果应该
    join_idx_2_task_lv_server_lv_num = generate_idx_2_joins(
        a_task_df, remain_servers_df
    )
    # print(f"{join_idx_2_task_lv_server_lv_num=}")
    cnt = 0
    # print("（业务员编号id，业务员城市，分配去的城市编号，城市等级，业务员等级）")

    join_idx_2_revenue_and_combination_list_dict = {}
    revenue_sum = 0
    # 对于每一个集合，进行集合内分配
    for (
        join_idx,
        decisions_for_each_joins,
    ) in join_idx_2_task_lv_server_lv_num.items():
        cnt += 1
        # 根据集合分组后的，集合内level的数量分配
        # print("----join_idx, decisions_for_each_join----", join_idx, decisions_for_each_joins)
        # decisions_for_each_join = [[(1, 1, 4)]]
        revenue_and_combination_list_for_a_join = {}

        for idx, a_decision in enumerate(decisions_for_each_joins):
            # 集合分组后，level的分配方式可能很多, 一个decision里面是一个分配方案[(1, 1, 4)]
            # 可能是这样的 [(1, 1, 4)] / [(1, 1, 2), (2, 2, 2)]
            # 根据数量分配，进行实际的城市分配方案 list下的一个list是
            # print(f"{idx=} {a_decision=}")
            revenue_and_combination_for_decision = allocate_servers_2_cities(
                a_decision, a_task_df, remain_servers_df, a_city_distance_df
            )
            # revenue_and_combination_for_decision = {'revenue':final_revenue, 'combination':min_cost_city_id_of_server_and_task_combination}
            revenue_and_combination_list_for_a_join.update(
                {idx: revenue_and_combination_for_decision}
            )

        join_idx_2_revenue_and_combination_list_dict.update(
            {join_idx: revenue_and_combination_list_for_a_join}
        )

    # 有了不同的，开始合并决策，并reduce 并获取V历史记录的对应的收益。
    final_revenue_n_combination_list = get_all_allocations_for_decisions(
        join_idx_2_revenue_and_combination_list_dict
    )

    # 获取最好的分配
    final_revenue, final_allocation_for_a_day = get_allocation_for_a_day(
        final_revenue_n_combination_list,
        remain_servers_df,
        a_task_df,
        # arriving_rate_df, weekday, proveng_dict, city_num_2_name,reduce_V # TODO
    )
    return final_revenue, final_allocation_for_a_day


def save_a_state_revenue(
    a_servers_df,
    new_servers_df,
    new_task_df,
    a_task_df,
    allocate_task_df,
    final_revenue,
    reduce_V,
    reduce_V_iter,
    weekday,
    proveng_dict,
    city_num_2_name,
    a_iter,
):
    """
    对状态将城市缩减为省份之后做个保存

    输入:
    - a_servers_df: DataFrame,原始业务员数据框
    - new_servers_df: DataFrame,更新后的业务员数据框(业务员分配后的位置)
    - new_task_df: DataFrame,更新后的任务数据框(分配后新增任务)
    - a_task_df: DataFrame,原始任务数据框
    - allocate_task_df: DataFrame,分配的任务数据框
    - final_revenue: float,最终的总收益
    - reduce_V: dict,缩减后的状态值函数
    - reduce_V_iter: dict,缩减后的状态值函数(按迭代次数)
    - weekday: int,星期几(1-7)
    - proveng_dict: dict,城市编号到省份的映射
    - city_num_2_name: dict,城市编号到名称的映射
    - a_iter: int,当前的迭代次数

    输出:
    - reduce_V: dict,更新后的缩减状态值函数
    - reduce_V_iter: dict,更新后的缩减状态值函数(按迭代次数)
    """
    reduced_server = reduce_server_df(
        a_servers_df=a_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_server_allocated = reduce_server_df(
        a_servers_df=new_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_task = reduce_task_df(
        a_task_df=a_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_allocate_task = reduce_task_df(
        a_task_df=allocate_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_new_task = reduce_task_df(
        a_task_df=new_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    # S2 new server task 组成
    # key = (reduced_server, reduced_task.values.tolist()) # 修改
    
    key = (str(reduced_server_allocated), str(reduced_new_task.values.tolist()))
    # key = (
    #     str(reduced_server),
    #     str(reduced_server_allocated),
    #     str(reduced_task.values.tolist()),
    #     str(reduced_allocate_task.values.tolist()),
    # )
    reduce_V_iter[a_iter][weekday - 1].update({key: final_revenue})
    reduce_V[weekday - 1].update({key: final_revenue})

    return reduce_V, reduce_V_iter


def get_a_state_revenue(
    a_servers_df,
    new_servers_df,
    new_task_df,
    a_task_df,
    allocate_task_df,
    reduce_V,
    reduce_V_iter,
    reduce_V_actual,
    weekday,
    proveng_dict,
    city_num_2_name,
    a_iter,
):
    # 获取状态
    reduced_server = reduce_server_df(
        a_servers_df=a_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_server_allocated = reduce_server_df(
        a_servers_df=new_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_task = reduce_task_df(
        a_task_df=a_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_allocate_task = reduce_task_df(
        a_task_df=allocate_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )

    reduced_new_task = reduce_task_df(
        a_task_df=new_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    # S2 new server task 组成
    key = (str(reduced_server_allocated), str(reduced_new_task.values.tolist()))
    # key = (
    #         str(reduced_server),
    #         str(reduced_server_allocated),
    #         str(reduced_task.values.tolist()),
    #         str(reduced_allocate_task.values.tolist()),
    #     )
    if key in reduce_V_actual[weekday - 1].keys():
        revenue = reduce_V_actual[weekday - 1][key]
    else:
        revenue = 0
    # print(f"get_a_state_revenue {weekday=} {revenue=} {key=}")
    return revenue


def allocate_reduce_df(
    final_allocation_for_a_day, a_task_df, arriving_rate_df, a_servers_df
):
    # 本函数用于对于确定的分配,获取状态用于后续的计算
    ''' 输入:
    - final_allocation_for_a_day: list[tuple],一天的最优分配,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)
    - a_task_df: DataFrame,任务数据框
    - arriving_rate_df: DataFrame,到达率数据框
    - a_servers_df: DataFrame,业务员数据框

    输出:
    - new_task_df: DataFrame,更新后的任务数据框(分配后新增任务)
    - new_servers_df: DataFrame,更新后的业务员数据框(业务员分配后的位置)
    - allocate_task_df: DataFrame,分配的任务数据框
    '''
    # 传入分配情况/状态举证/到达矩阵
    # 返回新增任务后的状态矩阵/ 分配的矩阵
    # todo 这里应该对任务新增做个保存，为了后续的其他算法的计算
    # 保存矩阵和迭代次数以及周期有关系
    # 算法计算其可以设计为class 类 有全局变量，保存当前的迭代次数，计算周期，状态情况。
    # {}

    allocate_task_df = allcocate_comb_2_allocate_task_df(
        final_allocation_for_a_day, a_task_df
    )
    new_task_df = a_task_df - allocate_task_df  # 分配

    current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
    # 生成新任务
    arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
    new_tasks = generate_tasks(arriving_rate_matrix)

    # 更新状态矩阵
    new_task_df = update_state(current_task_df, new_tasks)

    # 更新业务员矩阵
    new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)
    return new_task_df, new_servers_df, allocate_task_df


def save_reduct_v(
    a_task_df,
    a_servers_df,
    final_allocation_for_a_day,
    arriving_rate_df,
    weekday,
    proveng_dict,
    city_num_2_name,
    final_revenue,
    reduce_V,
    reduce_V_iter,
    a_iter,
):
    """
    本函数用于保存收益矩阵

    输入:
    - a_task_df: DataFrame,任务数据框
    - a_servers_df: DataFrame,业务员数据框
    - final_allocation_for_a_day: list[tuple],一天的最优分配,每个元素为一个元组(业务员id,业务员城市,分配去的城市编号,城市等级,业务员等级)
    - arriving_rate_df: DataFrame,到达率数据框
    - weekday: int,星期几(0-6)
    - proveng_dict: dict,省份字典
    - city_num_2_name: dict,城市编号到名称的映射
    - final_revenue: float,最终的总收益
    - reduce_V: dict,缩减后的状态值函数
    - reduce_V_iter: int,缩减的迭代次数
    - a_iter: int,当前的迭代次数

    输出:
    - reduce_V: dict,更新后的缩减状态值函数
    - reduce_V_iter: int,更新后的缩减迭代次数
    - new_task_df: DataFrame,更新后的任务数据框
    - new_servers_df: DataFrame,更新后的业务员数据框
    - allocate_task_df: DataFrame,分配的任务数据框
    """
    new_task_df, new_servers_df, allocate_task_df = allocate_reduce_df(
        final_allocation_for_a_day, a_task_df, arriving_rate_df, a_servers_df
    )

    reduce_V, reduce_V_iter = save_a_state_revenue(
        a_servers_df=a_servers_df,
        new_servers_df=new_servers_df,
        new_task_df=new_task_df,
        a_task_df=a_task_df,
        allocate_task_df=allocate_task_df,
        final_revenue=final_revenue,
        reduce_V=reduce_V,
        reduce_V_iter=reduce_V_iter,
        weekday=weekday,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
        a_iter=a_iter,
    )

    return reduce_V, reduce_V_iter, new_task_df, new_servers_df, allocate_task_df


def cul_a_cycle(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
    reduce_V_iter,
    reduce_V_actual,
    a_iter,
):
    '''
    本函数用于对一个周期内求解状态和周期的收益

    输入:
    - T: int, 周期的天数
    - a_servers_df: DataFrame, 业务员状态 (40*3)
    - a_task_df: DataFrame, 任务状态 (26*5)
    - arriving_rate_df: DataFrame, 任务到达率,来自excel (26*5) 
    - a_city_distance_df: DataFrame, 城市距离的数据 (26*26) (city_num*city_num)
    - proveng_dict: dict, 省份字典
    - city_num_2_name: dict, 城市编号到名称的映射
    - reduce_V: dict, 缩减后的状态值函数
    - reduce_V_iter: list[list], 缩减的迭代次数
    - reduce_V_actual: list[], 实际的缩减状态值函数
    - a_iter: int, 当前的迭代次数

    输出:
    - reduce_V: dict, 更新后的缩减状态值函数
    - reduce_V_iter: int, 更新后的缩减迭代次数
    '''
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for time in range(1, T + 1):
        weekday = time % 7
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")

        # remain_servers_df
        final_revenue, final_allocation_for_a_day = allocate_servers_2_citys_MDP(
            remain_servers_df=remain_servers_df,
            a_task_df=a_task_df,
            a_city_distance_df=a_city_distance_df,
            arriving_rate_df=arriving_rate_df,
            weekday=weekday,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
            reduce_V=reduce_V,
            reduce_V_iter=reduce_V_iter,
            reduce_V_actual=reduce_V_actual,
            a_iter=a_iter,
        )

        reduce_V, reduce_V_iter, new_task_df, new_servers_df, allocate_task_df = (
            save_reduct_v(
                a_task_df=a_task_df,
                a_servers_df=a_servers_df,
                final_allocation_for_a_day=final_allocation_for_a_day,
                arriving_rate_df=arriving_rate_df,
                weekday=weekday,
                proveng_dict=proveng_dict,
                city_num_2_name=city_num_2_name,
                final_revenue=final_revenue,
                reduce_V=reduce_V,
                reduce_V_iter=reduce_V_iter,
                a_iter=a_iter,
            )
        )

        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V, reduce_V_iter


def cul_a_cycle_rnd(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
):
    # 本函数用于计算随机分配的策略
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T + 1):
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")
        final_revenue, final_allocation_for_a_day = rnd_allocate_servers_2_citys(
            remain_servers_df, a_task_df, a_city_distance_df
        )

        allocate_task_df = allcocate_comb_2_allocate_task_df(
            final_allocation_for_a_day, a_task_df
        )

        new_task_df = a_task_df - allocate_task_df  # 分配
        arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
        current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
        # 生成新任务
        new_tasks = generate_tasks(arriving_rate_matrix)

        # 更新状态矩阵
        new_task_df = update_state(current_task_df, new_tasks)

        # 更新业务员矩阵
        new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)

        reduced_server = reduce_server_df(
            a_servers_df=a_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_server_allocated = reduce_server_df(
            a_servers_df=new_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_task = reduce_task_df(
            a_task_df=a_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_allocate_task = reduce_task_df(
            a_task_df=allocate_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )

        reduce_V[weekday - 1].update(
            {
                (
                    str(reduced_server),
                    str(reduced_server_allocated),
                    str(reduced_task.values.tolist()),
                    str(reduced_allocate_task.values.tolist()),
                ): final_revenue
            }
        )
        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V


def cul_a_cycle_nearest(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
):
    # 本函数用于决策每次获取最近距离的策略
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T + 1):
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")
        (
            final_revenue,
            final_allocation_for_a_day,
        ) = nearest_distance_allocate_servers_2_citys(
            remain_servers_df, a_task_df, a_city_distance_df
        )

        allocate_task_df = allcocate_comb_2_allocate_task_df(
            final_allocation_for_a_day, a_task_df
        )

        new_task_df = a_task_df - allocate_task_df  # 分配
        arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
        current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
        # 生成新任务
        new_tasks = generate_tasks(arriving_rate_matrix)

        # 更新状态矩阵
        new_task_df = update_state(current_task_df, new_tasks)

        # 更新业务员矩阵
        new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)

        reduced_server = reduce_server_df(
            a_servers_df=a_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_server_allocated = reduce_server_df(
            a_servers_df=new_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_task = reduce_task_df(
            a_task_df=a_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_allocate_task = reduce_task_df(
            a_task_df=allocate_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )

        reduce_V[weekday - 1].update(
            {
                (
                    str(reduced_server),
                    str(reduced_server_allocated),
                    str(reduced_task.values.tolist()),
                    str(reduced_allocate_task.values.tolist()),
                ): final_revenue
            }
        )
        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V


def cul_a_cycle_single_stage(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
):
    # 本函数用于决策单阶段的策略
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T + 1):
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")
        (
            final_revenue,
            final_allocation_for_a_day,
        ) = single_stage_opt_allocate_servers_2_citys(
            remain_servers_df, a_task_df, a_city_distance_df
        )

        allocate_task_df = allcocate_comb_2_allocate_task_df(
            final_allocation_for_a_day, a_task_df
        )

        new_task_df = a_task_df - allocate_task_df  # 分配
        arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
        current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
        # 生成新任务
        new_tasks = generate_tasks(arriving_rate_matrix)

        # 更新状态矩阵
        new_task_df = update_state(current_task_df, new_tasks)

        # 更新业务员矩阵
        new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)

        reduced_server = reduce_server_df(
            a_servers_df=a_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_server_allocated = reduce_server_df(
            a_servers_df=new_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_task = reduce_task_df(
            a_task_df=a_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_allocate_task = reduce_task_df(
            a_task_df=allocate_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )

        reduce_V[weekday - 1].update(
            {
                (
                    str(reduced_server),
                    str(reduced_server_allocated),
                    str(reduced_task.values.tolist()),
                    str(reduced_allocate_task.values.tolist()),
                ): final_revenue
            }
        )
        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V


# 先判断是否可以分配，划分分配的区间，区间内任意分配，
def generate_joins(
    task_lv_count: pd.core.series.Series, server_lv_count: pd.core.series.Series
) -> list[list]:
    # 输入
    # task_lv_count pandas.core.series.Series(level_num,) 任务不同等级
    # server_lv_count pandas.core.series.Series(level_num,)
    # server number tag task number
    # lv 1 6 > 5
    # lv 2 2 > 1
    # lv 3 3 ≤ 6
    # lv 4 2 > 1
    # lv 5 4 ≤ 17
    # lv 6 5 > 4
    # lv 7 7 > 3
    # 需要根据不同等级任务和员工的等级做个计数判断，当连续的每个等级各自同时加起来大于现有的任务等级，表示这一组level之前可以任意分配，可以分配出不同的决策，
    # 比如上面的表里level1和level 2 server number > task number
    # lv 1可以分配给level2，也可以level2 都分配给level2，这是不同的决策空间，
    # 但是level 1 task也必须被做完。当level3 也考虑进去的时候，123相加就server数量就比task少了
    # 这时候level4 就不能和123 一起考虑，但是123 之间可以一起决策，level3 全部分配到task
    # 同时level 2 个也得分配到level3 或者level2
    # 输出 return joins = [[1,2,3],[4],[5]]
    # 输入校验：
    if task_lv_count.shape[0] != server_lv_count.shape[0]:
        raise Exception("输入pandas.core.series.Series形状不一致")

    joins = []  # 存储决策空间的列表
    current_join = []  # 当前的决策空间划分
    task_lv_num = task_lv_count.shape[0]  # 任务等级数量
    task_join_count = 0
    server_join_count = 0

    for idx, (task_count, server_count) in enumerate(
        zip(task_lv_count, server_lv_count)
    ):
        task_join_count += task_count
        server_join_count += server_count
        lv = idx + 1
        if server_join_count <= task_join_count:
            # 当任务数量大于员工数量时，将当前决策空间划分添加到列表中
            if current_join:
                current_join.append(lv)
                joins.append(current_join)
            else:
                joins.append([lv])

            current_join = []  # 重置当前决策空间划分
            task_join_count = 0
            server_join_count = 0
        else:
            # 当任务数量小于等于员工数量时，将当前等级添加到当前决策空间划分中
            current_join.append(lv)
    if current_join:
        joins.append(current_join)  # 将最后一个决策空间划分添加到列表中
    return joins


def form_all_decision_list_by_tuple(
    decision_task_done_by_server_level: tuple, tasks, servers, levels
) -> list[tuple]:
    """
    根据任务完成情况的元组,生成决策列表

    输入:
    - decision_task_done_by_server_level: tuple,任务完成情况的元组,表示每个任务由哪个等级的业务员完成
    - tasks: list,每个等级的任务数量列表
    - servers: list,每个等级的业务员数量列表
    - levels: list,等级列表

    输出:
    - decision_list_tuple: list[tuple],决策列表,每个决策为一个元组(任务等级,业务员等级,数量)

    注意:
    - 决策列表按照任务等级的顺序生成
    - 对于每个任务等级,统计由不同业务员等级完成的任务数量,生成相应的决策元组
    """
    
    # （任务等级，业务员等级，数量）
    last_task_idx = 0
    decision_list_tuple = []
    for idx, task_num in enumerate(tasks):
        task_level = levels[idx]
        task_num = tasks[idx]
        level_count_for_a_level_task = Counter(
            decision_task_done_by_server_level[last_task_idx : last_task_idx + task_num]
        )
        decision_list_tuple.extend(
            [
                (task_level, num, count)
                for num, count in level_count_for_a_level_task.items()
            ]
        )
        last_task_idx += task_num
    return decision_list_tuple


def decisions_2_decisions_list_tuple(
    decisions, tasks, servers, levels
) -> list[list[tuple]]:
    """
    将决策列表转换为元组形式的决策列表

    输入:
    - decisions: list[list],决策列表,每个决策为一个列表
    - tasks: list,每个等级的任务数量列表
    - servers: list,每个等级的业务员数量列表
    - levels: list,等级列表

    输出:
    - decisions_list_tuple: list[list[tuple]],元组形式的决策列表,每个决策为一个元组列表

    注意:
    - 决策列表中的每个决策都需要转换为元组形式
    - 转换后的决策列表保持与原决策列表相同的顺序
    """
    decisions_list_tuple = []

    for decision in decisions:
        decisions_list_tuple.append(
            form_all_decision_list_by_tuple(decision, tasks, servers, levels)
        )

    return decisions_list_tuple


def remove_duplicates(list_of_lists_of_tuples):
    """去除 list 中重复的 list of tuples。

    Args:
      list_of_lists_of_tuples: 一个包含 list of tuples 的 list。

    Returns:
      一个不包含重复 list of tuples 的 list。
    """

    # 将 list of tuples 转换为 tuple of tuples。
    list_of_tuples_of_tuples = [
        tuple(list_of_tuples) for list_of_tuples in list_of_lists_of_tuples
    ]

    # 使用 set 来去除重复的 tuple of tuples。
    set_of_tuples_of_tuples = set(list_of_tuples_of_tuples)

    # 将 tuple of tuples 转换为 list of tuples。
    list_of_lists_of_tuples = [
        list(tuple_of_tuples) for tuple_of_tuples in set_of_tuples_of_tuples
    ]

    # 返回不包含重复 list of tuples 的 list。
    return list_of_lists_of_tuples


def generate_idx_2_joins(state_df, remain_servers_df):
    """
    本函数用于将集合分组

    输入:
    - state_df: DataFrame,状态数据框,包含城市对应等级任务矩阵
    - remain_servers_df: DataFrame,剩余的业务员数据

    输出:
    - join_idx_2_task_lv_server_lv_num: dict,分组结果的字典
      - 键: int,分组的索引
      - 值: list,每个元素为一个元组(任务等级,业务员等级,数量)
    """
    # 城市对应等级任务矩阵
    city_lv_matrix_df = state_df.iloc[:, :]
    # 任务计数，用于划分集合
    task_lv_count = city_lv_matrix_df.sum()
    # 业务员等级计数，用于划分集合
    server_lv_count = remain_servers_df["lv"].value_counts().sort_index()
    all_lv_values = range(1, 6)
    server_lv_count = server_lv_count.reindex(all_lv_values, fill_value=0)
    level_num = city_lv_matrix_df.shape[0]  # 任务等级数量
    city_num = city_lv_matrix_df.shape[0]  # 城市数量
    # 根据levels， servers，tasks区分组别
    servers = list(server_lv_count)
    tasks = task_lv_count.to_list()
    levels = list(range(1, len(tasks) + 1))
    # print(
    #     f"{levels=}",
    # )
    # print(f"{tasks=} {sum(tasks)=}")
    # print(f"{servers=} {sum(servers)=}")
    joins = generate_joins(task_lv_count, server_lv_count)
    # print(f"根据不同level的task数量和server数量, 组内数量分配结果如下 {joins=}")
    # 组别内allocate
    join_idx_2_task_lv_server_lv_num = {}
    for join_idx, join in enumerate(joins):
        level_indices = [
            levels.index(lv) for lv in join
        ]  # 获取join中每个level对应的索引
        a_tasks = [
            tasks[index] for index in level_indices
        ]  # 获取与join中每个level相对应的任务数量
        a_servers = [
            servers[index] for index in level_indices
        ]  # 获取与join中每个level相对应的业务员数量
        a_lvs = [levels[index] for index in level_indices]
        decisions = allocate(tasks=a_tasks, servers=a_servers, levels=a_lvs)
        decisions_list_tuple = decisions_2_decisions_list_tuple(
            decisions=decisions, tasks=a_tasks, servers=a_servers, levels=a_lvs
        )

        join_idx_2_task_lv_server_lv_num.update(
            {join_idx: remove_duplicates(decisions_list_tuple)}
        )
    # 对每一个decision，对于不同地方和不同的人，进行排列组合分配
    # 获取每个城市情况
    # 每个server情况

    return join_idx_2_task_lv_server_lv_num


# 2024年1月21日 21点42分
def change_df_city_name_2_idx(cities: pd.DataFrame) -> (pd.DataFrame, dict):
    """改名字为idx，同时返回新df和idx2name， 例如修改zhejiang 转为1 这种数字，用于后续爬代码"""
    city_names = cities.columns
    city_num = len(city_names)
    city_name_nums = list(range(1, city_num + 1))
    city_num_2_name = {key: value for key, value in zip(city_name_nums, city_names)}
    new_cities = cities.copy()
    new_cities.columns = new_cities.index = city_name_nums
    return new_cities, city_num_2_name


# 2024年1月21日 20点42分
def get_proveng_city_dist_mat_df(
    path=r"C:\Users\dylan\Desktop\code\paper\data\中国各城市空间权重矩阵(1).xlsx",
):
    proveng_city_dist_mat_df = pd.read_excel(
        path,
        sheet_name="地理距离矩阵",
    )
    return proveng_city_dist_mat_df


def get_city_2_proveng_dict(proveng_city_dist_mat_df=None):
    '''获取城市转省份字典'''
    if proveng_city_dist_mat_df is None:
        proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    city_to_proveng_dict = (
        proveng_city_dist_mat_df[["cityeng", "proveng"]]
        .drop_duplicates()
        .set_index("cityeng")["proveng"]
        .to_dict()
    )
    return city_to_proveng_dict


def get_city_series_to_city_dict(proveng_city_dist_mat_df=None):
    if proveng_city_dist_mat_df is None:
        proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()

    city_series_to_city_dict = (
        proveng_city_dist_mat_df[["cityeng", "cityseries"]]
        .drop_duplicates()
        .set_index("cityseries")["cityeng"]
        .to_dict()
    )
    return city_series_to_city_dict


def get_city_to_city_series_dict(proveng_city_dist_mat_df=None):
    if proveng_city_dist_mat_df is None:
        proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    city_to_city_series_dict = (
        proveng_city_dist_mat_df[["cityeng", "cityseries"]]
        .drop_duplicates()
        .set_index("cityeng")["cityseries"]
        .to_dict()
    )
    return city_to_city_series_dict


def get_dist_mat_from_xls_df(
    proveng_city_dist_mat_df, distance_mat_start_idx, cityseries_idx
):
    distance_mat = proveng_city_dist_mat_df.iloc[:, distance_mat_start_idx:]
    distance_mat.index = proveng_city_dist_mat_df.iloc[:, cityseries_idx].values
    distance_mat.columns = proveng_city_dist_mat_df.iloc[:, cityseries_idx].values

    return distance_mat


def get_select_city_df(
    city_num, proveng_city_dist_mat_df, proveng_idx, cityseries_idx, distance_mat
):
    # 挑选出安徽、江苏、浙江和上海的省份及对应的矩阵数据
    select_proveng = ["Anhui", "Jiangsu", "Zhejiang", "Shanghai"]  # TODO 可能要换省份
    provengs = proveng_city_dist_mat_df.iloc[
        :, proveng_idx
    ].values  # 表格第一列为城市名称
    select_proveng_idxs = [
        i for i, prov in enumerate(provengs) if prov in select_proveng
    ]  # xls里选取指定的省份的idx，是字母
    select_cityeng_idxs = proveng_city_dist_mat_df.values[
        select_proveng_idxs, cityseries_idx
    ]  # xls指定省份对应的城市id
    select_idx = pd.Index(select_cityeng_idxs)  # 创建对应索引
    # 使用 loc 方法选择相应的行和列
    select_proveng_city_df = distance_mat.loc[select_idx, select_idx]

    # 从城市列表中随机选择city_num-1个城市（不包括上海）
    rnd_cities = np.random.choice(
        select_cityeng_idxs[1:], city_num - 1, replace=False
    )  # 第一列为上海，所以从1开始
    # 将上海添加到随机选择的城市列表中 # City158 = shanghai
    select_cities = ["City158"] + rnd_cities.tolist()
    select_city_idx = pd.Index(select_cities)
    # 使用 loc 方法选择相应的行和列
    select_city_df = select_proveng_city_df.loc[select_city_idx, select_city_idx]

    return select_city_df


def generate_city(city_num: int = 26) -> (pd.DataFrame, dict):
    """本函数功能，生成指定数量城市的城市距离矩阵，必然包含上海。返回的dataframe包含城市坐标和省份坐标/和一个城市转省份的dict"""

    # 地理距离矩阵xls
    proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    # 从第四列开始，是距离矩阵内容
    proveng_idx = 0
    cityeng = 1
    cityseries_idx = 2
    distance_mat_start_idx = 3

    #
    city_series_to_city_dict = get_city_series_to_city_dict(
        proveng_city_dist_mat_df=proveng_city_dist_mat_df
    )
    city_to_proveng = get_city_2_proveng_dict(
        proveng_city_dist_mat_df=proveng_city_dist_mat_df
    )

    # 地理距离矩阵xls 第三列是城市id, city_*, 设定纵坐标
    distance_mat = get_dist_mat_from_xls_df(
        proveng_city_dist_mat_df, distance_mat_start_idx, cityseries_idx
    )
    select_city_df = get_select_city_df(
        city_num, proveng_city_dist_mat_df, proveng_idx, cityseries_idx, distance_mat
    )

    city_series_columns = select_city_df.columns
    city_columns = [
        (
            city_to_proveng[city_series_to_city_dict[item]],
            city_series_to_city_dict[item],
        )
        for item in city_series_columns
    ]  # 最为新矩阵的idx和col

    column_names = ["proveng", "cityeng"]  # 给idx的idx
    city_df = pd.DataFrame(
        select_city_df.values,
        index=pd.MultiIndex.from_tuples(city_columns, names=column_names),
        columns=pd.MultiIndex.from_tuples(city_columns, names=column_names),
    )
    city_df.to_excel(rf"C:\Users\dylan\Desktop\code\paper\city_{city_num}.xlsx")
    print(rf"D:\Users\sjc\algorithm\paper\city_{city_num}.xlsx")
    return city_df


if __name__ == "__main__":
    city = generate_city(city_num=10)
    city_to_proveng = get_city_2_proveng_dict()
