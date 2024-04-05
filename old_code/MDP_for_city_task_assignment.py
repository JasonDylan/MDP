import pandas as pd
import numpy as np
from city_cls import (
    generate_city,
    get_city_2_proveng_dict,
    change_df_city_name_2_idx,
    cul_a_cycle,
)
import random

global revenue_for_lv, a_city_distance_df, province_dict, city_num_2_name

if __name__ == "__main__":
    # 全局的一些设定
    ## 收益率
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    # 全局设定种子，保证，每次随机结果一致
    np.random.seed(42)
    # 生成城市规模/一个省 生成 26个城市 todo 修改此处可以控制生成的城市数量
    city_num = 26
    city_distance_df = generate_city(city_num=city_num)
    city_to_proveng = get_city_2_proveng_dict()

    proveng = city_distance_df.index.get_level_values(
        "proveng"
    ).unique()  # 获得所有province的name
    proveng_dict = {proveng[i]: i + 1 for i in range(len(proveng))} # 省份转id
    city_names = city_distance_df.columns
    a_city_distance_df, city_num_2_name = change_df_city_name_2_idx(
        cities=city_distance_df
    )
    global arriving_rate_df

    # 读取excel的城市/员工状态/arriving_rate
    arriving_rate_df = pd.read_excel(
        "./data/数据.xlsx", sheet_name="arriving rate", index_col=0
    )
    initial_state_df = pd.read_excel(
        "./data/数据.xlsx", sheet_name="initial state", index_col=0
    )
    servers_df = pd.read_excel("./data/数据.xlsx", sheet_name="servers", index_col=0)
    # 员工
    servers_df.columns = ["current_city", "lv", "day off"]

    # 对于每个join，产生其分配方案，生成所有分配方案，分配方案是指
    # 当前日子，对于每个城市的状态，业务员的状态，生成一组对业务员的分配
    # 可能为（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级

    arriving_rate_df = arriving_rate_df[:city_num]
    a_state_df = initial_state_df.copy()[:city_num]
    a_servers_df = servers_df.copy()[servers_df["current_city"] <= city_num]
    saved_params = {}
    T = 7
    # 输入一个缩减为三个省的state 和 一个action 对应结果为 对应的收益
    # 这个循环，进行一次指定周期内的迭代。
    random.seed(42)
    iters = 10000
    # 修改保存的key value_S1 的key 值为状态，而不是当前状态+决策状态/应该改为当前状态+新到达任务
    # 需要修改的地方是？保存的地方 save 和get
    # 同时需要一个保存每个新产生的城市的矩阵
    reduce_V = [{} for _ in range(T)]
    reduce_V_actual = [{} for _ in range(T)]
    reduce_V_iter = [[{} for _ in range(T)] for a_iter in range(iters)]
    reduce_V_actual_iter = [[{} for _ in range(T)] for a_iter in range(iters)]

    arriving_df_iter = [[0 for _ in range(T)] for a_iter in range(iters)]
    for it in range(iters):
        reduce_V, reduce_V_iter = cul_a_cycle(
            T=T,
            a_servers_df=a_servers_df,
            a_task_df=a_state_df,
            arriving_rate_df=arriving_rate_df,
            a_city_distance_df=a_city_distance_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
            reduce_V=reduce_V,
            reduce_V_iter=reduce_V_iter,
            reduce_V_actual=reduce_V_actual,
            a_iter=it,
        )
        print(
            f"---------------------------------------------------------------iter:{it=}--------------------------------------------------------------- "
        )

        ## 计算V

        all_V_from_a_round = reduce_V_iter[it]
        new_V_actual_S1 = 0
        sum_diff = 0
        for weekday in range(T - 1, 0, -1):
            S1 = all_V_from_a_round[weekday - 1]
            S2 = all_V_from_a_round[weekday]
            # V(S1) = R(S1)+V(S2')
            if len(S2.keys()) > 1:
                print(S2)
            key_S1 = list(S2.keys())[0]
            value_S1 = list(S2.values())[0]
            key_S2 = list(S2.keys())[0]
            # print(f"{value_S1}")
            old_V_actual_S2 = reduce_V_actual[weekday].get(key_S2, 0)
            new_V_actual_S1 = value_S1 + old_V_actual_S2
            old_V_actual_S1 = reduce_V_actual[weekday].get(key_S1, 0)
            sum_diff += abs(old_V_actual_S1 - new_V_actual_S1)
            print(f"{old_V_actual_S2=}, {new_V_actual_S1=}")
            reduce_V_actual[weekday].update({key_S1: new_V_actual_S1})
            reduce_V_actual_iter[it][weekday].update({key_S1: new_V_actual_S1})
            # reduce_V_actual[it][] =
        ## 上一次迭代和本地迭代结果差距计算, 问题：结果差如何计算，整个V矩阵计算么，还是对于单独一次的上下次计算？
        print(f"{sum_diff=}")
        if sum_diff < 1:
            break
    print(
        f"---------------------------------------------------------------DONE--------------------------------------------------------------- "
    )
