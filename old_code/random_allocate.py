# %%
import pandas as pd
import numpy as np
from city_cls import (
    generate_city,
    get_city_2_proveng_dict,
    change_df_city_name_2_idx,
    # cul_a_cycle,
    cul_a_cycle_rnd,
    cul_a_cycle_nearest,
    cul_a_cycle_single_stage
)
import os
import random
# 获取当前工作目录
current_dir = os.getcwd()
print("当前工作目录:", current_dir)

# 设置新的工作目录
new_dir = r"D:\Users\sjc\algorithm\paper"
os.chdir(new_dir)

# 再次获取当前工作目录
updated_dir = os.getcwd()
print("更新后的工作目录:", updated_dir)

# %%
# 全局的一些设定
## 收益率
revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
# 全局设定种子，保证，每次随机结果一致
np.random.seed(42)
# 生成城市规模/一个省
city_num = 13
# 生成 26个城市
city_distance_df = generate_city(city_num=city_num)
city_to_proveng = get_city_2_proveng_dict()

proveng = city_distance_df.index.get_level_values(
    "proveng"
).unique()  # 获得所有province的name
proveng_dict = {proveng[i]: i + 1 for i in range(len(proveng))}
city_names = city_distance_df.columns
a_city_distance_df, city_num_2_name = change_df_city_name_2_idx(
    cities=city_distance_df
)

arriving_rate_df = pd.read_excel(
    "./data/数据.xlsx", sheet_name="arriving rate", index_col=0
)
travel_fee_df = pd.read_excel(
    "./data/数据.xlsx", sheet_name="travel fee", index_col=0
)
initial_state_df = pd.read_excel(
    "./data/数据.xlsx", sheet_name="initial state", index_col=0
)
servers_df = pd.read_excel("./data/数据.xlsx", sheet_name="servers", index_col=0)
# 员工
servers_df.columns = ["current_city", "lv", "day off"]

# 对于每个join，产生其分配方案，生成所有分配方案，分配方案是指
# 当前日子，对于每个城市的状态，业务员的状态，生成一组对业务员的分配，
# 可能为（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）

arriving_rate_df = arriving_rate_df[:city_num]
a_state_df = initial_state_df.copy()[:city_num]
a_servers_df = servers_df.copy()[servers_df["current_city"] <= city_num]
saved_params = {}
T = 7

def get_avg_weekend_revenue(reduce_V, T):
    weekday_revenues = []
    for weekday in range(1, T + 1):
        revenues = [revenue for allocate, revenue in reduce_V[weekday-1].items()]
        mean_revenue = sum(revenues) / len(revenues)
        weekday_revenues.append(mean_revenue)
    return weekday_revenues

# %% [markdown]
# # benchmark 
iter_num = 100
# %%
# 输入一个缩减为三个省的state 和 一个action 对应结果为 对应的收益
# 这个循环，进行一次指定周期内的迭代。
reduce_V_1 = [{} for _ in range(T)]
random.seed(42)
iters = iter_num
for i in range(iters):
    print(f"{i=}")
    reduce_V_1 = cul_a_cycle_rnd(
    T,
    a_servers_df,
    a_state_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V_1,
    )
    # print(f"{i=} {reduce_V_1=}")

weekday_revenues1 = get_avg_weekend_revenue(reduce_V_1, T)


# %%
# 输入一个缩减为三个省的state 和 一个action 对应结果为 对应的收益
# 这个循环，进行一次指定周期内的迭代。
reduce_V_2 = [{} for _ in range(T)]
random.seed(42)
iters = iter_num
for i in range(iters):
    print(f"{i=}")
    reduce_V_2 = cul_a_cycle_nearest(
        T,
        a_servers_df,
        a_state_df,
        arriving_rate_df,
        a_city_distance_df,
        proveng_dict,
        city_num_2_name,
        reduce_V_2,
    )
    # print(f"{i=} {reduce_V_1=}")
weekday_revenues2 = get_avg_weekend_revenue(reduce_V_2, T)
# %%
# 输入一个缩减为三个省的state 和 一个action 对应结果为 对应的收益
# 这个循环，进行一次指定周期内的迭代。
reduce_V_3 = [{} for _ in range(T)]
random.seed(42)
iters = iter_num
for i in range(iters):
    print(f"{i=}")
    reduce_V_3 = cul_a_cycle_single_stage(
        T,
        a_servers_df,
        a_state_df,
        arriving_rate_df,
        a_city_distance_df,
        proveng_dict,
        city_num_2_name,
        reduce_V_3,
    )
    # print(f"{i=} {reduce_V_1=}")

# %%

weekday_revenues3 = get_avg_weekend_revenue(reduce_V_3, T)


