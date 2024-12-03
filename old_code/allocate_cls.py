import itertools
from collections import Counter

# TODO 改成class


def is_comb_valid(comb, levels, servers):
    """
    判断给定的分配组合是否有效

    输入:
    - comb: tuple,分配组合,表示每个任务分配到的等级
    - levels: list,等级列表
    - servers: list,每个等级的业务员数量列表

    输出:
    - is_valid: bool,表示分配组合是否有效

    注意:
    - 分配组合需要满足以下条件:
      每个等级分配的任务数量不能超过该等级的业务员数量
    """
    count = Counter(comb)
    for level in count.keys():
        if  count[level] > servers[levels.index(level)]:
            return False
    return True


def is_comb_valid2(comb, levels, tasks):
    last_task_num = 0
    len_tasks = len(tasks)
    for idx, task_num in enumerate(tasks):
        # #_print(comb[last_task_num:last_task_num+task_num], "\r")
        for allocate_level in comb[last_task_num : last_task_num + task_num]:
            if idx + 1 < len_tasks and allocate_level in levels[idx + 1 :]:
                return False
        last_task_num = last_task_num + task_num
    return True


def encode_comb(comb):
    return "-".join(str(num) for num in comb)


def decode_comb(comb_str):
    return tuple(int(num) for num in comb_str.split("-"))


def encode_key(servers: list, tasks: list, levels: list):
    return f"{servers}-{tasks}-{levels}"


def decode_key(encoded_key):
    servers_str, tasks_str, levels_str = encoded_key.split("-")
    servers = list(eval(servers_str))
    tasks = list(eval(tasks_str))
    levels = list(eval(levels_str))
    return servers, tasks, levels


# TODO 打表，加快生成速度
all_server_task_level_2_comb = []


def allocate(servers: list, tasks: list, levels: list) -> list[tuple]:
    """
    根据业务员数量、任务数量和等级列表,生成有效的分配组合

    输入:
    - servers: list,每个等级的业务员数量列表
    - tasks: list,每个等级的任务数量列表
    - levels: list,等级列表

    输出:
    - selected_combinations: list[tuple],有效的分配组合列表,每个组合为一个元组

    注意:
    - 分配组合需要满足以下条件:
      1. 每个任务的分配等级不能高于其本身的等级 （这个在生成的时候已经保证了这个条件）
      2. 每个等级分配的任务数量不能超过该等级的业务员数量
    """
    # _print(f"{servers=}\n{tasks=}\n{levels=}")
    # for i in range(task_sum):
    # levels
    # allocate_lv_idx_range_list = list(range(last_server_lv_idx, last_task_lv_idx+1))

    task_sum = sum(tasks)
    server_sum = sum(servers)
    is_case_task_less_than_server = task_sum < server_sum
    if is_case_task_less_than_server:
        should_allocate_tasks = tasks
    else:
        task_should_allocate_less = task_sum - server_sum
        should_allocate_tasks = tasks.copy()
        should_allocate_tasks[-1] -= task_should_allocate_less

    allocate_ranges = [
        levels[: idx + 1]
        for idx, task_num in enumerate(should_allocate_tasks)
        for i in range(task_num)
    ]
    allocate_combinations = list(itertools.product(*allocate_ranges))
    selected_combinations = []
    for comb in allocate_combinations:
        # 满足两个条件的则放入
        # 对comb 相同等级分配的比如 / 同时一个tasks的分组里，
        if is_comb_valid(comb, levels, servers):

            # #_print(f"{comb=}")
            selected_combinations.append(comb)

    return selected_combinations


if __name__ == "__main__":
    selected_combinations = allocate(servers=[3, 0, 5], tasks=[2, 0, 3], levels=[2, 3, 4])
    print(selected_combinations)
    selected_combinations2 = allocate(servers=[1, 0, 0, 2, 1], tasks=[0, 0, 0, 2, 0] , levels=[1, 2, 3, 4, 5])
    print(selected_combinations2)
