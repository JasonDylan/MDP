import pandas as pd
import numpy as np

class MDP():
    # 生成城市规模/一个省
    city_num = 26
    DATA_TABLE_PATH = './data/数据.xlsx' # 此处修改文件路径 TODO
    DIST_MATRIX_PATH = r"C:\Users\dylan\Desktop\code\paper\data\中国各城市空间权重矩阵(1).xlsx"
    PATH_CITY_DIST = f"C:\Users\dylan\Desktop\code\paper\city_{city_num}.xlsx"
    ## 收益率
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    
    # 全局设定种子，保证每次随机结果一致
    np.random.seed(42)
    

    def __init__(self):
        self.generate_city()
        proveng = self.city_distance_df.index.get_level_values(
            "proveng"
        ).unique()  # 获得所有province的name
        proveng_dict = {proveng[i]: i + 1 for i in range(len(proveng))}
        city_names = self.city_distance_df.columns
        a_city_distance_df, city_num_2_name = self.change_df_city_name_2_idx(
            cities=self.city_distance_df
        )
        pass
    
    def change_df_city_name_2_idx(cities: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """改名字为idx，同时返回新df和idx2name"""
        city_names = cities.columns
        city_num = len(city_names)
        city_name_nums = list(range(1, city_num + 1))
        city_num_2_name = {key: value for key, value in zip(city_name_nums, city_names)}
        new_cities = cities.copy()
        new_cities.columns = new_cities.index = city_name_nums

    def read_excel_table(self):
        # 一个通用方法，用于读取excel表
        self.arriving_rate_df = pd.read_excel(self.DATA_TABLE_PATH, sheet_name='arriving rate', index_col=0)
        self.travel_fee_df = pd.read_excel(self.DATA_TABLE_PATH, sheet_name="travel fee", index_col=0)
        self.initial_state_df = pd.read_excel(self.DATA_TABLE_PATH, sheet_name="initial state", index_col=0)
        self.servers_df = pd.read_excel(self.DATA_TABLE_PATH, sheet_name="servers", index_col=0)
    
    def get_proveng_city_dist_mat_df(self):
        self.proveng_city_dist_mat_df = pd.read_excel(
            self.DIST_MATRIX_PATH,
            sheet_name="地理距离矩阵",
        )

    def get_city_series_to_city_dict(self):
        self.city_series_to_city_dict = (
            self.proveng_city_dist_mat_df[["cityeng", "cityseries"]]
            .drop_duplicates()
            .set_index("cityseries")["cityeng"]
            .to_dict()
        )
    
    def get_city_2_proveng_dict(self):
        self.city_to_proveng_dict = (
            self.proveng_city_dist_mat_df[["cityeng", "proveng"]]
            .drop_duplicates()
            .set_index("cityeng")["proveng"]
            .to_dict()
        )

    def get_dist_mat_from_xls_df(self,):
        self.distance_mat = self.proveng_city_dist_mat_df.iloc[:, self.distance_mat_start_idx:]
        self.distance_mat.index = self.proveng_city_dist_mat_df.iloc[:, self.cityseries_idx].values
        self.distance_mat.columns = self.proveng_city_dist_mat_df.iloc[:, self.cityseries_idx].values


    def get_select_city_df(self):
        # 挑选出安徽、江苏、浙江和上海的省份及对应的矩阵数据
        select_proveng = ["Anhui", "Jiangsu", "Zhejiang", "Shanghai"]  # TODO 可能要换省份
        provengs = self.proveng_city_dist_mat_df.iloc[
            :, self.proveng_idx
        ].values  # 表格第一列为城市名称
        select_proveng_idxs = [
            i for i, prov in enumerate(provengs) if prov in select_proveng
        ]  # xls里选取指定的省份的idx，是字母
        select_cityeng_idxs = self.proveng_city_dist_mat_df.values[
            select_proveng_idxs, self.cityseries_idx
        ]  # xls指定省份对应的城市id
        select_idx = pd.Index(select_cityeng_idxs)  # 创建对应索引
        # 使用 loc 方法选择相应的行和列
        select_proveng_city_df = self.distance_mat.loc[select_idx, select_idx]

        # 从城市列表中随机选择city_num-1个城市（不包括上海）
        rnd_cities = np.random.choice(
            select_cityeng_idxs[1:], self.city_num - 1, replace=False
        )  # 第一列为上海，所以从1开始
        # 将上海添加到随机选择的城市列表中 # City158 = shanghai
        select_cities = ["City158"] + rnd_cities.tolist()
        select_city_idx = pd.Index(select_cities)
        # 使用 loc 方法选择相应的行和列
        self.select_city_df = select_proveng_city_df.loc[select_city_idx, select_city_idx]


    def generate_city(self):
        """本函数功能，生成指定数量城市的城市距离矩阵，必然包含上海。
        返回的dataframe包含城市坐标和省份坐标/和一个城市转省份的dict"""

        # 地理距离矩阵xls
        self.get_proveng_city_dist_mat_df()
        # 从第四列开始，是距离矩阵内容
        self.proveng_idx = 0
        self.cityeng = 1
        self.cityseries_idx = 2
        self.distance_mat_start_idx = 3

        self.get_city_series_to_city_dict()
        self.get_city_2_proveng_dict()

        # 地理距离矩阵xls 第三列是城市id, city_*, 设定纵坐标
        self.get_dist_mat_from_xls_df()
        self.get_select_city_df()

        city_series_columns = self.select_city_df.columns
        city_columns = [
            (
                self.city_to_proveng_dict[self.city_series_to_city_dict[item]],
                self.city_series_to_city_dict[item],
            )
            for item in city_series_columns
        ]  # 最为新矩阵的idx和col

        column_names = ["proveng", "cityeng"]  # 给idx的idx
        self.city_distance_df = pd.DataFrame(
            self.select_city_df.values,
            index=pd.MultiIndex.from_tuples(city_columns, names=column_names),
            columns=pd.MultiIndex.from_tuples(city_columns, names=column_names),
        )
        
        self.city_distance_df.to_excel(self.PATH_CITY_DIST)
        print(self.PATH_CITY_DIST)