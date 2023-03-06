# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/6 16:41
@Auth ： Ben Qi
@File ：data_process.py
@IDE ：PyCharm
@Motto：LIM(Less Is More)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = './data/SGTR(1kgs)_400_100.csv'

class ProcessData:
    """
    原始数据预处理
    """
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, index_col="TIME(10ms)", encoding='gbk')
    def plot_trend(self):
        """
        绘制原始数据各参数变化趋势，确实异常开始位置
        """
        selected_cols = ['N00\JEG\CTEC_ATAG0@Out.Val(1#一回路压力)', 'N00\JEG\CTEC_ATAG1@Out.Val(2#一回路压力)',
                         'JKA01\JKA01\CTEC_ATAG0@Out.Val(1#核功率)', 'JKA02\JKA01\CTEC_ATAG0@Out.Val(2#核功率)']
        df_selected = self.df[selected_cols]
        plt.plot(df_selected['N00\JEG\CTEC_ATAG0@Out.Val(1#一回路压力)'], label='col1')
        plt.plot(df_selected['N00\JEG\CTEC_ATAG1@Out.Val(2#一回路压力)'], label='col2')
        plt.xlabel('Time(10ms)')
        plt.ylabel('Pressure(MPa)')
        plt.title('Trend graph')
        plt.legend()
        plt.show()
    @staticmethod
    def data_norm(df):
        """
        对数据进行归一化
        Args:
            df: 归一化前的dataframe

        Returns:
            df:归一化后的dataframe
        """
        for i in df.columns:
            min_value = np.min(df[i])
            max_value = np.max(df[i])
            df[i] = (df[i] - min_value) / (max_value - min_value + 1e-6)
        return df
    def add_label(self):
        """
        在原始数据最后一列添加标签，标注为正常（0）或者异常状态（1）
        """
        # 定义阈值
        nuc_power_th = 186
        new_df = self.df
        label_col = new_df['JKA01\JKA01\CTEC_ATAG0@Out.Val(1#核功率)'].apply(lambda x: 1 if x < nuc_power_th else 0)
        new_df = self.data_norm(new_df)
        new_df['label'] = label_col
        return new_df


if __name__ == "__main__":
    data_pro = ProcessData(path)
    df = data_pro.add_label()
    df = df.iloc[0:28,:]
    count = df['label'].value_counts()
    print(count)
    print(count[1])
