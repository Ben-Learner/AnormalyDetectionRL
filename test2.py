# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/6 18:22
@Auth ： Ben Qi
@File ：test2.py
@IDE ：PyCharm
@Motto：LIM(Less Is More)
"""
import pandas as pd

# 创建一个包含数据的 DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Charlie', 'Alice', 'Alice'],
        'Age': [25, 30, 35, 40, 45, 50, 55, 60]}
df = pd.DataFrame(data)

# 查看 'Name' 列中 'Eve' 的个数
print(df['Name'].value_counts()['Eve'])
