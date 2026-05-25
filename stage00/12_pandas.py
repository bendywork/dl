# ============================================================
# 章节：Pandas —— Python 数据分析扩展库
# 涵盖：Series / DataFrame 的创建、访问、修改、属性、方法、
#        文件读写（CSV / Excel）
# ============================================================

import numpy as np
import pandas as pd


# === 创建 Series：标量 ===

d = 99
ser = pd.Series(data=d)
print(ser)

ser = pd.Series(data=d, index=[1, 2, 3])
print(ser)


# === 创建 Series：字符串（当作标量处理）===

d = 'abc'
ser = pd.Series(data=d, index=[1, 2, 3])
print(ser)


# === 创建 Series：列表 ===

d = ['a', 'b', 'c']
ser = pd.Series(data=d)
print(ser)


# === 创建 Series：ndarray ===

d = np.array([1, 2, 3])
ser = pd.Series(data=d, dtype=np.float64, index=('one', 'two', 'three'), name='test-series')
print(ser)


# === 创建 Series：字典（键作为 index）===

d = {'a': 1, 'b': 2, 'c': 3}
ser = pd.Series(data=d)
print(ser)


# === 创建 Series：字典（指定不存在的 index → NaN）===

d = {'a': 1, 'b': 2, 'c': 3}
ser = pd.Series(data=d, index=['a', 'y', 'z'])
print(ser)


# === 访问 Series 数据：位置索引 ===

d = np.array([1, 2, 3, 4, 5])
ser = pd.Series(data=d, index=('a', 'e', 'c', 'd', 'e'))
print(ser)
print(ser[1])
print(ser[1:3])
print(ser[:-2:2])
print(ser[[2, 1, 3]])


# === 访问 Series 数据：索引标签 ===

d = np.array([1, 2, 3, 4, 5])
ser = pd.Series(data=d, index=('a', 'e', 'c', 'd', 'e'))
print(ser)
print(ser['c'])
print(ser['e'])
# 索引标签切片时，右边是闭区间
print(ser['a':'d'])
print(ser[:'c':2])
print(ser[['c', 'e', 'd']])


# === 修改 Series 索引 ===

ser = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'd'])
print(ser)
ser.index = ['aa', 'bb', 'cc', 'dd']
print(ser)


# === 修改 Series 数据 ===

ser = pd.Series([2, 3, 4, 5], index=['a', 'b', 'c', 'd'])
ser['a'] = 8
print(ser)
ser['b':'d'] = [7, 8, 9]
print(ser)


# === Series 常用属性 ===

d = [1, 2, 3, 4]
ser = pd.Series(data=d, index=['a', 'b', 'c', 'd'], name="Test-Series")
print(ser.dtype)
print(ser.name)
print(ser.size)
print(ser.values)
print(ser.index)


# === Series 运算 ===

ser1 = pd.Series([15, 20], index=["a", "b"])
print(ser1 + 1)
print(ser1 - 1)
print(ser1 * 2)
print(ser1 / 2)

ser2 = pd.Series([1, 2], index=["c", "a"])
print(ser1 + ser2)
print(ser1 - ser2)
print(ser1 * ser2)
print(ser1 / ser2)


# ============================================================
# DataFrame
# ============================================================

# === 创建 DataFrame：ndarray ===

d = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(data=d, dtype=np.float64)
print(df)


# === 创建 DataFrame：单一列表 ===

d = ['Tom', 'Bob', 'Linda']
df = pd.DataFrame(data=d)
print(df)


# === 创建 DataFrame：嵌套列表 ===

d = [['Tom', 17], ['Bob', 18], ['Linda', 26]]
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'], columns=['name', 'age'])
print(df)


# === 创建 DataFrame：字典嵌套列表 ===

d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)


# === 创建 DataFrame：Series 字典 ===

d = {
    'name': pd.Series(['Tom', 'Bob', 'Linda'], index=['p1', 'p2', 'p3']),
    'age': pd.Series([17, 18, 26], index=['p1', 'p2', 'p8'])
}
df = pd.DataFrame(data=d)
print(df)

d = [
    pd.Series(['Tom', 'Bob', 'Linda'], index=['p1', 'p2', 'p3'], name="name"),
    pd.Series([17, 18, 26], index=['p1', 'p2', 'p3'], name='age')
]
df = pd.DataFrame(data=d)
print(df)


# === 访问 DataFrame 数据：索引获取列，切片获取行 ===

d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26], 'height': [172, 176, 188]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)

# 索引获取列数据
print(df['age'])
print(df[['age', 'name']])

# 切片获取行数据
print(df[0: 1])       # 下标切片左闭右开
print(df['p1': 'p2'])  # 标签切片两边都是闭区间

# 组合使用
print(df[['name', 'age']][0: : 2])
print(df[0: : 2][['name', 'age']])


# === 访问 DataFrame 数据：loc（标签）/ iloc（下标）===

d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26], 'height': [172, 176, 188]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)

# loc：只能接收标签索引
print(df.loc['p1'])
print(df.loc['p2', 'age'])
print(df.loc['p2', ['age', 'name']])
print(df.loc[['p3', 'p2'], ['age', 'name']])

# iloc：只能接收整数索引
print(df.iloc[0])
print(df.iloc[1, 1])
print(df.iloc[1, [1, 0]])
print(df.iloc[[2, 1], [1, 0]])


# === 修改 DataFrame 索引 ===

d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26], 'height': [172, 176, 188]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)

df.index = ['n1', 'n2', 'n3']
df.columns = ['names', 'ages', 'heights']
print(df)


# === 修改 DataFrame 数据 ===

d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26], 'height': [172, 176, 188]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)

# 修改单列数据（多种等价写法）
df['height'] = pd.Series([1.72, 1.88, 1.76], index=df.index)
df['height'] = [1.72, 1.88, 1.76]
df.loc[:, 'height'] = [1.72, 1.88, 1.76]
df.iloc[:, 2:3] = [1.72, 1.88, 1.76]
print(df)

# 修改多列数据
df[['name', 'age']] = pd.DataFrame({'name': ['Bob', 'Tom', 'Jack'], 'age': [19, 22, 27]}, index=df.index)
df[['name', 'age']] = [['Bob', 19], ['Tom', 22], ['Jack', 27]]
df.loc[:, ['name', 'age']] = [['Bob', 19], ['Tom', 22], ['Jack', 27]]
df.iloc[:, :2] = [['Bob', 19], ['Tom', 22], ['Jack', 27]]
print(df)

# 追加单列数据
df['weight'] = pd.Series([65, 75, 60], index=df.index)
df['weight'] = [65, 75, 60]
df.loc[:, 'weight'] = [65, 75, 60]
print(df)

# 追加多列数据
df[['grade', 'address']] = pd.DataFrame(
    {'grade': ['一', '二', '三'], 'address': ['威宁路', '长宁路', '大马路']},
    index=df.index
)
df[['grade', 'address']] = [['一', '威宁路'], ['二', '长宁路'], ['三', '大马路']]
df.loc[:, ['grade', 'address']] = [['一', '威宁路'], ['二', '长宁路'], ['三', '大马路']]
print(df)

# 修改单行数据
df[1:2] = ['Tony', 23, 178]
df.loc['p2'] = pd.Series(['Tony', 23, 178], index=df.columns)
df.iloc[1] = ['Tony', 23, 178]
df.iloc[1:2] = ['Tony', 23, 178]
print(df)

# 修改多行数据
df[:2] = [['Jack', 27, 1.76], ['Tony', 19, 1.72]]
df.loc[:'p2'] = [['Jack', 27, 1.76], ['Tony', 19, 1.72]]
df.iloc[[0, 1]] = [['Jack', 27, 1.76], ['Tony', 19, 1.72]]
print(df)

# 追加单行数据
df.loc['p4'] = ['Toby', 23, 178]
print(df)


# === DataFrame 常用属性 ===

d = [['Tom', 17], ['Bob', 18], ['Linda', 26]]
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'], columns=['name', 'age'])
print(df)
print(df.T)
print(df.dtypes)
print(df.shape)
print(df.size)
print(df.index)
print(df.columns)
print(df.axes)
print(df.values)


# ============================================================
# DataFrame 常用方法
# ============================================================

# === isnull / notnull：检测缺失值 ===

d = [[8, np.nan],
     [np.nan, 7],
     [0, 2],
     [np.nan, np.nan]]
df = pd.DataFrame(data=d)
print(df)
print(df.isnull())
print(df.notnull())


# === insert：插入新列 ===

d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)
df.insert(2, 'weight', [65, 75, 60])
print(df)


# === reindex：重新索引 ===

data = np.arange(12).reshape(3, 4)
df = pd.DataFrame(data, index=['n1', 'n2', 'n3'], columns=['a', 'b', 'c', 'd'])
print(df)

df2 = df.reindex(labels=['n2'], axis=0)
print(df2)
df2 = df.reindex(index=['n2'])
print(df2)

df2 = df.reindex(labels=['c'], axis=1)
print(df2)
df2 = df.reindex(columns=['c'])
print(df2)

df2 = df.reindex(index=['n2', 'n1', 'n4'], fill_value=np.pi)
print(df2)


# === pd.concat：拼接 DataFrame ===

df = pd.DataFrame([[1, 2], [3, 4]], index=['p1', 'p2'], columns=list('AB'))
print(df)
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AC'))
print(df2)

print(pd.concat([df, df2]))
print(pd.concat([df, df2], join='inner'))
print(pd.concat([df, df2], axis=1))

df2.index = [0, 'p1']
print(df2)
print(pd.concat([df, df2], axis=1, join='inner'))


# === pd.merge：合并 DataFrame ===

d1 = {'name': ['Tom', 'Bob', 'Jack'], 'age': [18, 17, 19], 'weight': [65, 66, 67]}
df1 = pd.DataFrame(data=d1)
d2 = {'name': ['Tom', 'Jack'], 'height': [168, 187], 'weight': [65, 68]}
df2 = pd.DataFrame(data=d2)
print(df1)
print(df2)

print(pd.merge(df1, df2, how='inner', on='name'))
print(pd.merge(df1, df2, how='left', on='name'))
print(pd.merge(df1, df2, how='right', on='name'))
print(pd.merge(df1, df2, how='outer', on='name'))


# === drop：删除行或列 ===

df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=['n1', 'n2', 'n3'], columns=['a', 'b'])
print(df)

print(df.drop(labels='n2', axis=0))
print(df.drop(index='n2'))

print(df.drop(labels='b', axis=1))
print(df.drop(columns='b'))

print(df.drop(labels=['n2', 'n1'], axis=0))
print(df.drop(index=['n2', 'n1']))

print(df.drop(labels=['a', 'b'], axis=1))
print(df.drop(columns=['a', 'b']))

df.drop(index='n1', inplace=True)
print(df)


# === dropna：删除缺失值所在行或列 ===

d = {'name': ['Tom', np.nan, 'Bob'], 'age': [np.nan, np.nan, 19], 'height': [177, 182, 179]}
df = pd.DataFrame(data=d)
print(df)

print(df.dropna())
print(df.dropna(axis=1))

df.loc[2, 'age'] = np.nan
print(df)

print(df.dropna(axis=1, how='all'))
print(df.dropna(axis=1, thresh=2))
print(df.dropna(subset=['name', 'height']))
print(df.dropna(axis=1, subset=[1, 2]))

df.dropna(axis=1, inplace=True)
print(df)


# === fillna：填充缺失值 ===

df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, np.nan],
                   [np.nan, 3, np.nan, 4]],
                  columns=list("ABCD"))
print(df)

print(df.fillna(0))

dic = {'A': 6, 'B': 7}
print(df.fillna(dic))

np.random.seed(3)
arr = np.random.randint(1, 10, size=(3, 5))
df2 = pd.DataFrame(arr, columns=list("CFAHB"))
print(df2)
print(df.fillna(df2))

print(df.fillna(method='ffill'))
print(df.fillna(method='bfill'))
print(df.fillna(method='ffill', axis=1))
print(df.fillna(method='bfill', axis=1))
print(df.fillna({'A': 6, 'C': 7}, limit=2))


# === info：打印简明摘要 ===

df = pd.DataFrame(data={'name': ['Tom', 'Bob', np.nan], 'age': [18, 19, 17], 'height': [167, 177, 178]},
                  index=['n1', 'n2', 'n3'])
print(df)
df.info()
df.info(verbose=False)
df.info(show_counts=False)


# === describe：描述性统计 ===

df = pd.DataFrame(data={'name': ['Tom', 'Bob', 'Bob'], 'age': [18, 19, 17], 'height': [167, 177, 178]},
                  index=['n1', 'n2', 'n3'])
print(df)
print(df.describe())
print(df.describe(include='all'))
print(df.describe(include='object'))
print(df.describe(include=['number', 'object']))


# === count / max / min / mean / var / std：统计方法 ===

df = pd.DataFrame(data={'name': ['Tom', np.nan, 'Linda'], 'age': [18, 19, 17]},
                  index=['n1', 'n2', 'n3'])
print(df)
print(df.count())
print(df.count(axis=1))

d = np.random.normal(size=(7, 2))
df = pd.DataFrame(data=d)
print(df)
print(df.max(axis=0))
print(df.max(axis=1))
print(df.min(axis=0))
print(df.min(axis=1))
print(df.mean(axis=0))
print(df.mean(axis=1))
print(df.var(axis=0))
print(df.var(axis=1))
print(df.std(axis=0))
print(df.std(axis=1))


# === sample：随机采样 ===

df = pd.DataFrame(data={'name': ['Tom', 'Bob', 'Jack', 'Linda'],
                        'age': [18, 19, 17, 21],
                        'height': [167, 177, 178, 188]},
                  index=['n1', 'n2', 'n3', 'n4'])
print(df)
print(df.sample())
print(df.sample(frac=0.75))
print(df.sample(n=2))
print(df.sample(n=2, replace=True))
print(df.sample(n=2, axis=1))
print(df.sample(n=2, random_state=3))


# === drop_duplicates：去重 ===

d = {'A': [1, 3, 3, 1], 'B': [0, 2, 5, 0], 'C': [4, 0, 4, 4], 'D': [1, 0, 0, 1]}
df = pd.DataFrame(data=d)
print(df)

print(df.drop_duplicates())
print(df.drop_duplicates(keep='last'))
print(df.drop_duplicates(keep=False))
print(df.drop_duplicates(subset=['A', 'D'], keep='last'))

df.drop_duplicates(subset=['A', 'D'], keep='last', inplace=True)
print(df)


# === sort_values：排序 ===

df = pd.DataFrame({'col1': [4, 1, 2, np.nan, 5, 2],
                   'col2': [2, 1, 9, 8, 7, 6],
                   'col3': [0, 1, 9, 4, 2, 3],
                   'col4': ['a', 'B', 'c', 'D', 'e', 1]})
print(df)
print(df.sort_values(by=['col1']))
print(df.sort_values(by='col1'))
print(df.sort_values(by=['col1', 'col2']))
print(df.sort_values(by=5, axis=1))
print(df.sort_values(by=5, axis=1, ascending=False))
print(df.sort_values(['col1', 'col2'], ascending=[True, False]))
print(df.sort_values(by='col1', na_position='first'))
df.sort_values(by='col1', inplace=True)
print(df)


# === apply：对每行/每列应用函数 ===

d = [[1, 2, 0], [4, 1, 9], [2, 5, 7], [4, 3, 6]]
df = pd.DataFrame(d, columns=['A', 'B', 'C'])
print(df)
print(df.apply(np.sum))
print(df.apply(np.sum, axis=1))


# === groupby：分组聚合 ===

d = {
    'company': ['A', 'B', 'A', 'C', 'C', 'B', 'C', 'A'],
    'salary': [8, 15, 10, 15, np.nan, 28, 30, 15],
    'age': [26, 29, 26, 30, 50, 30, 30, 35]
}
df = pd.DataFrame(data=d)
print(df)

df_gb = df.groupby(by='company', as_index=False)

for g, data in df_gb:
    print(g)
    print(data)

print(df_gb.ngroups)
print(df_gb.groups)
print(df_gb.indices)

print(df_gb.get_group('A'))
print(df_gb.get_group('B'))
print(df_gb.get_group('C'))

print(df_gb.agg('mean'))
print(df_gb.agg(np.mean))
print(df_gb.agg('max'))
print(df_gb.agg('min'))
print(df_gb.agg('sum'))
print(df_gb.agg('median'))
print(df_gb.agg('std'))
print(df_gb.agg('var'))
print(df_gb.agg('count'))

print(df_gb.transform('mean'))
print(df_gb.transform(np.mean))

df[['avg_salary', 'avg_age']] = df_gb.transform('mean')
print(df)


# === DataFrame 运算 ===

d = np.arange(9).reshape((3, 3))
df1 = pd.DataFrame(data=d, columns=list('abc'), index=['n1', 'n2', 'n3'])
print(df1)
print(df1 + 1)
print(df1 - 1)
print(df1 * 2)
print(df1 / 2)

d = np.arange(16).reshape((4, 4))
df2 = pd.DataFrame(data=d, columns=list('dacf'), index=['n1', 'n2', 'n3', 'n4'])
print(df2)
print(df1 + df2)
print(df1 - df2)
print(df1 * df2)
print(df1 / df2)


# ============================================================
# Pandas 文件读写
# ============================================================

# === CSV 文件读取 ===
# 注意：以下读取示例需要对应的 CSV / Excel 文件存在，运行时请按
# 实际路径准备测试文件，或按需注释相关行。

# df = pd.read_csv('./test01.csv')
# print(df)
#
# df = pd.read_csv('./test02.csv', sep=';')
# print(df)
#
# df = pd.read_csv('./test03.csv', sep=';', header=None)
# print(df)
#
# df = pd.read_csv('./test03.csv', sep=';', header=2)
# print(df)
#
# df = pd.read_csv('./test02.csv', sep=';', names=['name', 'age', 'height'])
# print(df)
#
# df = pd.read_csv('./test01.csv', nrows=2)
# print(df)
#
# df = pd.read_csv('./test01.csv', skiprows=2)
# print(df)
#
# df = pd.read_csv('./test01.csv', skiprows=[0, 2])
# print(df)
#
# df = pd.read_csv('./test01.csv', usecols=[0, 2])
# print(df)
#
# obj = pd.read_csv('./test01.csv', chunksize=2)
# for i in obj:
#     print(i)


# === CSV 文件写入 ===

d = {
    '名字': ['张三', '李四', '王五', '赵六', '孙七'],
    '年龄': [18, 19, 20, 22, 17],
    '身高': [188, 178, 189, 175, 177]
}
df = pd.DataFrame(data=d)
print(df)

# df.to_csv('./test04.csv')
# df.to_csv('./test05.csv', sep=';')
# df.to_csv('./test06.csv', index=False)
# df.to_csv('./test07.csv', header=False)


# === Excel 文件写入 ===

d = {
    '名字': ['张三', '李四', '王五', '赵六', '孙七'],
    '年龄': [18, 19, 20, 22, 17],
    '身高': [188, 178, 189, 175, 177]
}
df = pd.DataFrame(data=d)
print(df)

# df.to_excel('./test08.xlsx')
# df.to_excel('./test09.xlsx', index=False)
# df.to_excel('./test10.xlsx', header=False)

# with pd.ExcelWriter('./test11.xlsx') as writer:
#     df.to_excel(writer, sheet_name='工作表1', index=False)
#     df.iloc[:, :2].to_excel(writer, sheet_name='工作表2', index=False)


# === Excel 文件读取 ===

# df = pd.read_excel('./test11.xlsx')
# print(df)
#
# df = pd.read_excel('./test11.xlsx', header=None)
# print(df)
#
# df = pd.read_excel('./test11.xlsx', header=2)
# print(df)
#
# df = pd.read_excel('./test11.xlsx', names=['name', 'age', 'height'])
# print(df)
#
# df = pd.read_excel('./test11.xlsx', header=None, names=['name', 'age', 'height'])
# print(df)
#
# df = pd.read_excel('./test11.xlsx', sheet_name=1)
# print(df)
#
# df = pd.read_excel('./test11.xlsx', sheet_name='工作表2')
# print(df)
#
# df = pd.read_excel('./test11.xlsx', sheet_name=[0, '工作表2'])
# print(df)
#
# df = pd.read_excel('./test11.xlsx', nrows=2)
# print(df)
#
# df = pd.read_excel('./test11.xlsx', skiprows=2)
# print(df)
#
# df = pd.read_excel('./test11.xlsx', skiprows=[0, 2])
# print(df)
#
# df = pd.read_excel('./test11.xlsx', usecols=[0, 2])
# print(df)

# === END p14 ===
