from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3,
                          include_bias=False,
                          interaction_only=True)
"""
degree：扩展的阶数
interaction_only： 是否只保留交互项
include_bias：是否包含偏置项
"""
x = [[1, 2, 3],
     [2, 3, 4]]
poly.fit(x)
x_poly = poly.transform(x)

# x_poly = poly.fit_transform(x)
print(x_poly)
