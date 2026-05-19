from sklearn.multiclass import OneVsRestClassifier                                                                                                                                                            
from sklearn.preprocessing import MultiLabelBinarizer                                                                                                                                                       

# 原始标签（每个样本有多个类别）                                                                                                                                                                              
y_raw = [{0, 2}, {1, 2}, {0, 1}]
                                                                                                                                                                                                            
mlb = MultiLabelBinarizer()                                                                                                                                                                                 
y = mlb.fit_transform(y_raw)                                                                                                                                                                                  
print(y.shape)  # (3, 3)                                                                                                                                                                                    
print(y)    