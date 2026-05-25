from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# 从文件中读取数据
dataFrame  = pd.read_csv("../../../base/datas/text_classify/train.csv", header=None, sep = "\t")
dataFrame_test  = pd.read_csv("../../../base/datas/text_classify/test.csv", header=None, sep = "\t")
X_train = dataFrame[0]
y_train = dataFrame[1]
X_test = dataFrame_test[0]
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

model.fit(X_train, y_train)

print(model)
y_pred = model.predict(X_test)
# print("accuracy:", accuracy_score(y_train, y_pred))
# print(classification_report(y_train, y_pred))
print(y_pred)


