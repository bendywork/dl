import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# SVM 工程案例：乳腺癌分类
# 核心步骤：标准化 → 核函数选择 → C/gamma 网格搜索
# ============================================================
data = load_breast_cancer()
X, y = data.data, data.target
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y)

# 必须标准化！SVM对尺度敏感
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

# ---------- 对比四种核函数 ----------
kernels = ['linear','poly','rbf','sigmoid']
for k in kernels:
    svm = SVC(kernel=k, C=1.0, random_state=42)
    svm.fit(X_tr, y_train)
    acc = svm.score(X_te, y_test)
    nsv = svm.n_support_.sum()
    print(f"kernel={k:<8} acc={acc:.4f}  支持向量数={nsv}")

# ---------- RBF 核网格搜索最优 C、gamma ----------
param_grid = {'C':[0.1,1,10,100],'gamma':['scale','auto',0.01,0.001]}
grid = GridSearchCV(SVC(kernel='rbf',random_state=42),param_grid,cv=5,n_jobs=-1)
grid.fit(X_tr, y_train)
print(f"\n最优参数: {grid.best_params_}")
print(f"最优CV准确率: {grid.best_score_:.4f}")
best = grid.best_estimator_
print(f"测试集准确率: {best.score(X_te,y_test):.4f}")
print("\n分类报告:\n", classification_report(y_test, best.predict(X_te),
      target_names=data.target_names))

# ---------- 可视化：C 对准确率的影响（固定gamma='scale'）----------
Cs = [0.01,0.1,1,10,100,1000]
accs = []
for c in Cs:
    m = SVC(kernel='rbf',C=c,gamma='scale',random_state=42)
    m.fit(X_tr,y_train); accs.append(m.score(X_te,y_test))
plt.figure(figsize=(7,4))
plt.semilogx(Cs, accs, 'o-', color='steelblue')
plt.xlabel('C (log scale)'); plt.ylabel('Test Accuracy')
plt.title('RBF SVM：C 参数 vs 测试准确率')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/sunchengxin/PycharmProjects/dl/stage01/09.SVM/svm_result.png",dpi=120)
print("\n图表已保存 svm_result.png")
