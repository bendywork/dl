import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 工程案例：乳腺癌分类（对比单棵决策树 vs 随机森林 vs GBDT）
# 数据集：sklearn内置，569样本，30特征，二分类（良性/恶性）
# ============================================================

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"类别分布: {np.bincount(y_train)}")

# ---------- 模型定义 ----------
models = {
    "单棵决策树(depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "单棵决策树(无限制)": DecisionTreeClassifier(random_state=42),
    "随机森林(100棵)":    RandomForestClassifier(n_estimators=100, random_state=42),
    "GBDT(100轮)":       GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ---------- 训练、评估 ----------
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    cv_scores = cross_val_score(model, X, y, cv=5)
    results[name] = {"train": train_acc, "test": test_acc, "cv": cv_scores.mean()}
    print(f"\n{name}")
    print(f"  训练集准确率: {train_acc:.4f}  测试集: {test_acc:.4f}  5折CV: {cv_scores.mean():.4f}")

# ---------- 随机森林特征重要性 ----------
rf = models["随机森林(100棵)"]
importances = rf.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]
print("\n随机森林 Top5 重要特征:")
for i in top5_idx:
    print(f"  {feature_names[i]:<35} {importances[i]:.4f}")

# ---------- 可视化 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names = list(results.keys())
train_accs = [results[n]["train"] for n in names]
test_accs  = [results[n]["test"]  for n in names]
x = np.arange(len(names))
axes[0].bar(x-0.2, train_accs, 0.4, label="Train", color="steelblue")
axes[0].bar(x+0.2, test_accs,  0.4, label="Test",  color="tomato")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
axes[0].set_ylim(0.8, 1.01)
axes[0].set_title("模型对比：训练 vs 测试准确率")
axes[0].legend()

axes[1].barh(range(5), importances[top5_idx], color="seagreen")
axes[1].set_yticks(range(5))
axes[1].set_yticklabels([feature_names[i] for i in top5_idx], fontsize=8)
axes[1].set_title("随机森林 Top5 特征重要性")
plt.tight_layout()
plt.savefig("/Users/sunchengxin/PycharmProjects/dl/stage01/08.集成学习/ensemble_result.png", dpi=120)
print("\n图表已保存 ensemble_result.png")
