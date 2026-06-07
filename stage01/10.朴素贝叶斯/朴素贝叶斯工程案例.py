import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 朴素贝叶斯工程案例：新闻文本分类（4类）
# 数据：20 Newsgroups 子集
# 流程：TF-IDF向量化 → MultinomialNB → 评估
# ============================================================
cats = ['rec.sport.hockey','sci.space','talk.politics.guns','comp.graphics']
train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers','quotes'))
test  = fetch_20newsgroups(subset='test',  categories=cats, remove=('headers','footers','quotes'))
print(f"训练样本: {len(train.data)}, 测试样本: {len(test.data)}")
print(f"类别: {train.target_names}")

# Pipeline: TF-IDF + MultinomialNB
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
    ('nb',    MultinomialNB(alpha=0.1)),
])
pipe.fit(train.data, train.target)
pred = pipe.predict(test.data)
acc = (pred == test.target).mean()
print(f"\nMultinomialNB 准确率: {acc:.4f}")
print("\n分类报告:")
print(classification_report(test.target, pred, target_names=cats))

# ---------- 拉普拉斯平滑对比（alpha参数影响）----------
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
accs = []
tfidf = pipe.named_steps['tfidf']
X_tr = tfidf.transform(train.data)
X_te = tfidf.transform(test.data)
for a in alphas:
    m = MultinomialNB(alpha=a)
    m.fit(X_tr, train.target)
    accs.append(m.score(X_te, test.target))
    print(f"alpha={a:<6} acc={accs[-1]:.4f}")

# ---------- 混淆矩阵可视化 ----------
cm = confusion_matrix(test.target, pred)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
axes[0].set_xticklabels([c.split('.')[-1] for c in cats], rotation=20, fontsize=9)
axes[0].set_yticklabels([c.split('.')[-1] for c in cats], fontsize=9)
axes[0].set_title('混淆矩阵')
for i in range(4):
    for j in range(4):
        axes[0].text(j,i,cm[i,j],ha='center',va='center',fontsize=11,
                     color='white' if cm[i,j]>cm.max()/2 else 'black')

axes[1].semilogx(alphas, accs, 'o-', color='steelblue')
axes[1].set_xlabel('alpha（拉普拉斯平滑）'); axes[1].set_ylabel('准确率')
axes[1].set_title('平滑参数 alpha vs 准确率')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/sunchengxin/PycharmProjects/dl/stage01/10.朴素贝叶斯/nb_result.png", dpi=120)
print("\n图表已保存 nb_result.png")
