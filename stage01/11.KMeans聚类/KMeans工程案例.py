import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# KMeans 工程案例
# 1. 肘部法则选 K
# 2. 对比 K-Means vs DBSCAN（月牙形数据）
# 3. 可视化聚类结果
# ============================================================

# ---------- 数据生成 ----------
X_blob, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
X_moon, _      = make_moons(n_samples=300, noise=0.08, random_state=42)
scaler = StandardScaler()
X_blob = scaler.fit_transform(X_blob)
X_moon = scaler.fit_transform(X_moon)

# ---------- 肘部法则：SSE 随 K 变化 ----------
ks = range(1, 11)
sses, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_blob)
    sses.append(km.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(X_blob, km.labels_))
    else:
        silhouettes.append(0)
print("K\tSSE\t\t轮廓系数")
for k, s, sil in zip(ks, sses, silhouettes):
    print(f"{k}\t{s:.2f}\t\t{sil:.4f}")

# ---------- 最终聚类 & 对比 DBSCAN ----------
km_best = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
km_best.fit(X_blob)
km_moon = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
km_moon.fit(X_moon)
db_moon = DBSCAN(eps=0.3, min_samples=5)
db_moon.fit(X_moon)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 肘部曲线
axes[0].plot(list(ks), sses, 'o-', color='steelblue')
axes[0].axvline(4, color='tomato', ls='--', label='最优K=4')
axes[0].set_xlabel('K'); axes[0].set_ylabel('SSE (Inertia)')
axes[0].set_title('肘部法则选K'); axes[0].legend()

# KMeans on blobs
axes[1].scatter(X_blob[:,0], X_blob[:,1], c=km_best.labels_, cmap='tab10', s=20, alpha=0.7)
centers = km_best.cluster_centers_
axes[1].scatter(centers[:,0], centers[:,1], marker='X', s=200, c='red', zorder=5)
axes[1].set_title('K-Means 聚类（球形数据）')

# KMeans vs DBSCAN on moons
axes[2].scatter(X_moon[:,0], X_moon[:,1], c=db_moon.labels_, cmap='tab10', s=20, alpha=0.7)
axes[2].set_title('DBSCAN 聚类（月牙形，K-Means失效场景）')

plt.tight_layout()
plt.savefig("/Users/sunchengxin/PycharmProjects/dl/stage01/11.KMeans聚类/kmeans_result.png", dpi=120)
print("\nK-Means最优轮廓系数(K=4):", silhouette_score(X_blob, km_best.labels_))
print("图表已保存 kmeans_result.png")
