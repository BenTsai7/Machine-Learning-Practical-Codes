import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

np.random.seed(0)
variables = ['X', 'Y', 'Z']
labels = ['GROUP_0', 'GROUP_1', 'GROUP_2', 'GROUP_3', 'GROUP_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)
# 计算距离关联矩阵，两两样本间的欧式距离
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
print(pd.DataFrame(row_clusters, columns=['row label1', 'row label2', 'distance', 'no. of items in clust.'],
                   index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]))
# 层次聚类树
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()
# 层次聚类热度图
fig = plt.figure(figsize=(8, 8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='right')
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

# 凝聚层次聚类，应用对层次聚类树剪枝
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print(labels)
