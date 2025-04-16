import numpy as np

def kmeans(X, K, max_iters=100, tol=1e-4):
    # 初始化 K 个随机质心
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    
    for i in range(max_iters):
        # 计算每个样本到 K 个质心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 更新质心
        old_centroids = centroids.copy()
        for k in range(K):
            centroids[k] = X[labels == k].mean(axis=0)
        
        # 判断是否收敛
        if np.linalg.norm(centroids - old_centroids) < tol:
            break
    
    return centroids, labels


if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=[0, 0], scale=1, size=(100, 2)),
        np.random.normal(loc=[5, 5], scale=1, size=(100, 2))
    ])
    print("Data:\n", X)

    K = 2
    centroids, labels = kmeans(X, K)
    
    print("Centroids:\n", centroids)
    print("Labels:\n", labels)