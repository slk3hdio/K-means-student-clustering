import numpy as np
from scipy.sparse.linalg import eigsh

def spectral_embedding(adj_matrix, n_components=2, debug = False):
    """
    实现谱嵌入算法

    参数:
    adj_matrix: numpy.ndarray, 邻接矩阵 (n x n)
    n_components: int, 嵌入向量的维度

    返回:
    embedding: numpy.ndarray, 嵌入向量 (n x n_components)
    """
    # 1. 计算度矩阵 D
    degree = np.sum(adj_matrix, axis=1)  # 每个节点的度
    # 2. 计算归一化拉普拉斯矩阵 L_sym
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))  # D^{-1/2}
    L_sym = np.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    if debug:
        print("度：")
        print(degree)
        print("归一化拉普拉斯矩阵：")
        print(L_sym)

    # 3. 特征值分解
    eigenvalues, eigenvectors = eigsh(L_sym, k=n_components, which='SM', sigma = 1e-10)

    if debug:
        print("特征值：")
        print(eigenvalues)
        print("特征向量：")
        print(eigenvectors)

    # 4. 选择特征向量
    embedding = eigenvectors

    # embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    return embedding


if __name__ == '__main__':
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])

    # G = nx.from_numpy_array(adj_matrix)
    # pos = nx.spring_layout(G)  # 使用 Fruchterman-Reingold 布局
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16, edge_color='gray')
    # plt.show()

    # 计算谱嵌入
    embedding = spectral_embedding(adj_matrix, n_components=2, debug=True)
    print("嵌入向量：")
    print(embedding)

    # 可视化
    # plt.scatter(embedding[:, 0], embedding[:, 1], c='r', s=10)
    # plt.show()

