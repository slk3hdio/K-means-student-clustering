import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def visualize_undirected_graph(adj_matrix, labels, colors):
    """
    可视化一个无向图。

    参数:
    - adj_matrix: numpy 数组，表示邻接矩阵。
    - labels: 字典，键为节点索引，值为节点标签。
    """
    # 检查邻接矩阵是否对称（无向图的邻接矩阵必须对称）
    if not np.allclose(adj_matrix, adj_matrix.T):
        raise ValueError("邻接矩阵必须是关于主对角线对称的（无向图）。")
    
    # 创建一个无向图
    G = nx.Graph()
    
    # 添加节点
    num_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    
    # 添加边（根据邻接矩阵）
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # 只遍历上三角部分
            if adj_matrix[i, j] != 0:
                G.add_edge(i, j)
    
    
    # 绘制图
    pos = nx.spring_layout(G, k=0.1, iterations=10, scale=2) 
    nx.draw(G, pos, with_labels=False, node_color=colors, node_size=40, edge_color='gray', width=0.5)
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='black')
    
    # 显示图
    plt.title("graph")
    plt.show()


def visulize_3d_plots(X, Y, Z, colors, labels):
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X, Y, Z, marker='o', c = colors, s = 10)
    for i, txt in enumerate(labels):
        ax.text(X[i], Y[i], Z[i], txt, size=2, zorder=1, color='red')

    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

    plt.title("3D")
    plt.show()

def visualize_2d_plots(X, Y, colors, labels):
    fig, ax = plt.subplots()

    scatter = ax.scatter(X, Y, marker='o', c=colors, s = 10)
    for i, txt in enumerate(labels):
        ax.text(X[i], Y[i], txt, size=5, zorder=1, color='red')

    plt.title("2D ")
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 示例邻接矩阵（无向图）
    adj_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    
    # 示例标签字典
    labels = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }
    
    # 可视化无向图
    visualize_undirected_graph(adj_matrix, labels)