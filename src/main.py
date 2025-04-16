from spectral_embedding import spectral_embedding
from visualization import visualize_undirected_graph, visulize_3d_plots, visualize_2d_plots
from K_means import kmeans

import numpy as np


# 从数据集和数据库中获取学生互验关系图和学生信息
if False:
    from get_data import get_graph, graph2mat
    graph = get_graph()
    mat = graph2mat(graph)
    count = 0
    labels = {}
    for student in graph.students:
        labels[count] = student.no
        count += 1

# 若数据库已经关闭，则直接使用以前保存的mat和labels
else:
    from result import labels, mat



embedding = spectral_embedding(mat, n_components=2, debug=True) # 使用谱嵌入将学生互验关系转化为2维空间
centrol_points, colors = kmeans(embedding, K=10)  # 使用K-means聚类算法将2维空间中的学生划分为K类， 使用不同颜色表示不同的类别
X = embedding[:, 0]
Y = embedding[:, 1]
visualize_2d_plots(X, Y, colors, labels )   # 绘制2维空间中的学生划分结果
visualize_undirected_graph(mat, labels, colors)  # 绘制学生互验关系图