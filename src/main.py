#!/usr/bin/env python3
from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import time

global graph
Graph = namedtuple("Graph", ["nodes", "edges"])


def create_graph_by_list_adjacency(graph):
    start_time = time.time()

    list_adjacence = {}

    for node in graph.nodes:
        list_adjacence[node] = []

    for node1, node2 in graph.edges:
        list_adjacence[node1].append(node2)
        # if its graph not oriented add this
        # list_adjacence[node2].append(node1)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return list_adjacence


def create_graph_by_matrix_adjacency(graph):
    matrix = []
    for _ in nodes:
        matrix.append([0 for _ in nodes])

    for node1, node2 in graph.edges:
        matrix[node1 - 1][node2 - 1] += 1
        # if its graph not oriented add this
        # matrix[node2][node1] += 1

    return matrix


def calc_density_by_list():
    global graph
    start_time = time.time()

    number_of_edges = len(graph.keys())
    number_of_nodes = len(graph.values())

    densité = (2 * number_of_edges) / (number_of_nodes * (number_of_nodes - 1))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return densité


def calc_density_by_matrix(graph):
    matrix = create_graph_by_matrix_adjacency(graph)

    number_of_edges = 0
    number_of_nodes = len(matrix)

    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if matrix[i][j] == 1:
                number_of_edges += 1

    return (2 * number_of_edges) / (number_of_nodes * (number_of_nodes - 1))


def calc_degre_by_matrix(graph):
    matrix = create_graph_by_matrix_adjacency(graph)

    number_of_edges = 0
    number_of_nodes = len(matrix)

    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if matrix[i][j] == 1:
                number_of_edges += 1

    return number_of_edges


def calc_degre_by_list():
    global graph
    start_time = time.time()

    degree = len(graph.keys())

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return degree


def is_graph_complete_by_mtrix(graph):
    matrix = create_graph_by_matrix_adjacency(graph)

    number_of_edges = 0
    number_of_nodes = len(matrix)

    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if matrix[i][j] == 1:
                number_of_edges += 1

    return number_of_edges == number_of_nodes * (number_of_nodes - 1) / 2


def is_graph_complete_by_list():
    global graph
    start_time = time.time()

    number_of_edges = len(graph.keys())
    number_of_nodes = len(graph.values())

    is_complet = number_of_edges == number_of_nodes * (number_of_nodes - 1) / 2

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return is_complet


def search_node_by_matrix(graph, node):
    matrix = create_graph_by_matrix_adjacency(graph)

    if node - 1 >= len(matrix):
        return []

    seq = []
    for index, val in enumerate(matrix[node - 1]):
        if val == 1:
            seq.append(index + 1)

    return seq


def search_node_by_list(node):
    global graph
    start_time = time.time()

    nodes = graph.get(node, [])

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return nodes


def find_paths_by_matrix(graph, depart, arrivee):
    visited = []
    visited_nodes = set()

    def dfs(node, path):
        if node == arrivee:
            visited.append(path + [node])

        visited_nodes.add(node)

        voisins = search_node_by_matrix(graph, node)
        for voisin in voisins:
            if voisin not in visited_nodes:
                dfs(voisin, path + [node])

        visited_nodes.remove(node)

    dfs(depart, [])
    return visited


def find_paths_by_list(depart, arrivee):
    global graph
    start_time = time.time()

    visited = []
    visited_nodes = set()

    def dfs(node, path):
        if node == arrivee:
            visited.append(path + [node])
            return

        visited_nodes.add(node)

        for voisin in graph.get(node, []):
            if voisin not in visited_nodes:
                dfs(voisin, path + [node])

        visited_nodes.remove(node)

    dfs(depart, [])

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return visited


def find_shortest_path_by_matrix(graph, depart, arrivee):
    paths = find_paths_by_matrix(graph, depart, arrivee)

    return min(paths, key=lambda path: len(path))


def find_shortest_path_by_list(depart, arrivee):
    start_time = time.time()

    paths = find_paths_by_list(depart, arrivee)
    min_path = min(paths, key=lambda path: len(path))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return min_path


def show_graph():
    global graph
    G = nx.DiGraph()

    for node, edges in graph.items():
        for edge in edges:
            G.add_edge(node, edge)

    nx.draw_shell(G, with_labels=True)
    plt.show()


def const_graph1():
    global graph

    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 1), (3, 5), (3, 4), (4, 5)]

    g = Graph(nodes, edges)
    graph = create_graph_by_list_adjacency(g)


def const_graph2():
    global graph
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    edges = [
        (1, 16),
        (2, 10),
        (3, 4),
        (4, 15),
        (5, 1),
        (6, 4),
        (7, 19),
        (8, 4),
        (9, 8),
        (10, 13),
        (11, 14),
        (12, 15),
        (13, 2),
        (14, 19),
        (15, 20),
        (16, 14),
        (17, 3),
        (18, 1),
        (19, 9),
        (20, 11),
    ]
    g = Graph(nodes, edges)
    graph = create_graph_by_list_adjacency(g)


def const_graph5():
    global graph
    nodes = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
    ]

    edges = [
        (1, 3),
        (1, 41),
        (2, 25),
        (3, 5),
        (4, 28),
        (5, 43),
        (5, 38),
        (6, 37),
        (6, 33),
        (7, 13),
        (8, 47),
        (8, 48),
        (9, 32),
        (9, 40),
        (9, 23),
        (10, 23),
        (11, 47),
        (11, 43),
        (12, 5),
        (12, 27),
        (13, 28),
        (13, 41),
        (14, 48),
        (14, 40),
        (15, 40),
        (15, 37),
        (15, 29),
        (16, 4),
        (17, 39),
        (18, 6),
        (19, 29),
        (19, 32),
        (20, 22),
        (20, 25),
        (20, 32),
        (21, 28),
        (21, 38),
        (21, 20),
        (22, 30),
        (22, 37),
        (23, 23),
        (23, 15),
        (24, 43),
        (25, 36),
        (25, 43),
        (26, 46),
        (27, 29),
        (28, 19),
        (28, 27),
        (28, 45),
        (29, 32),
        (29, 4),
        (29, 40),
        (30, 29),
        (31, 7),
        (31, 25),
        (31, 47),
        (32, 13),
        (32, 9),
        (33, 22),
        (33, 14),
        (33, 3),
        (34, 50),
        (34, 2),
        (34, 43),
        (35, 35),
        (35, 1),
        (35, 3),
        (36, 20),
        (36, 22),
        (37, 43),
        (37, 44),
        (37, 35),
        (38, 15),
        (38, 44),
        (39, 50),
        (39, 6),
        (39, 49),
        (40, 38),
        (40, 34),
        (41, 1),
        (42, 42),
        (42, 32),
        (43, 26),
        (43, 6),
        (43, 18),
        (44, 37),
        (45, 16),
        (46, 21),
        (47, 6),
        (47, 35),
        (47, 34),
        (48, 14),
        (49, 5),
        (49, 48),
        (49, 37),
        (50, 10),
    ]
    g = Graph(nodes, edges)
    graph = create_graph_by_list_adjacency(g)


def main():
    global graph

    print(
        """
1: Construction d’un graphe orienté (/non orienté)
2: Affichage de graphe
3: Calculer la densité de graphe
4: Calculer le degree de graphe
5: Verifie si le graphe est complet
6: Recherche d’un nœud a dans le graphe (afficher le nœud et ses liens)
7: Trouver tous les chemins entre 2 points
8: Trouver le plus court chemins entre 2 points
"""
    )

    while True:
        choice = input("> ")

        match choice:
            case "1":
                const_graph2()
            case "2":
                show_graph()
            case "3":
                densité = calc_density_by_list()
                print(f"La densité est {densité}")
            case "4":
                degree = calc_degre_by_list()
                print(f"La degree est {degree}")
            case "5":
                is_complect = is_graph_complete_by_list()
                print(f"Le graphe est complet: {is_complect}")
            case "6":
                node = input(">> entre le noeud: ")
                links = search_node_by_list(int(node))
                print(f"Le noued et ses lien: {links}")
            case "7":
                nodes = input("entre le noeud de depart et fin: ").split(" ")
                links = find_paths_by_list(int(nodes[0]), int(nodes[1]))
                print(f"tous les chemins entre {nodes[0]} et {nodes[1]} sont: {links}")
            case "8":
                nodes = input("entre le noeud de depart et fin: ").split(" ")
                links = find_shortest_path_by_list(int(nodes[0]), int(nodes[1]))
                print(
                    f"Le plus court chemin entre {nodes[0]} et {nodes[1]} sont: {links}"
                )

            case "quit()":
                break


if __name__ == "__main__":
    main()
