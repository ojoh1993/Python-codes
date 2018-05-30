import math
import os
import pickle
import sys


def euclidean_similarity(a, b):
    _sum = 0
    if len(a) != len(b):
        raise ArithmeticError("length of two vector isn't match.")
    else:
        for i in range(0, len(a)):
            _sum += (a[i] - b[i])**2
    return math.sqrt(_sum)


def cos_similarity(a, b):
    return dot_product(a, b)/(norm(a) * norm(b))


def norm(a):
    _sum = 0
    for i in range(0, len(a)):
        _sum += a[i]**2
    return math.sqrt(_sum)


def dot_product(a, b):
    _sum = 0
    if len(a) != len(b):
        raise ArithmeticError("length of two vector isn't match.")
    else:
        for i in range(0, len(a)):
            _sum += a[i] * b[i]
    return _sum


def entropy():

    return 0


class Tree(object):
    def __init__(self, rep, sim, word="", children=None):
        self.word = word
        self.represent = rep
        self.similarity = sim
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.word+": "+str(self.represent)

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

    def find_child(self, rep):
        for i in range(len(self.children)):
            if self.children[i].represent == rep:
                return self.children[i]

    def replace_child(self, node, new):
        assert isinstance(node, Tree)
        for i in range(len(self.children)):
            if self.children[i] == node:
                self.children[i] = new
                return

    def delete_child(self, node):
        assert isinstance(node, Tree)
        for i in range(len(self.children)):
            if self.children[i] == node:
                del self.children[i]
                return


def complete_link_clustering(node_root, node_amount, sim_matrix):
    linkage_count = node_amount - 1
    while linkage_count > 1:
        max_sim = -math.inf
        node1 = -1
        node2 = -1

        for i in range(0, node_amount):
            for j in range(i + 1, node_amount):
                if sim_matrix[i][j] == -math.inf:
                    continue
                if max_sim < sim_matrix[i][j]:
                    max_sim = sim_matrix[i][j]
                    node1 = i
                    node2 = j

        if node1 == -1 and node2 == -1:
            break

        cur1 = node_root.find_child(node1)
        cur2 = node_root.find_child(node2)
        new = Tree(node1, max_sim)

        if cur1 is None and cur2 is None:
            new.add_child(Tree(node1, 0, wordList[node1][0]))
            new.add_child(Tree(node2, 0, wordList[node2][0]))
            node_root.add_child(new)
        elif cur1 is not None and cur2 is None:
            new.add_child(cur1)
            new.add_child(Tree(node2, 0, wordList[node2][0]))
            node_root.replace_child(cur1, new)
        elif cur1 is None and cur2 is not None:
            new.add_child(Tree(node1, 0, wordList[node1][0]))
            new.add_child(cur2)
            node_root.replace_child(cur2, new)
        else:
            new.add_child(cur1)
            new.add_child(cur2)
            node_root.replace_child(cur1, new)
            node_root.delete_child(cur2)

        for i in range(0, node_amount - 1):
            if i == node1:
                continue
            sim_matrix[node1][i] = min(sim_matrix[node1][i], sim_matrix[node2][i])
            sim_matrix[i][node1] = min(sim_matrix[node1][i], sim_matrix[node2][i])

        for i in range(0, node_amount - 1):
            sim_matrix[node2][i] = -math.inf
            sim_matrix[i][node2] = -math.inf

        linkage_count -= 1


cur_cluster = []
def clustering(node, threshold, cluster_list):
    assert isinstance(node, Tree)
    assert isinstance(cluster_list, list)
    global cur_cluster
    cur = node
    if len(cur.children) == 0:
        cur_cluster.append(cur.word)
        return

    for i in range(len(cur.children)):
        clustering(cur.children[i], threshold, cluster_list)
        if cur.similarity < threshold:
            if len(cur_cluster) > 0:
                cluster_list.append(cur_cluster)
            cur_cluster = []


def main():
    print(sys.argv[1:])

    wordlist = []

    with open("WordEmbedding.txt", 'r') as f:
        while True:
            word = f.readline()
            line = f.readline()

            if not line:
                break
            v = line.split(',')
            for i in range(0, len(v)):
                v[i] = float(v[i])
            wordlist.append([word, v])

    print("Data Loaded")

    Cosine_similarity_matrix = [[-math.inf]*len(wordlist) for i in range(len(wordlist))]
    Euclidean_similarity_matrix = [[-math.inf]*len(wordlist) for i in range(len(wordlist))]

    Cosine_tree_root = Tree(-1, -2)
    Euclidean_tree_root = Tree(-1, -2)

    cos_sim_mat_loaded = False
    euc_sim_mat_loaded = False
    cos_cluster_loaded = False
    euc_cluster_loaded = False

    if os.path.isfile("pickle_cos_sim_mat.txt"):
        with open("pickle_cos_sim_mat.txt", 'rb') as f:
            Cosine_similarity_matrix = pickle.load(f)
            cos_sim_mat_loaded = True

    if os.path.isfile("pickle_euc_sim_mat.txt"):
        with open("pickle_euc_sim_mat.txt", 'rb') as f:
            Euclidean_similarity_matrix = pickle.load(f)
            euc_sim_mat_loaded = True

    if os.path.isfile("pickle_cos_cluster.txt"):
        with open("pickle_cos_cluster.txt", 'rb') as f:
            Cosine_tree_root = pickle.load(f)
            cos_cluster_loaded = True

    if os.path.isfile("pickle_euc_cluster.txt"):
        with open("pickle_euc_cluster.txt", 'rb') as f:
            Euclidean_tree_root = pickle.load(f)
            euc_cluster_loaded = True

    if not cos_sim_mat_loaded:
        for i in range(0, len(wordlist)-1):
            for j in range(i+1, len(wordlist)-1):
                cos_sim = cos_similarity(wordlist[i][1], wordlist[j][1])
                Cosine_similarity_matrix[i][j] = cos_sim
                Cosine_similarity_matrix[j][i] = cos_sim
        with open("pickle_cos_sim_mat.txt", "wb") as f:
            pickle.dump(Cosine_similarity_matrix, f)
        print("Cosine Similarity Matrix Calculated")

    if not euc_sim_mat_loaded:
        for i in range(0, len(wordlist)-1):
            for j in range(i+1, len(wordlist)-1):
                euc_sim = euclidean_similarity(wordlist[i][1], wordlist[j][1])
                Euclidean_similarity_matrix[i][j] = euc_sim
                Euclidean_similarity_matrix[j][i] = euc_sim
        with open("pickle_euc_sim_mat.txt", "wb") as f:
            pickle.dump(Euclidean_similarity_matrix, f)
        print("Euclidean Similarity Matrix Calculated")

    if not cos_cluster_loaded:
        complete_link_clustering(Cosine_tree_root, len(wordlist), Cosine_similarity_matrix)
        with open("pickle_cos_cluster.txt", "wb") as f:
            pickle.dump(Cosine_tree_root, f)
        print("Cosine Complete Link Clustering Complete")

    if not euc_cluster_loaded:
        complete_link_clustering(Euclidean_tree_root, len(wordlist), Euclidean_similarity_matrix)
        with open("pickle_euc_cluster.txt", "wb") as f:
            pickle.dump(Euclidean_tree_root, f)
        print("Euclidean Complete Link Clustering Complete")

    cluster_list = []
    clustering(Cosine_tree_root, 0.2, cluster_list)

    print("clustering complete")


if __name__ == __main__
    main()
