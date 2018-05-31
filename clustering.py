import math
import os
import pickle
import sys

wordlist = []
"""
WordEmbedding.txt에서 읽어들일 단어의 리스트.
"""


class Tree(object):
    """
    Complete Link Cluster를 구현하기 위한 트리 자료구조
    """
    def __init__(self, rep, sim, word="", children=None):
        """
        트리 노드의 생성자.
        :param rep: 이 노드를 대표하는 단어의 index. 작은 쪽이 항상 대표값을 가지게끔 구현
        :param sim: 이 노드 아래의 단어는 sim값 만큼의 유사도를 가짐.
        :param word: Leaf노드만 이 값을 가진다. 해당 노드가 표현하는 단어를 의미
        :param children: 이 노드의 자식노드들.
        """
        self.word = word
        self.represent = rep
        self.similarity = sim
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        """
        트리 노드 정보를 출력해주는 함수
        :return: "단어: 인덱스" 값을 출력
        """
        return self.word+": "+str(self.represent)

    def add_child(self, node):
        """
        자식 노드를 추가하는 함수
        :param node: 자식 노드로 들어갈 노드
        """
        assert isinstance(node, Tree)
        self.children.append(node)

    def find_child(self, rep):
        """
        자식 노드를 찾아주는 함수
        :param rep: 찾을 노드의 대표값
        :return: 찾은 노드를 반환. None이 반환 될 수 있음.
        """
        for i in range(len(self.children)):
            if self.children[i].represent == rep:
                return self.children[i]
        return None

    def replace_child(self, node, new):
        """
        지정된 노드와 인자로 준 노드를 바꾸는 함수.
        :param node: 바꾸려고 하는 대상 노드
        :param new: 대상 노드를 이 노드로 변경한다.
        """
        assert isinstance(node, Tree)
        for i in range(len(self.children)):
            if self.children[i] == node:
                self.children[i] = new
                break

    def delete_child(self, node):
        """
        자식 노드중 지정한 노드를 삭제하는 함수
        :param node: 이 자식 노드를 삭제한다.
        """
        assert isinstance(node, Tree)
        for i in range(len(self.children)):
            if self.children[i] == node:
                del self.children[i]
                break


def dot_product(a, b):
    """
    두 벡터간의 내적을 계산하는 함수.
    인자로 주어지는 두 벡터는 차원이 같아야 한다.
    :param a: 1차원 벡터
    :param b: 2차원 벡터
    :return: 두 벡터의 내적값
    """
    _sum = 0
    if len(a) != len(b):
        raise ArithmeticError("Dimension of two vector isn't match.")
    else:
        for i in range(0, len(a)):
            _sum += a[i] * b[i]
    return _sum


def euclidean_similarity(a, b):
    """
    두 벡터간의 유클리드 거리를 계산하는 함수.
    인자로 주어지는 두 벡터는 차원이 같아야 한다.
    :param a: 1차원 벡터
    :param b: 1차원 벡터
    :return: 두 벡터간의 유클리드 거리
    """
    _sum = 0
    if len(a) != len(b):
        raise ArithmeticError("Dimension of two vector isn't match.")
    else:
        for i in range(0, len(a)):
            _sum += (a[i] - b[i])**2
    return math.sqrt(_sum)


def cos_similarity(a, b):
    """
    두 벡터간의 코사인 유사도를 계산하는 함수.
    인자로 주어지는 두 벡터는 차원이 같아야 한다.
    :param a: 1차원 벡터
    :param b: 2차원 벡터
    :return: 두 벡터의 코사인 값.
    """
    return dot_product(a, b)/(norm(a) * norm(b))


def norm(a):
    """
    :param a: 1차원 벡터
    :return: 벡터의 norm값
    """
    _sum = 0
    for i in range(0, len(a)):
        _sum += a[i]**2
    return math.sqrt(_sum)


def complete_link_clustering(node_root, node_amount, sim_matrix):
    """
    Complete Link Clustering에 필요한 트리 생성을 수행하는 함수.
    :param node_root: Clustering Tree의 Root 노드.
    :param node_amount: 노드의 총 갯수.
    :param sim_matrix: 클러스터링을 위해 필요한 유사도 행렬
    :return:
    """
    while True:
        """
        sim_matrix에서 유사도가 가장 큰 두개의 단어를 찾은 뒤,
        wordlist에서의 단어 인덱스를 node1, node2에 배치한다.
        """
        max_sim = -math.inf
        node1 = -1
        node2 = -1
        for i in range(0, node_amount):
            for j in range(i + 1, node_amount):
                # -inf값은 빈 칸이라는 의미로 사용 되었다.
                if sim_matrix[i][j] == -math.inf:
                    continue
                if max_sim < sim_matrix[i][j]:
                    max_sim = sim_matrix[i][j]
                    node1 = i
                    node2 = j
        """
        만약 찾은 단어가 없다면 끝난 것으로 간주.
        """
        if node1 == -1 and node2 == -1:
            break

        """
        루트 노드에서 각자의 단어에 해당하는 노드가 있는지 찾아 본다.
        """
        cur1 = node_root.find_child(node1)
        cur2 = node_root.find_child(node2)
        new = Tree(node1, max_sim)

        if cur1 is None and cur2 is None:
            """
            둘다 없는 경우, 이 두개의 단어로 트리를 생성해서 루트 노드에 추가한다.
            """
            new.add_child(Tree(node1, 0, wordlist[node1][0]))
            new.add_child(Tree(node2, 0, wordlist[node2][0]))
            node_root.add_child(new)
        elif cur1 is not None and cur2 is None:
            """
            둘 중 하나만 있는 경우, 먼저 생성된 노드가 유사도가 큰 노드이므로,
            새로 생성할 노드의 자식으로 먼저 생성된 노드를 추가한 뒤, 
            새로 생성한 노드를 먼저 생성된 노드와 바꾼다. 
            """
            new.add_child(cur1)
            new.add_child(Tree(node2, 0, wordlist[node2][0]))
            node_root.replace_child(cur1, new)
        elif cur1 is None and cur2 is not None:
            """
            위와 마찬가지.
            """
            new.add_child(Tree(node1, 0, wordlist[node1][0]))
            new.add_child(cur2)
            node_root.replace_child(cur2, new)
        else:
            """
            추가 시킬 노드가 둘다 존재하는 경우,
            두 노드를 붙여서 루트에 등록한다.
            cur1에 해당하는 노드와 새로 생성한 노드를 바꾸고, cur2에 해당하는 노드는 삭제.
            """
            new.add_child(cur1)
            new.add_child(cur2)
            node_root.replace_child(cur1, new)
            node_root.delete_child(cur2)

        """
        노드 추가가 완료 되면, 유사도 행렬에서 두 노드에 해당되는 행과 열을 합친다.
        행과 열을 삭제하는 대신, 삭제된 칸은 -inf로 두는 것으로 하였다.
        """
        for i in range(0, node_amount - 1):
            if i == node1:
                continue
            sim_matrix[node1][i] = min(sim_matrix[node1][i], sim_matrix[node2][i])
            sim_matrix[i][node1] = min(sim_matrix[node1][i], sim_matrix[node2][i])

        for i in range(0, node_amount - 1):
            sim_matrix[node2][i] = -math.inf
            sim_matrix[i][node2] = -math.inf


cur_cluster = []
def clustering(node, threshold, cluster_list):
    """
    complete_link_clustering 함수를 통해 생성된 트리를 가지고 실제 클러스터링을 하는 함수.
    :param node: complete_link_clustering 함수에서 생성 완료된 트리의 Root 노드.
    :param threshold: 유사도 threshold 값
    :param cluster_list: 클러스터의 List. list의 각 요소들 마다 단어들의 list가 들어가 있다. 파라미터이자 반환값이기도 하다.
    ex) [[secret, banned],[happy, delighted],...]
    """
    assert isinstance(node, Tree)
    assert isinstance(cluster_list, list)
    global cur_cluster
    cur = node
    if len(cur.children) == 0:
        # 현재 클러스터에 포함되는 단어들을 저장. 해당 변수는 전역변수.
        cur_cluster.append(cur.word)
        return

    for i in range(len(cur.children)):
        # 트리는 DFS로 순회한다.
        clustering(cur.children[i], threshold, cluster_list)
        if cur.similarity < threshold:
            # 현재 노드의 유사도가 threshold 값보다 작다면, 클러스터를 분리한다.
            if len(cur_cluster) > 0:
                cluster_list.append(cur_cluster)
            cur_cluster = []



def entropy(cluster_list, word_topic_list, word_total_amount):
    """
    전체 클러스터의 엔트로피를 계산하는 함수.
    각 클러스터에 담긴 단어를 해당 토픽으로 치환 한 뒤에 엔트로피를 계산한다.
    :param cluster_list: 전체 클러스터
    :param word_topic_list: WordTopic에서 가져온 토픽
    :param word_total_amount: 단어의 총 갯수
    :return: 계산된 전체 클러스터의 엔트로피
    """
    def find_in_topic(word):
        """
        :param word: 토픽 리스트에서 찾고자 하는 단어
        :return: 해당 단어가 어떤 토픽에 들어 있었는가. 여기서는 topic list의 index값을 반환.
        만약 해당 단어가 토픽 리스트에 없었다면 -1을 반환.
        """
        for i in range(len(word_topic_list)):
            for _word in word_topic_list[i]:
                if word.lower() == _word.lower():
                    return i
        return -1

    cluster_topic_list = []
    for item in cluster_list:
        cur_cluster_topic_num_check = [0] * len(word_topic_list)
        for word in item:
            """
            item은 클러스터 하나를 의미.
            """
            cur_topic = find_in_topic(word)
            assert (cur_topic != -1)
            cur_cluster_topic_num_check[cur_topic] += 1

        cluster_topic_list.append(cur_cluster_topic_num_check)
        """
        [0,2,0,3,0,0,1,0] 이런 식으로 한 클러스터에 토픽 정보가 입력 된다.
        2번 토픽 단어가 2개, 4번 토픽 단어가 3개, 7번 토픽 단어가 1개, 해당 클러스터에 단어가 6개 있음을 나타낸다.
        """

    entropy_sum = 0
    """
    엔트로피를 계산 한다. 각각의 클러스터에 대해 엔트로피를 계산하고, 
    총 엔트로피는 각 클러스터에 대해 weighted sum으로 계산 된다.
    """
    for item in cluster_topic_list:
        cluster_entropy_sum = 0
        cluster_word_amount = 0
        for i in item:
            cluster_word_amount += i
        for i in item:
            if i == 0:
                continue
            cluster_entropy_sum -= math.log(i/cluster_word_amount) * i/cluster_word_amount
        entropy_sum += cluster_entropy_sum * cluster_word_amount / word_total_amount

    return entropy_sum


def main():

    usepickle = True
    if len(sys.argv) < 3:
        print("usage: clustering <threshold> <wordEmbeddingFilePath> --recalc \n"
              "--recalc : 유사도 행렬 계산과 클러스터 트리 생성을 재 수행 합니다. \n"
              "이 옵션이 없을 경우 이전에 수행한 계산 결과를 다시 가져 옵니다.")
        return

    if len(sys.argv) > 3:
        if sys.argv[3] == "--recalc":
            usepickle = False
        else:
            print("알려지지 않은 옵션입니다. "+sys.argv[2])
            return

    try:
        with open(sys.argv[2], 'r') as f:
            while True:
                word = f.readline()
                line = f.readline()

                if not line:
                    break
                v = line.split(',')
                for i in range(0, len(v)):
                    v[i] = float(v[i])
                wordlist.append([word, v])
    except OSError:
        print("wordEmbedding File 이 해당 경로에 존재하지 않습니다.")
        return

    print("Data Loaded - # of words : " + str(len(wordlist)))

    Cosine_similarity_matrix = [[-math.inf]*len(wordlist) for i in range(len(wordlist))]
    Euclidean_similarity_matrix = [[-math.inf]*len(wordlist) for i in range(len(wordlist))]

    Cosine_tree_root = Tree(-1, -math.inf)
    Euclidean_tree_root = Tree(-1, -math.inf)

    """
    pickle을 사용 할 것인지 말 것인지에 관련된 구간.
    이전에 계산해 놓은 유사도 행렬과 클러스터 트리가 있다면 피클을 사용해 저장한 뒤, 그것을 불러들여 사용 할 수 있다.
    """
    cos_sim_mat_loaded = False
    euc_sim_mat_loaded = False
    cos_cluster_loaded = False
    euc_cluster_loaded = False
    if usepickle:
        if os.path.isfile("pickle_cos_sim_mat"):
            with open("pickle_cos_sim_mat", 'rb') as f:
                Cosine_similarity_matrix = pickle.load(f)
                cos_sim_mat_loaded = True

        if os.path.isfile("pickle_euc_sim_mat"):
            with open("pickle_euc_sim_mat", 'rb') as f:
                Euclidean_similarity_matrix = pickle.load(f)
                euc_sim_mat_loaded = True

        if os.path.isfile("pickle_cos_cluster"):
            with open("pickle_cos_cluster", 'rb') as f:
                Cosine_tree_root = pickle.load(f)
                cos_cluster_loaded = True

        if os.path.isfile("pickle_euc_cluster"):
            with open("pickle_euc_cluster", 'rb') as f:
                Euclidean_tree_root = pickle.load(f)
                euc_cluster_loaded = True

    """
    유사도 행렬을 구성한다. 만약 pickle로 이미 로드가 되었다면 건너 뛴다.
    """
    if not cos_sim_mat_loaded:
        for i in range(0, len(wordlist)):
            for j in range(i+1, len(wordlist)):
                cos_sim = cos_similarity(wordlist[i][1], wordlist[j][1])
                Cosine_similarity_matrix[i][j] = cos_sim
                Cosine_similarity_matrix[j][i] = cos_sim
        with open("pickle_cos_sim_mat", "wb") as f:
            pickle.dump(Cosine_similarity_matrix, f)
        print("Cosine Similarity Matrix Calculated")

    if not euc_sim_mat_loaded:
        euc_sim_max = -math.inf
        for i in range(0, len(wordlist)):
            for j in range(i+1, len(wordlist)):
                euc_sim = euclidean_similarity(wordlist[i][1], wordlist[j][1])
                if euc_sim_max < euc_sim:
                    euc_sim_max = euc_sim
                Euclidean_similarity_matrix[i][j] = euc_sim
                Euclidean_similarity_matrix[j][i] = euc_sim
        for i in range(0, len(wordlist)):
            for j in range(i+1, len(wordlist)):
                if Euclidean_similarity_matrix[i][j] == -math.inf:
                    continue
                Euclidean_similarity_matrix[i][j] = (euc_sim_max - Euclidean_similarity_matrix[i][j]) / euc_sim_max
                Euclidean_similarity_matrix[j][i] = (euc_sim_max - Euclidean_similarity_matrix[j][i]) / euc_sim_max
        with open("pickle_euc_sim_mat", "wb") as f:
            pickle.dump(Euclidean_similarity_matrix, f)
        print("Euclidean Similarity Matrix Calculated")

    """
    complete_link_clustering을 통해 클러스터 트리를 구성한다. 이미 pickle을 사용해 로드 한 경우 건너 뛴다.
    """
    if not cos_cluster_loaded:
        complete_link_clustering(Cosine_tree_root, len(wordlist), Cosine_similarity_matrix)
        with open("pickle_cos_cluster", "wb") as f:
            pickle.dump(Cosine_tree_root, f)
        print("Cosine Complete Link Clustering Complete")

    if not euc_cluster_loaded:
        complete_link_clustering(Euclidean_tree_root, len(wordlist), Euclidean_similarity_matrix)
        with open("pickle_euc_cluster", "wb") as f:
            pickle.dump(Euclidean_tree_root, f)
        print("Euclidean Complete Link Clustering Complete")

    """
    클러스터 트리와 주어진 threshold를 이용하여 클러스터링을 수행한다.
    """
    cos_cluster_list = []
    clustering(Cosine_tree_root, float(sys.argv[1]), cos_cluster_list)
    with open("WordClustering_cos_sim.txt", 'wt') as f:
        for wordinfo in wordlist:
            for item in cos_cluster_list:
                if wordinfo[0] in item:
                    f.write(wordinfo[0]
                            + str(wordinfo[1]) + "\n"
                            + str(cos_cluster_list.index(item)) + '\n')
                    break
    print("# of cluster - cosine similarity : " + str(len(cos_cluster_list)))

    euc_cluster_list = []
    clustering(Euclidean_tree_root, float(sys.argv[1]), euc_cluster_list)
    with open("WordClustering_euc_sim.txt", 'wt') as f:
        for wordinfo in wordlist:
            for item in euc_cluster_list:
                if wordinfo[0] in item:
                    f.write(wordinfo[0]
                            + str(wordinfo[1]) + "\n"
                            + str(euc_cluster_list.index(item)) + '\n')
                    break
    print("# of cluster - euclidean similarity : " + str(len(euc_cluster_list)))
    print("clustering complete")

    """
    토픽 리스트를 가져와 각각의 유사도 계산법에 대해 엔트로피를 계산한다.
    """
    topic_list = []
    with open("WordTopic.txt", 'r') as f:
        while True:
            topic = f.readline()
            cur_topic = []
            while True:
                word = f.readline()
                if word == '\n' or not word:
                    break
                cur_topic.append(word)
            if not topic:
                break
            topic_list.append(cur_topic)

    print("Topic Data Loaded")

    print("cos entropy : " + str(entropy(cos_cluster_list, topic_list, len(wordlist))))
    print("euc entropy : " + str(entropy(euc_cluster_list, topic_list, len(wordlist))))


if __name__ == "__main__":
    main()
