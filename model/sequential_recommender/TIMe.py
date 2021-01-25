# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from util import timer
from util import DataIterator
from util.tool import csr_to_user_dict_bytime
from model.AbstractRecommender import SeqAbstractRecommender
from itertools import combinations
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import estimate_rank
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
from treelib import Tree
from tqdm import tqdm

def facesiter(simplex):
    for i in range(len(simplex)):
        yield simplex[:i]+simplex[(i+1):]


def flattening_simplex(simplices):
    for simplex in simplices:
        for point in simplex:
            yield point

def get_allpoints(simplices):
    return set(flattening_simplex(simplices))

def drawComplex(data, sc, axes=[-6, 8, -6, 6]):
    plt.clf()
    plt.axis(axes)  # axes = [x1, x2, y1, y2]
    # add points
    plt.scatter(data[:, 0], data[:, 1])
    # add labels
    for i, txt in enumerate(data):
        plt.annotate(i, (data[i][0] + 0.05, data[i][1]))

    # add lines for edges
    for edge in sc.n_faces(1):
        # print(edge)
        pt1, pt2 = [data[pt] for pt in [n for n in edge]]
        # plt.gca().add_line(plt.Line2D(pt1,pt2))
        line = plt.Polygon([pt1, pt2], closed=None, fill=None, edgecolor='r')
        plt.gca().add_line(line)

    # add triangles
    for triangle in sc.n_faces(2):
        pt1, pt2, pt3 = [data[pt] for pt in [n for n in triangle]]
        line = plt.Polygon([pt1, pt2, pt3], closed=False,
                           color="blue", alpha=0.3, fill=True, edgecolor=None)
        plt.gca().add_line(line)
    plt.show()
    
def faces(simplices):
    faceset = set()
    for simplex in simplices:
        numnodes = len(simplex)
        for r in range(numnodes, 0, -1):
            for face in combinations(simplex, r):
                faceset.add(tuple(sorted(face)))
    return faceset

class SimplicialComplex(object):
    def __init__(self, simplices=[]):
        self.import_simplices(simplices=simplices)

    def import_simplices(self, simplices=[]):
        self.simplices = list(map(lambda simplex: tuple(sorted(simplex)), simplices))
        self.face_set = faces(self.simplices)
        
    def import_simplical_complexes(self, simplical_complexes=[]):
        self.simplical_complexes = list(map(lambda simplical_complex: tuple(sorted(simplical_complex)), simplical_complexes))

    def n_faces(self, n):
        return list(filter(lambda face: len(face) == n+1, self.face_set))

    def boundary_operator(self, i):
        source_simplices = self.n_faces(i)
        target_simplices = self.n_faces(i-1)

        if len(target_simplices) == 0:
            S = dok_matrix((1, len(source_simplices)), dtype=np.float64)
            S[0, 0:len(source_simplices)] = 1
        else:
            source_simplices_dict = {source_simplices[j]:
                                     j for j in range(len(source_simplices))}
            target_simplices_dict = {target_simplices[i]:
                                     i for i in range(len(target_simplices))}

            S = dok_matrix((len(target_simplices),
                            len(source_simplices)),
                           dtype=np.float64)
            for source_simplex in source_simplices:
                for a in range(len(source_simplex)):
                    target_simplex = source_simplex[:a]+source_simplex[(a+1):]
                    i = target_simplices_dict[target_simplex]
                    j = source_simplices_dict[source_simplex]
                    S[i, j] = -1 if a % 2 == 1 else 1   # S[i, j] = (-1)**a
        return S

    def betti_number(self, i, eps=None):
        boundop_i = self.boundary_operator(i)
        boundop_ip1 = self.boundary_operator(i+1)

        if i == 0:
            boundop_i_rank = 0
        else:
            if eps is None:
                try:
                    boundop_i_rank = np.linalg.matrix_rank(boundop_i.toarray())
                except (np.linalg.LinAlgError, ValueError):
                    boundop_i_rank = boundop_i.shape[1]
            else:
                boundop_i_rank = estimate_rank(
                                aslinearoperator(boundop_i),
                                eps)

        if eps is None:
            try:
                boundop_ip1_rank = np.linalg.matrix_rank(boundop_ip1.toarray())
            except (np.linalg.LinAlgError, ValueError):
                boundop_ip1_rank = boundop_ip1.shape[1]
        else:
            boundop_ip1_rank = estimate_rank(
                                aslinearoperator(boundop_ip1),
                                eps)

        return ((boundop_i.shape[1]-boundop_i_rank)-boundop_ip1_rank)

    def euler_characteristics(self):
        max_n = max(map(len, self.simplices))
        return sum([(-1 if a % 2 == 1 else 1)*self.betti_number(a) for a in range(max_n)])

class VietorisRipsComplex(SimplicialComplex):
    def __init__(self,
                 points,
                 epsilon,
                 labels=None,
                 distfcn=distance.euclidean):
        self.pts = points
        self.labels = (range(len(self.pts))
                       if labels is None or len(labels) != len(self.pts)
                       else labels)
        self.epsilon = epsilon
        self.distfcn = distfcn
        self.network = self.construct_network(self.pts,
                                              self.labels,
                                              self.epsilon,
                                              self.distfcn)
        # self.import_simplices(map(tuple, nx.find_cliques(self.network)))
        self.import_simplical_complexes(map(tuple, nx.connected_components(self.network)))

    def construct_network(self, points, labels, epsilon, distfcn):
        g = nx.Graph()
        g.add_nodes_from(labels)
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                if i<j:
                    dist = distfcn(points[i], points[j])
                    if dist < epsilon:
                        g.add_edge(labels[i], labels[j])
        return g

def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis,:],
                           d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        
    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # attention 乘上value
    output = tf.matmul(attention_weights, v) # （.., seq_len_v, depth）
    
    return output, attention_weights

class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)
        
        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))
        
        # 全连接重塑
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
class TreeAggregationLayer(tf.keras.layers.Layer):
    def __init__(self, agg):
        super(TreeAggregationLayer, self).__init__()
        self.agg = agg
        
    def seq2tree(self, tree, seq, x):
        idx = 0
        for nid in tree.expand_tree(mode=Tree.WIDTH, sorting=False):
            if idx < len(seq):
                tree[nid].data[x] = seq[idx]
                idx += 1
    
        return tree
    
    def tree2seq(self, tree, x):
        seq = []
        for nid in tree.expand_tree(mode=Tree.WIDTH, sorting=False):
            # node embedding
            seq.append(tree[nid].data[x])
        return seq
    
    def calc_node_embedding(self, tree, node):
        if node.data[1] is None:
            children_embedding = []
            children_nodes = tree.children(node.identifier)
            for children in children_nodes:
                newnode = tree.get_node(children.identifier)
                embedding = self.calc_node_embedding(tree, newnode)
                children_embedding.append(embedding)
            children_embedding = tf.stack(children_embedding, axis=0)
            node.data[1] = self.get_node_embedding(children_embedding)
            if node.data[3] is None:
                node_items_pos = np.array(node.data[2], dtype=np.float32)
                node.data[3] = np.mean(node_items_pos)
            return node.data[1]
        else:
            return node.data[1]
        
    def get_node_embedding(self, input_embedding):
        if self.agg == 'sum':
            output = tf.nn.tanh(tf.reduce_sum(input_embedding, axis=0)).numpy()
        elif self.agg == 'max':
            output = tf.reduce_max(input_embedding, axis=0).numpy()
        elif self.agg == 'mean':
            output = tf.reduce_max(input_embedding, axis=0).numpy()
        else:
            print("unknown agg: %s" % self.agg)
            output = None
            
        return output
        
    def tree_init(self, tree):
        for nid in tree.expand_tree(mode=Tree.WIDTH, sorting=False):
            node = tree[nid]
            if not node.is_leaf():
                node.data[1] = None
                
        return tree
    
    def tree_aggregation(self, tree):
        tree = self.tree_init(tree)
        root_node = tree.get_node(tree.root)
        self.calc_node_embedding(tree, root_node)
        
        return tree
        
    def call(self, seqs, trees):
        seq_len = seqs.shape[1]
        max_nodes = seq_len - 1
        num_trees = len(trees)
        output = []
        for idx in range(num_trees):
            tree = trees[idx]
            seq = seqs[idx].numpy()
            num_tree_nodes = tree.size()
            if num_tree_nodes < max_nodes:
                pad_length = max_nodes - num_tree_nodes
                # extract real value
                subseq = seq[pad_length:max_nodes]
                tree = self.seq2tree(tree, subseq, 1)
                tree = self.tree_aggregation(tree)
                aggsubseq = self.tree2seq(tree, 1)
                seq[pad_length:max_nodes] = aggsubseq
            else:
                # extract real value
                subseq = seq[:max_nodes]
                tree = self.seq2tree(tree, subseq, 1)
                tree = self.tree_aggregation(tree)
                aggsubseq = self.tree2seq(tree, 1)
                seq[:max_nodes] = aggsubseq[:max_nodes]
            output.append(tf.convert_to_tensor(seq))
        output = tf.stack(output, axis=0)
        return output

class TreeEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, agg, dropout_rate=0.1):
        super(TreeEncoderLayer, self).__init__()
        
        self.mha = MutilHeadAttention(d_model, n_heads)
        self.tab = TreeAggregationLayer(agg)
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, mask):
        # 多头注意力网络
        att_output, att_weights = self.mha(inputs[0], inputs[0], inputs[0], mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs[0] + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        tab_output = self.tab(out1, inputs[1])
        tab_output = self.dropout2(tab_output, training=training)
        out2 = self.layernorm2(out1 + tab_output)  # (batch_size, input_seq_len, d_model)
        # user to nodes attention L1 loss
        loss = tf.reduce_sum(tf.abs(att_weights[:,:,-1,:]))
        inputs[0] = out2
        
        return inputs,loss,att_weights[:,:,-1,:]
    
class TreeTransformer(tf.keras.layers.Layer):
    def __init__(self, num_blocks, embedding_size, num_heads, agg, drop_rate=0.1):
        super(TreeTransformer, self).__init__()
        
        self.num_blocks = num_blocks
        self.encode_layer = [TreeEncoderLayer(embedding_size, num_heads, agg, drop_rate)
                            for _ in range(num_blocks)]
        
    def call(self, inputs, training, mark):
        l1_loss = []
        # batch_size * seq_len * embedding_size
        x = inputs
        for i in range(self.num_blocks):
            x,loss,weights = self.encode_layer[i](x, training, mark)
            l1_loss.append(loss)
        total_l1_loss = tf.reduce_sum(tf.stack(l1_loss))
        
        return x[0],total_l1_loss,weights
    
class TIMeModel(keras.Model):
    def __init__(self, model):
        super(TIMeModel, self).__init__()
        self.num_items = model.num_items
        self.max_nodes = model.max_nodes
        self.embedding_size = model.embedding_size
        self.agg = model.agg
        # embedding layer
        self.user_embedding_layer = keras.layers.Embedding(model.num_users, model.embedding_size)
        self.item_embedding_layer = keras.layers.Embedding(model.num_items, model.embedding_size)
        self.pos_embedding = positional_encoding(model.num_items, model.embedding_size)
        # transformer layer
        self.transformer_layer = TreeTransformer(model.num_blocks,
                                            model.embedding_size,
                                            model.num_heads,
                                            model.agg,
                                            model.drop_rate)
        self.max_complex_tree_nodes = 0
        
    def create_padding_mark(self, complex_tree_input):
        sequence_padding_mark = []
        num_complex_trees = len(complex_tree_input)
        for idx in range(num_complex_trees):
            complex_tree = complex_tree_input[idx]
            num_complex_tree_nodes = complex_tree.size()
            padding_mark = np.zeros(self.max_nodes+1, dtype=np.float32)
            if num_complex_tree_nodes < self.max_nodes:
                pad_length = self.max_nodes - num_complex_tree_nodes
                padding_mark[:pad_length] = 1.0
                # remove item to user attention
                padding_mark[-1] = 1.0
            sequence_padding_mark.append(padding_mark)
        # batch_size * seq_len
        sequence_padding_mark = tf.stack(sequence_padding_mark, axis=0)
        # 扩充维度以便用于attention矩阵
        return sequence_padding_mark[:, np.newaxis, np.newaxis, :] # (batch_size,1,1,seq_len)

    def calc_node_embedding(self, tree, node):
        if node.data[1] is None:
            if node.is_leaf():
                node_items = np.array(node.data[0], dtype=np.int32)
                node_items_embedding = self.item_embedding_layer(node_items)
                node.data[1] = self.get_node_embedding(node_items_embedding)
                if node.data[3] is None:
                    node_items_pos = np.array(node.data[2], dtype=np.float32)
                    node.data[3] = np.mean(node_items_pos)
                return node.data[1]
            else:
                children_embedding = []
                children_nodes = tree.children(node.identifier)
                for children in children_nodes:
                    newnode = tree.get_node(children.identifier)
                    embedding = self.calc_node_embedding(tree, newnode)
                    children_embedding.append(embedding)
                children_embedding = tf.stack(children_embedding, axis=0)
                node.data[1] = self.get_node_embedding(children_embedding)
                if node.data[3] is None:
                    node_items_pos = np.array(node.data[2], dtype=np.float32)
                    node.data[3] = np.mean(node_items_pos)
                return node.data[1]
        else:
            return node.data[1]
        
    def get_node_embedding(self, input_embedding):
        if self.agg == 'sum':
            output = tf.nn.tanh(tf.reduce_sum(input_embedding, axis=0)).numpy()
        elif self.agg == 'max':
            output = tf.reduce_max(input_embedding, axis=0).numpy()
        elif self.agg == 'mean':
            output = tf.reduce_max(input_embedding, axis=0).numpy()
        else:
            print("unknown agg: %s" % self.agg)
            output = None
            
        return output
        
    def tree_init(self, tree):
        for nid in tree.expand_tree(mode=Tree.WIDTH, sorting=False):
            node = tree[nid]
            if node.is_leaf():
                if node.data[1] is None:
                    node_items = np.array(node.data[0], dtype=np.int32)
                    node_items_embedding = self.item_embedding_layer(node_items)
                    node.data[1] = self.get_node_embedding(node_items_embedding)
                if node.data[3] is None:
                    node_items_pos = np.array(node.data[2], dtype=np.float32)
                    node.data[3] = np.mean(node_items_pos)
            else:
                node.data[1] = None
                
        return tree
    
    def tree_aggregation(self, tree):
        tree = self.tree_init(tree)
        root_node = tree.get_node(tree.root)
        self.calc_node_embedding(tree, root_node)
        
        return tree
    
    def tree2seq(self, tree, x):
        seq = []
        for nid in tree.expand_tree(mode=Tree.WIDTH, sorting=False):
            # node embedding
            seq.append(tree[nid].data[x])
        return seq
    
    def seq2tree(self, tree, seq, x):
        idx = 0
        for nid in tree.expand_tree(mode=Tree.WIDTH, sorting=False):
            if idx < len(seq):
                tree[nid].data[x] = seq[idx]
                idx += 1
    
        return tree
    
    def embedding_layer(self, user_input, complex_tree_input):
        complex_sequence_embedding = []
        num_complex_trees = len(complex_tree_input)
        for idx in range(num_complex_trees):
            complex_tree = complex_tree_input[idx]
            complex_tree = self.tree_aggregation(complex_tree)
            num_complex_tree_nodes = complex_tree.size()
            if num_complex_tree_nodes > self.max_complex_tree_nodes:
                self.max_complex_tree_nodes = num_complex_tree_nodes
            complex_tree_node_embedding = self.tree2seq(complex_tree, 1)
            complex_tree_node_time = self.tree2seq(complex_tree, 3)
            if num_complex_tree_nodes < self.max_nodes:
                pad_length = self.max_nodes - num_complex_tree_nodes
                for _ in range(pad_length):
                    zeros = tf.zeros(self.embedding_size)
                    complex_tree_node_embedding.insert(0, zeros)
                complex_tree_node_pos = sorted(range(len(complex_tree_node_time)), 
                                               key=lambda k: complex_tree_node_time[k])
                pad_pos = [pos for pos in range(pad_length)]
                node_pos = [pos+pad_length for pos in complex_tree_node_pos]
                complex_tree_node_pos = pad_pos + node_pos
                complex_tree_pos_embedding = tf.nn.embedding_lookup(self.pos_embedding, complex_tree_node_pos)
            else:
                complex_tree_node_embedding = complex_tree_node_embedding[:self.max_nodes]
                complex_tree_node_time = complex_tree_node_time[:self.max_nodes]
                complex_tree_node_pos = sorted(range(len(complex_tree_node_time)),
                                               key=lambda k: complex_tree_node_time[k])
                complex_tree_pos_embedding = tf.nn.embedding_lookup(self.pos_embedding, complex_tree_node_pos)
            # seq_len * embedding_size
            complex_tree_node_embedding = tf.stack(complex_tree_node_embedding, axis=0)
            complex_tree_pos_embedding = tf.stack(complex_tree_pos_embedding, axis=0)
            # add pos embedding
            complex_tree_embedding = complex_tree_node_embedding + complex_tree_pos_embedding
            complex_sequence_embedding.append(complex_tree_embedding)
        # batch_size * seq_len * embedding_size
        complex_sequence_embedding = tf.stack(complex_sequence_embedding, axis=0)
        # batch_size * embedding_size
        user_embedding = self.user_embedding_layer(user_input)
        user_pos = complex_sequence_embedding.shape[1]
        # 1 * embedding_size
        user_pos_embedding = tf.nn.embedding_lookup(self.pos_embedding, [user_pos])
        # add pos embedding
        user_embedding = user_embedding + user_pos_embedding
        # batch_size * 1 * embedding_size
        user_embedding = tf.expand_dims(user_embedding, axis=1)
        # batch_size * (seq_len+1) * embedding_size
        sequence_embedding = tf.concat([complex_sequence_embedding,user_embedding], axis=1)
        
        return sequence_embedding
    
    def output_layer(self, user_embedding):
        # get all items embedding
        all_items = tf.range(self.num_items)
        # num_items * embedding_size
        all_items_embedding = self.item_embedding_layer(all_items)
        # batch_size * num_items
        output = tf.matmul(user_embedding, all_items_embedding, transpose_b=True)
        output = tf.nn.softmax(output)
        
        return output

    def call(self, inputs, training=None):
        # batch_size
        user_input = inputs[0]
        # batch_size
        complex_tree_input = inputs[1]
        # batch_size * (seq_len+1) * embedding_size
        sequence_embedding = self.embedding_layer(user_input, complex_tree_input)
        # tree transformer layer
        # batch_size * 1 * 1 * (seq_len+1)
        sequence_padding_mark = self.create_padding_mark(complex_tree_input)
        sequence_embedding,loss,weights = self.transformer_layer([sequence_embedding, complex_tree_input],
                                                    training,
                                                    sequence_padding_mark)
        # batch_size * embedding_size
        user_embedding = sequence_embedding[:, -1, :]
        output = self.output_layer(user_embedding)
        
        return output,loss,weights

def find_sub_complex_sequence_index(complex_sequence, query):
    start_outer,start_inner,end_outer,end_inner = None,None,None,None
    query_start,query_end = query[0],query[-1]
    for i in range(len(complex_sequence)):
        item_set = complex_sequence[i]
        for j in range(len(item_set)):
            item = item_set[j]
            if item == query_start:
                start_outer = i
                start_inner = j
            if item == query_end:
                end_outer = i
                end_inner = j
                return start_outer,start_inner,end_outer,end_inner

def find_sub_complex_sequence(complex_sequence, query):
    # complex_sequence = [(2,3),(4,5,6),(7,8),(9,10,11,12),(13,14)]
    # query = [5,6,7,8]
    # output = [(5, 6), (4, 5)]
    output = []
    start_outer,start_inner,end_outer,end_inner = find_sub_complex_sequence_index(complex_sequence,query)
    if start_outer == end_outer:
        tmp = complex_sequence[start_outer]
        output.append(tmp[start_inner:end_inner+1])
    else:
        left = complex_sequence[start_outer]
        right = complex_sequence[end_outer]
        output.append(left[start_inner:])
        for idx in range(start_outer+1, end_outer):
            output.append(complex_sequence[idx])
        output.append(right[:end_inner+1])
    
    return output

def add_tree_children(tree, leaves, simplicial_complex, user, time_matrix):
    node_items = simplicial_complex
    identifier=str(node_items)
    if tree.contains(identifier) == False:
        for leaf in leaves:
            if set(simplicial_complex).issubset(set(leaf.data[0])):
                # node items, node embedding
                node_items_pos = [time_matrix[user, item_id] for item_id in node_items]
                # node_items_pos = [pos_start+idx for idx in range(len(node_items))]
                data = [node_items,None,node_items_pos,None]
                tree.create_node(tag=identifier, identifier=identifier, data=data, parent=leaf)
                return

# [227, 257, 252, 218, 229, 292, 918, 939, 224, 865, 938, 288, 1232]
# ├── [227, 257, 252]
# │   ├── [227]
# │   ├── [257]
# │   └── [252]
# ├── [218, 229, 292]
# │   ├── [218]
# │   ├── [229]
# │   └── [292]
# └── [918, 939, 224, 865, 938, 288, 1232]
#     ├── [918]
#     ├── [939]
#     ├── [224]
#     ├── [865]
#     ├── [938]
#     ├── [288]
#     └── [1232]
def build_tree(user, time_matrix, simplicial_complex_sequence, epsilons):
    tree = Tree(identifier="tree")
    eps = epsilons[-1]
    if len(simplicial_complex_sequence[eps]) == 1:
        node_items = simplicial_complex_sequence[eps][0]
        identifier=str(node_items)
        node_items_pos = [time_matrix[user, item_id] for item_id in node_items]
        # node_items_pos = [idx for idx in range(len(node_items))]
        # node items, node embedding, node item pos, node pos embedding
        data = [node_items,None,node_items_pos,None]
        tree.create_node(tag=identifier,identifier=identifier,data=data)
    else:
        root_items = []
        for simplicial_complex in simplicial_complex_sequence[eps]:
            root_items.extend(simplicial_complex)
        root_items_pos = [time_matrix[user, item_id] for item_id in root_items]
        # root_items_pos = [idx for idx in range(len(root_items))]
        # node items, node embedding, node item pos, node pos embedding
        data = [root_items,None,root_items_pos,None]
        root_identifier=str(root_items)
        tree.create_node(tag=root_identifier,identifier=root_identifier,data=data)
        # pos_start = 0
        for simplicial_complex in simplicial_complex_sequence[eps]:
            node_items = simplicial_complex
            node_items_pos = [time_matrix[user, item_id] for item_id in node_items]
            # node_items_pos = [pos_start+idx for idx in range(len(node_items))]
            # node items, node embedding, node item pos, node pos embedding
            data = [node_items,None,node_items_pos,None]
            identifier=str(node_items)
            tree.create_node(tag=identifier, identifier=identifier, data=data, parent=root_identifier)
            # pos_start = pos_start + len(node_items)
    
    for idx in range(1, len(epsilons)):
        epsidx = len(epsilons) - idx -1
        eps = epsilons[epsidx]
        leaves = tree.leaves()
        # pos_start = 0
        for simplicial_complex in simplicial_complex_sequence[eps]:
            add_tree_children(tree, leaves, simplicial_complex, user, time_matrix)
            # pos_start = pos_start + len(simplicial_complex)
            
    return tree

class TIMe(SeqAbstractRecommender):
    def __init__(self, dataset, conf):
        super(TIMe, self).__init__(dataset, conf)
        self.data_path = os.path.join(conf["data.input.path"], dataset.dataset_name)
        self.embedding_size = conf["embedding_size"]
        self.batch_size = conf["batch_size"]
        self.num_epochs = conf["epochs"]
        self.verbose = conf["verbose"]
        self.learning_rate = conf["learning_rate"]
        self.l1_reg = conf["l1_reg"]
        self.l2_reg = conf["l2_reg"]
        self.min_len = conf["min_len"]
        self.max_len = conf["max_len"]
        self.max_nodes = conf["max_nodes"]
        self.epsilons = conf["epsilons"]
        self.num_blocks = conf["num_blocks"]
        self.num_heads = conf["num_heads"]
        self.drop_rate = conf["drop_rate"]
        self.agg = conf["agg"]
        
        self.train_matrix = dataset.train_matrix
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_dict = csr_to_user_dict_bytime(dataset.time_matrix, dataset.train_matrix)
        self.user_complex_sequence = self.get_user_complex_sequence()
        

    def build_graph(self):
        pass
    
    def get_user_complex_sequence(self):
        get_user_complex_sequence_begin = time()
        user_complex_sequence_path = "%s.user.complex.sequence.pkl" % self.data_path
        if os.path.exists(user_complex_sequence_path):
            with open(user_complex_sequence_path, 'rb') as f:
                user_complex_sequence = pickle.load(f)
        else:
            user_complex_sequence = self.load_user_complex_sequence()
        get_user_complex_sequence_time = time() - get_user_complex_sequence_begin
        print("model get user complex sequence time=%.4fs." % (get_user_complex_sequence_time))
        
        return user_complex_sequence
    
    def load_user_complex_sequence(self):
        user_complex_sequence = {}
        time_matrix = self.dataset.time_matrix
        for user, item_sequence in self.train_dict.items():
            points = [[user, time_matrix[user, item_id]] for item_id in item_sequence]
            for idx in range(len(self.epsilons)):
                complex_sequence = self.topological_transform(points, item_sequence, self.epsilons[idx])
                if user not in user_complex_sequence:
                    user_complex_sequence[user] = {}
                user_complex_sequence[user][self.epsilons[idx]] = complex_sequence
        # save train data
        user_complex_sequence_path = "%s.user.complex.sequence.pkl" % self.data_path
        with open(user_complex_sequence_path, 'wb') as f:
            pickle.dump(user_complex_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("save user complex sequence to %s." % (user_complex_sequence_path))
        
        return user_complex_sequence
    
    def topological_transform(self,points,item_sequence,epsilon):
        complex_sequence = []
        points = np.array(points, dtype=np.float32)
        vr = VietorisRipsComplex(points, epsilon)
        for simplical_complex in vr.simplical_complexes:
            item_set = [item_sequence[idx] for idx in simplical_complex]
            complex_sequence.append(item_set)

        return complex_sequence
    
    def fast_topological_transform(self,user,item_sequence,epsilon):
        complex_sequence = self.user_complex_sequence[user][epsilon]
        sub_complex_sequence = find_sub_complex_sequence(complex_sequence,item_sequence)

        return sub_complex_sequence
    
    def topological_transform_layer(self,user_input,item_sequence_input,points_input):
        multi_scale_complex_tree_input = []
        time_matrix = self.dataset.time_matrix
        num_instances = len(item_sequence_input)
        for instance in range(num_instances):
            user = user_input[instance]
            item_sequence = item_sequence_input[instance]
            multi_scale_complex_sequence = {}
            for idx in range(len(self.epsilons)):
                complex_sequence = self.fast_topological_transform(user,item_sequence,self.epsilons[idx])
                multi_scale_complex_sequence[self.epsilons[idx]] = complex_sequence
            multi_scale_complex_tree = build_tree(user, time_matrix, multi_scale_complex_sequence, self.epsilons)
            multi_scale_complex_tree_input.append(multi_scale_complex_tree)
            
        return multi_scale_complex_tree_input
            
    def get_train_data(self):
        get_train_data_begin = time()
        train_data_path = "%s.TIMe.train.pkl" % self.data_path
        if os.path.exists(train_data_path):
            with open(train_data_path, 'rb') as f:
                user_input,item_sequence_input,points_input,next_item = pickle.load(f)
        else:
            user_input,item_sequence_input,points_input,next_item = self.load_train_data()
        get_train_data_time = time() - get_train_data_begin
        print("model get train data time=%.4fs." % (get_train_data_time))
        
        return user_input,item_sequence_input,points_input,next_item
    
    def load_train_data(self):
        user_input,item_sequence_input,points_input,next_item = [],[],[],[]
        time_matrix = self.dataset.time_matrix
        for user, item_sequence in self.train_dict.items():
            if len(item_sequence) > 0 and len(item_sequence) <= self.min_len:
                idx = len(item_sequence) - 1
                user_input.append(user)
                next_item.append(item_sequence[idx])
                one_item_sequence = item_sequence[:idx]
                if len(one_item_sequence) > self.max_len:
                    one_item_sequence = one_item_sequence[-self.max_len:]
                    item_sequence_input.append(one_item_sequence)
                    points = [[user, time_matrix[user, item_id]] for item_id in one_item_sequence]
                    points_input.append(points)
                else:
                    item_sequence_input.append(item_sequence[:idx])
                    points = [[user, time_matrix[user, item_id]] for item_id in item_sequence[:idx]]
                    points_input.append(points)
            elif len(item_sequence) > self.min_len:
                for idx in range(self.min_len, len(item_sequence)):
                    user_input.append(user)
                    next_item.append(item_sequence[idx])
                    one_item_sequence = item_sequence[:idx]
                    if len(one_item_sequence) > self.max_len:
                        one_item_sequence = one_item_sequence[-self.max_len:]
                        item_sequence_input.append(one_item_sequence)
                        points = [[user, time_matrix[user, item_id]] for item_id in one_item_sequence]
                        points_input.append(points)
                    else:
                        item_sequence_input.append(item_sequence[:idx])
                        points = [[user, time_matrix[user, item_id]] for item_id in item_sequence[:idx]]
                        points_input.append(points)
            else:
                print("items num of user %d = 0." % user)
        # save train data
        train_data_path = "%s.TIMe.train.pkl" % self.data_path
        with open(train_data_path, 'wb') as f:
            pickle.dump([user_input,item_sequence_input,points_input,next_item], 
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print("save train data to %s." % (train_data_path))
        
        return user_input,item_sequence_input,points_input,next_item
    
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        user_input,item_sequence_input,points_input,next_item = self.get_train_data()
        num_training_instances = len(user_input)
        print("training instances num = %d." % num_training_instances)
        data_iter = DataIterator(user_input,
                                 item_sequence_input,
                                 points_input,
                                 next_item,
                                 batch_size=self.batch_size,
                                 shuffle=True)
        # build network
        self.network = TIMeModel(self)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        for epoch in  range(1, self.num_epochs+1):
            total_loss = 0.0
            training_start_time = time()
            for bat_user,bat_item_sequence,bat_points,bat_next_item in tqdm(data_iter):
                bat_multi_scale_complex_tree = self.topological_transform_layer(bat_user, bat_item_sequence, bat_points)
                bat_user = np.array(bat_user, dtype=np.int32)
                bat_next_item = np.array(bat_next_item, dtype=np.int32)
                with tf.GradientTape() as tape:
                    # batch_size * num_items
                    output,l1_loss,weights = self.network([bat_user, bat_multi_scale_complex_tree], training=True)
                    # crossentropy loss for L(Theta)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(bat_next_item,output)
                    loss = tf.reduce_sum(loss)
                    # L1 loss
                    loss += self.l1_reg * l1_loss
                    # L2 loss
                    regularization = []
                    for v in self.network.trainable_variables:
                        regularization.append(tf.nn.l2_loss(v))
                    loss_regularization = tf.reduce_sum(tf.stack(regularization))
                    loss += self.l2_reg * loss_regularization
                grads = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
                total_loss += loss.numpy()
            self.logger.info("[iter %d : loss : %f, time: %f, max_complex_tree_nodes:%d]" %
                             (epoch, total_loss/num_training_instances, time()-training_start_time,
                              self.network.max_complex_tree_nodes))
            
            if epoch % self.verbose == 0 or epoch == self.num_epochs:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate_model()))

    @timer
    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        item_sequence_input,points_input = [],[]
        time_matrix = self.dataset.time_matrix
        for user_id in user_ids:
            item_sequence = self.train_dict[user_id]
            if len(item_sequence) > self.max_len:
                one_item_sequence = item_sequence[-self.max_len:]
                item_sequence_input.append(one_item_sequence)
                points = [[user_id, time_matrix[user_id, item_id]] for item_id in one_item_sequence]
                points_input.append(points)
            else:
                item_sequence_input.append(item_sequence)
                points = [[user_id, time_matrix[user_id, item_id]] for item_id in item_sequence]
                points_input.append(points)
        multi_scale_complex_tree = self.topological_transform_layer(user_ids,item_sequence_input, points_input)
        user_ids = np.array(user_ids, dtype=np.int32)
        # batch_size * num_items
        output,l1_loss,weights = self.network([user_ids, multi_scale_complex_tree], training=False)
        if candidate_items is None:
            return output.numpy()
        else:
            ratings = output.numpy()
            ratings = [ratings[idx, u_item] for idx, u_item in enumerate(candidate_items)]
            return np.vstack(ratings)