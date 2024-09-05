
import torch.backends.cudnn as cudnn
import torch, random,time,os,logging
import numpy as np

from genotypes import Genotype_mapping

import networkx as nx


os.environ['CUDA_VISIBLE_DEVICES']= '0'

from copy import deepcopy

from threading import Thread

import matplotlib.pyplot as plt
from collections import defaultdict



class Node():
    def __init__(self,item,left = None,right =None,shape=None):
        self.item = item
        self.left = left
        self.right = right
        self.shape = shape
        self.No = None

class Tree():
    def __init__(self,root=None):
        self.root = root

    def level_travel(self):
        if self.root==None:
            return
        Queue = [self.root]
        while Queue:
            cur  = Queue.pop(0)
            print(cur.item,end=' ')
            if cur.left!=None:
                Queue.append(cur.left)
            if cur.right!=None:
                Queue.append(cur.right)
    def preorder_travel(self,node):
        if node ==None:
            return
        print(node.item,end=' ')
        self.preorder_travel(node.left)
        self.preorder_travel(node.right)
    def postorder_travel(self,node):
        if node ==None:
            return
        self.postorder_travel(node.left)
        self.postorder_travel(node.right)
        print(node.item,end=' ')
    def inorder_travel(self,node):
        if node ==None:
            return
        self.inorder_travel(node.left)
        print(node.item,end=' ')
        self.inorder_travel(node.right)

    def add(self,item):
        node = Node(item)
        if self.root is None:
            self.root = node
            return
        Queue = [self.root]
        while Queue:
            cur  = Queue.pop(0)
            if cur.left ==None:
                cur.left = node
                return
            else:
                Queue.append(cur.left)
            if cur.right == None:
                cur.right = node
                return
            else:
                Queue.append(cur.right)
    def sample(self,mapping,nodes):
        if nodes==0:
            return
        op = np.random.randint(0,len(mapping))
        # op = 11
        node = Node(item=mapping[op])
        if self.root is None:
            self.root = node
            self.sample(mapping,nodes-1)
            return

        Queue = [self.root]
        while len(Queue)!=0:
            cur = Queue.pop(0)
            if cur.item not in ['add','mul','concat']:
                cur.right = None
                if cur.left is None:
                    cur.left = node
                    self.sample(mapping,nodes-1)
                    return
                else:
                    Queue.append(cur.left)

            else:
                if np.random.rand()<0.5:
                    if cur.left is None:
                        cur.left = node
                        self.sample(mapping,nodes-1)
                        return

                    else:
                        Queue.append(cur.left)

                    if cur.right is None:
                        cur.right = node
                        self.sample(mapping,nodes-1)
                        return
                    else:
                        Queue.append(cur.right)
                else:

                    if cur.left is None:
                        cur.left = node
                        self.sample(mapping,nodes-1)
                        return

                    else:
                        Queue.append(cur.left)

                    if cur.right is None:
                        cur.right = node
                        self.sample(mapping,nodes-1)
                        return
                    else:
                        Queue.append(cur.right)

                pass






    def visualization(self,paht=None):
        def create_graph(G, node, p_name, pos={}, x=0, y=0, layer=1):
            if node == None:
                return
            if node.shape is not None:
                # name = str(node.item) +'\n'+ str(node.shape)
                name = str(node.item)
            else:
                name = str(node.item)
            saw[name] += 1
            if name in saw.keys():
                name += ' '* saw[name]

            G.add_edge(p_name, name)
            G.add_node(name,layer=layer)
            pos[name] = (x, y)


            if node.right is None or node.left is None:
                l_x, l_y = x, y - 0.02

            else:
                l_x, l_y = x - 0.01, y - 0.02
            l_layer = layer + 1
            create_graph(G, node.left, name, x=l_x, y=l_y, pos=pos, layer=l_layer)

            if node.right is None or node.left is None:
                r_x, r_y = x, y - 0.02
            else:
                r_x, r_y = x + 0.01, y - 0.02
            r_layer = layer + 1
            create_graph(G, node.right,name, x=r_x, y=r_y, pos=pos, layer=r_layer)
            return (G, pos)

        saw = defaultdict(int)
        graph = nx.DiGraph()
        graph, pos = create_graph(graph, self.root,"source")
        pos["source"] = (0,0)
        graph.remove_node("source")
        # fig, ax = plt.subplots(figsize=(100, 300))  # 比例可以根据树的深度适当调节


        x_values, y_values = zip(*pos.values())
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        # 归一化坐标
        normalized_pos = {node: ((x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y)) for node, (x, y) in pos.items()}

        # 创建图形并指定尺寸
        plt.figure(figsize=(8, 6))

        # 绘制图形
        # nx.draw_networkx(graph, pos=normalized_pos, node_size=40,font_size=4,width=0.1,font_family="Times New Roman",arrowsize=3,node_color="#B8F283")
        # nx.draw_networkx(graph, pos=normalized_pos, node_size=20,font_size=2,width=0.1,font_family="Times New Roman",arrowsize=3,node_color="#B8F283")
        nx.draw_networkx(graph, pos=normalized_pos, node_size=600,font_size=10,width=2.,font_family="Times New Roman",arrowsize=10,node_color="#B8F283")



        # 显示图形
        plt.show()





class TreeNode():
    def __init__(self,item,left = None,right =None,shape=None):
        self.item = item
        self.left = left
        self.right = right
        self.shape = shape
        self.No = None
class Individual():
    def __init__(self, Dec=None, num_Nodes=5,mapping=None,config=None, gen=0,id=0):
        self.config = config
        self.id = id
        self.gen = gen
        self.pre_inputs=['Stu','Exer','Conc']

        self.mapping =mapping
        self.Reverse_mapping = dict([val, key] for key, val in self.mapping.items())


        if Dec is None:
            self.Dec = Dec
            self.numNodes = num_Nodes
            self.RandomBuildTree()
        else:
            self.numNodes = len(Dec)//3
            self.build_treeFromDec(Dec)
            self.UpdateShape(self.tree.root)
            self.RepairConstraint(self.tree.root)
            self.getNumNode()
        self.Get_DecArrary()
        a=1




    def Get_DecArrary(self): # return int arrary for building NASCDNet, Post-Order travel

        s1 = []
        s2 = []
        s1.append(self.tree.root)  # post order travel by two stacks
        while len(s1)>0:
            cur = s1.pop()
            s2.append(cur)
            if cur.left is not None and cur.left.item not in self.pre_inputs:
                s1.append(cur.left)
            if cur.right is not None and cur.right.item not in self.pre_inputs:
                s1.append(cur.right)
        Dec = []
        candidate_inputs = deepcopy(self.pre_inputs)

        for idx,node_i in enumerate(s2[::-1]):
            node_i.No = idx+3

            # x1 = candidate_inputs.index(node_i.left.item)
            if node_i.left.item == 'Stu':
                x1=0
            elif node_i.left.item == 'Exer':
                x1=1
            elif node_i.left.item == 'Conc':
                x1=2
            else:
                x1 = node_i.left.No

            # x1 = candidate_inputs.index(node_i.left.item)
            # if x1>2:
            #     candidate_inputs[x1]='used'
            if node_i.item in ['add','mul','concat']:

                if node_i.right.item == 'Stu':
                    x2=0
                    x1,x2 = x2,x1  # 0, [0,1,2]
                elif node_i.right.item == 'Exer':   # exchange for unique encoding
                    x2=1
                    if x1>x2:
                        x1,x2 = x2,x1  # 1,2
                        # else:[0,1],1
                elif node_i.right.item == 'Conc':
                    x2=2   # [0,1,2],2
                else:
                    x2 = node_i.right.No
                # x2 = candidate_inputs.index(node_i.right.item)
                # if x2>2:
                #     candidate_inputs[x2]='used'
            else:
                x2 = 0

            # candidate_inputs.append(node_i.item)
            candidate_inputs.append(node_i.No)
            op_num = self.Reverse_mapping[node_i.item]
            Dec.extend([x1,x2,op_num])

        self.Dec = Dec



    def getNumNode(self):
        num = 0
        Queue = [self.tree.root]
        while len(Queue)!=0:
            cur = Queue.pop(0)

            if cur.item in self.mapping.values():
                num +=1
            if cur.left!=None:
                Queue.append(cur.left)
            if cur.right!=None:
                Queue.append(cur.right)
        self.numNodes = num
        return self.numNodes
    def getLeafNum(self):
        num = 0
        Queue = [self.tree.root]
        while len(Queue)!=0:
            cur = Queue.pop(0)
            if cur.item in self.pre_inputs:
                num +=1
            if cur.left!=None:
                Queue.append(cur.left)
            if cur.right!=None:
                Queue.append(cur.right)
        self.leafNum = num
        return self.leafNum







    def RandomBuildTree(self):
        tree = Tree()
        tree.sample(self.mapping,self.numNodes)
        self.tree = tree
        self.AddLeafNode(self.tree.root)

        # basic steps after a solution is generated
        self.UpdateShape(self.tree.root)
        self.RepairConstraint(self.tree.root)
        self.getNumNode()
        #------------------
        # abc = self.get_subTree(2)
        a = 1


    def RepairConstraint(self,node):
        if node is None:
            return
        elif node.item in self.pre_inputs:
            return

        self.RepairConstraint(node.left)
        self.RepairConstraint(node.right)

        if node.left.shape =='single' and node.item in ['mean','sum','ffn','concat']:   # 修复  mean 后续不能再直接 follow mean等操作
            op = node.item
            while op  in ['mean','sum','ffn','concat']:
                op = np.random.randint(0,len(self.mapping))
                op = self.mapping[op]
            if node.item =='concat' and op in  ['add','mul']: # for concat, directly used binary operator for replacement
                node.item = op
            elif op in ['add','mul']: #for ['mean','sum','ffn'],  binary operator
                node.item = op
                # adding right child
                candidate = np.random.randint(0,len(self.pre_inputs))
                node.right = TreeNode(item=self.pre_inputs[candidate],shape='same')
                node.shape = 'same'
            else: # unary operator
                node.item = op
                node.right = None # used for 'concat'
        #------------------------- 修复concat-----------
        if node.item=='concat' and node.left.shape!=node.right.shape:
            if np.random.rand()<0.5:
                node.item = 'add'
            else:
                node.item = 'mul'


        #----------------------------------------------
        if node.item not in ['add','mul','concat']: # 修复  连续相同的 （unary）操作
            if node.item == node.left.item:
                node.left = node.left.left

        #------------------ update shape information ----------------------------------
        if node.item in ['add','mul','concat']:
            if node.left.shape==node.right.shape:
                node.shape = node.left.shape
            else:
                node.shape ='same'

        elif node.item in ['sum','mean','ffn']:
            node.shape = 'single'
        else:
            node.shape = node.left.shape


    def AddLeafNode(self,node):
        if node is None:
            return
        elif node.item in self.pre_inputs:
            return

        if node.right is None and node.left is not None and node.item in ['add','mul','concat']:
            candidate = np.random.randint(0,len(self.pre_inputs))
            node.right = TreeNode(item=self.pre_inputs[candidate],shape='same')

        if node.left ==None and node.right==None:
            if node.item not in ['add','mul','concat']:
                candidate = np.random.randint(0,len(self.pre_inputs)-1)    # only select from stu and Exer to avoid mistakes
                # only select from stu and Exer to avoid mistakes
                node.left = TreeNode(item=self.pre_inputs[candidate],shape='same')
            else:
                # candidate = np.random.randint(0,len(self.pre_inputs)-1,2)  # only select from stu and Exer to avoid mistakes
                candidate = np.random.choice(range(len(self.pre_inputs)),2,replace=False) # avoid same inputs
                node.left = TreeNode(item=self.pre_inputs[candidate[0]],shape='same')
                node.right = TreeNode(item=self.pre_inputs[candidate[1]],shape='same')
        self.AddLeafNode(node.left)
        self.AddLeafNode(node.right)

    def UpdateShape(self,node):
        if node is None:
            return
        elif node.item in self.pre_inputs:
            return

        self.UpdateShape(node.left)
        self.UpdateShape(node.right)

        if node.item in ['add','mul','concat']:
            if node.left.shape==node.right.shape:
                node.shape = node.left.shape
            else:
                node.shape ='same'

        elif node.item in ['sum','mean','ffn']:
            node.shape = 'single'
        else:
            node.shape = node.left.shape

    def build_treeFromDec(self,Dec):
        self.Dec = Dec
        nodes = [TreeNode('Stu',shape='same'),TreeNode('Exer',shape='same'),TreeNode('Conc',shape='same')]
        self.numNodes = len(Dec)//3
        for i in range(self.numNodes):
            temp = Dec[3*i:3*(i+1)]
            x1,x2,op = temp[0],temp[1],temp[2]
            if self.mapping[op] in ['add','mul','concat']:
                node_i = TreeNode(item=self.mapping[op], left=nodes[x1], right=nodes[x2])
            else:
                node_i = TreeNode(item=self.mapping[op], left=nodes[x1], right=None)
            nodes.append(node_i)

        self.tree = Tree(root=nodes[-1])
    def visualization(self,path=None):
        self.tree.visualization(path)





if __name__ == '__main__':
    I1 = Individual(Dec=[0, 1, 11, 3, 0, 1, 1, 4, 10, 5, 0, 13, 6, 0, 1, 7, 0, 4, 8, 0, 2, 9, 0, 4, 10, 0, 1, 11, 0, 0, 12,
                         0, 7, 13, 0, 4, 2, 14, 11, 15, 0, 0, 16, 0, 2, 17, 0, 7, 18, 0, 9, 19, 0, 0, 20, 0, 7, 1, 21, 11,
                         22, 0, 6, 23, 0, 1, 24, 0, 13, 2, 25, 11, 26, 0, 0, 27, 0, 4, 1, 28, 10, 0, 29, 11, 30, 0, 7, 31, 0,
                         2, 0, 32, 10, 33, 0, 13, 34, 0, 7, 35, 0, 6]   ,mapping=Genotype_mapping )


    I1.visualization()
    A=1