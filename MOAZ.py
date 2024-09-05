
import torch.backends.cudnn as cudnn
import torch, random,time,os,logging
import numpy as np
import matplotlib.pyplot as plt

from utils.config import get_common_search_config
from utils.utils import get_dataset,write_txt
from utils.Evaluation_Model import solution_evaluation_MOAZ
from Tree.Genetic_Operator import  Generate_crossover_mutation_MOAZ, Generate_AZ
from genotypes import Genotype_mapping
from EMO_public import F_distance,NDsort,F_mating,F_EnvironmentSelect
from Tree.Node import Tree
from Tree.Node import Node as TreeNode
import sys,gc,pickle

from Models.NASCDNetV2 import NASCDNet

os.environ['CUDA_VISIBLE_DEVICES']= '0'

from copy import deepcopy

from threading import Thread

def get_latest_folder(directory):
    # 获取指定目录下所有条目的绝对路径
    absolute_paths = [os.path.join(directory, f) for f in os.listdir(directory)]

    # 筛选出是文件夹的条目，并获取其最后修改时间
    folder_paths = [(path, os.path.getmtime(path)) for path in absolute_paths if os.path.isdir(path)]

    if not folder_paths:
        return None

    # 按最后修改时间排序，并返回最新的文件夹
    latest_folder = max(folder_paths, key=lambda x: x[1])[0]
    return latest_folder

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

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
    def Deletion(self,index):  # the index is not same as No. (index is based on level travel, No. based on Post-Order travel)

        subtree = self.get_subTree(index)
        if subtree.right is not None and subtree.right.item not in self.pre_inputs: # select the left or right
            if np.random.rand()<0.5:
                used_tree = subtree.left
            else:
                used_tree = subtree.right
        else:
            used_tree = subtree.left
        self.set_subTree(subtree.No,used_tree)
        self.After_Genetic()

    def Insertion(self,index): # Insert a node as the parent pf the Node(index)
        subtree = self.get_subTree(index)
        #------- -------------generate the Insert_node --------------
        if index ==1 and np.random.rand()<0.7:  # big probability for binary operator
            randi = np.random.randint(0,2)
            if randi ==0:
                op ='add'
            elif randi==1:
                op = 'mul'
            else:
                op = 'concat'
        else:
            op = np.random.randint(0,len(self.mapping))
            op = self.mapping[op]

        Insert_node = TreeNode(item=op)

        #-----------------Set the Insert_node ---------------------
        # here we donot consider the feasiability  in terms of "shape", will be done by Repairing  -------

        if Insert_node.item in ['add','mul','concat']:
            Insert_node.left = subtree
            #--------- set the right
            candidate = np.random.randint(0,len(self.pre_inputs))
            Insert_node.right = TreeNode(item=self.pre_inputs[candidate],shape='same')
        else:
            Insert_node.left = subtree

        #-----------
        self.set_subTree(subtree.No,Insert_node)
        self.After_Genetic()

    def Replacement(self,index):
        subtree = self.get_subTree(index)

        #-------------generate the Replace_node --------------
        op = subtree.item
        while op == subtree.item:
            op = np.random.randint(0,len(self.mapping))
            op = self.mapping[op]
        Replace_node = TreeNode(item=op)
        #---------------Set the Replace_node, here we donot consider the feasiability  in terms of "shape", will be done by Repairing  --------------
        if subtree.item in ['add','mul','concat'] and Replace_node.item in ['add','mul','concat']:
            Replace_node.left = subtree.left
            Replace_node.right = subtree.right
        elif subtree.item in ['add','mul','concat'] and Replace_node.item not in ['add','mul','concat']:
            if np.random.rand()<0.5:
                Replace_node.left = subtree.left
            else:
                Replace_node.left = subtree.right
        elif subtree.item not in ['add','mul','concat'] and Replace_node.item in ['add','mul','concat']:
            Replace_node.left = subtree.left
            # randomly adding a input as child node
            candidate = np.random.randint(0,len(self.pre_inputs))
            Replace_node.right = TreeNode(item=self.pre_inputs[candidate],shape='same')
        else:
            Replace_node.left = subtree.left
        #---------------------
        self.set_subTree(subtree.No,Replace_node)
        self.After_Genetic()

    def After_Genetic(self):

        self.UpdateShape(self.tree.root)
        self.RepairConstraint(self.tree.root)
        self.getNumNode()
        self.Get_DecArrary()

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

    def set_subTree(self,Tree_No,another_subTree): # set tree according to the No., which is based on Post-Order

        if self.tree.root.No ==Tree_No:
            self.tree.root = another_subTree
            return

        Queue = [self.tree.root]
        while len(Queue)>0:
            cur = Queue.pop(0)

            if cur.left!=None:
                if cur.left.No ==Tree_No:
                    cur.left = another_subTree
                    return
                else:
                    Queue.append(cur.left)
            if cur.right!=None:
                if cur.right.No ==Tree_No:
                    cur.right= another_subTree
                    return
                else:
                    Queue.append(cur.right)


    def get_subTree(self, index): # counting from root node to maxi: level travel
        subtree = []
        Queue = [self.tree.root]
        while index>0:
            cur = Queue.pop(0)
            if cur.item in self.mapping.values():
                index -=1
                subtree = cur
            if cur.left!=None:
                Queue.append(cur.left)
            if cur.right!=None:
                Queue.append(cur.right)

        return subtree


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



    def tree_deep(self,node): # include root node and leaf node
        if node is None:
            return 0
        left, right = 0,0
        if node.left is not None:
            left = self.tree_deep(node.left)
        if node.right is not None:
            right = self.tree_deep(node.right)
        return max(left,right)+1

    def Compute_Complexity(self):
        deep_number = self.tree_deep(self.tree.root)-2 #  do not statistics root and leaf node
        self.deep_number = deep_number
        leaf_number = self.getLeafNum()
        node_number = self.getNumNode()
        # return deep_number,leaf_number,node_number
        fit_complexity = 0

        fit_complexity = self.deep_number/10 + (0.1-leaf_number/100)+ node_number/1000
        return 1-fit_complexity

    def Compute_Complexity_nodes(self):
        deep_number = self.tree_deep(self.tree.root)-2 #  do not statistics root and leaf node
        self.deep_number = deep_number
        leaf_number = self.getLeafNum()
        node_number = self.getNumNode()
        # return deep_number,leaf_number,node_number
        fit_complexity = 0

        fit_complexity = (leaf_number+node_number)/30
        return 1-fit_complexity




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



    def mkdir(self):

        self.save_dir = "{}/Gen_{}/[{}]/".format(self.config.exp_name,self.gen,self.id)
        self.training_log = self.save_dir+'training_log.txt'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save(self):
        # saving basic information
        self.dec_dir = self.save_dir+'dec.txt'
        self.fitness_dit = self.save_dir+'fitness.txt'
        information = 'Deep num:{}, LeafNode num:{}, Node num:{}'.format(self.deep_number,self.leafNum,self.numNodes)
        self.info_dir = self.save_dir+'infomation.txt'

        write_txt(self.info_dir,information)
        write_txt(self.dec_dir,self.Dec)
        self.visualization(self.save_dir)


    def evaluation(self,device):

        self.mkdir()
        f = open(self.training_log, "w+")
        print('Evaluating {}-th solution'.format(self.id), file=f,flush=True)
        print('Evaluating {}-th solution'.format(self.id), file=sys.stdout)
        logging.info('Evaluating {}-th solution'.format(self.id))
        #



        fit_complexity = self.Compute_Complexity()
        #fit_complexity = self.Compute_Complexity_nodes()
        self.save()

        Settings = [device,self.config,self.Dec ,self.save_dir,f]
        best_acc,best_auc, FLOPs = solution_evaluation_MOAZ(Settings)
        # best_acc,best_auc = np.random.rand(),np.random.rand()
        # self.fitness = np.random.rand(2,)

        self.fitness = [1-FLOPs,best_auc]
        print('{}-th solution: Best valid acc:{}, auc:{}  '.format(self.id,best_acc, self.fitness[1]),file=sys.stdout)
        logging.info('{}-th solution: Best valid acc:{}, auc:{}  '.format(self.id,best_acc, self.fitness[1]))



        np.savetxt( self.fitness_dit, np.array(self.fitness), delimiter=' ')
        gc.collect()
        f.close()



class MOAZ():
    def __init__(self,config):
        self.config = config


        if config.dataset=='Assistment':
            self.threshold = 0.75
        elif config.dataset=='slp':
            self.threshold = 0.82
        elif config.dataset=='junyi':
            self.threshold = 0.8

        self.Maxi_Gen = 100
        self.gen =0
        self.Popsize = 100
        #--------Population and offspring information-------------
        self.Population = []
        self.Pop_fitness = []

        self.offspring = []
        self.off_fitness=[]
        #-------other information--------------------
        self.tour_index = []
        self.FrontValue = []
        self.CrowdDistance =[]
        self.select_index = []
        self.Archive = []

        # self.LoadDataset()
        self.get_Boundary_Mapping()

    def LoadDataset(self):
        print('Loading Dataset....')
        self.config.student_n,self.config.exer_n,self.config.knowledge_n, \
        self.train_loader, self.val_loader = get_dataset(self.config)
        print('Loading Finish！')
    def get_Boundary_Mapping(self):
        self.mapping = Genotype_mapping
        logging.info('Genotype_mapping: '+str(self.mapping))
        print('Genotype_mapping: '+str(self.mapping))

    def Initialization(self):
        if config.Continue_path is None:
            self.set_dir(path='initial')
            self.Population=[]
            # self.Population.append(Individual(Dec=[2, 0, 8, 3, 0, 0],mapping=self.mapping,config=self.config,gen='initial',id=0) )
            # self.Population.append(Individual(Dec=[0,1,12, 3,0,9, 4,0,6],mapping=self.mapping,config=self.config,gen='initial',id=0) )
            # self.Population.append(Individual(Dec=[1, 0, 4, 3, 0, 3, 4, 0, 1, 0, 0, 13, 6, 0, 1, 5, 7, 10, 8, 0, 0],
            #                               mapping=self.mapping,config=self.config,gen='initial',id=1) )
            # self.Population.append(Individual(Dec=[1, 0, 1, 0, 0, 10, 3, 4, 12, 5, 0, 6, 6, 0, 13, 7, 0, 4],
            #                               mapping=self.mapping,config=self.config,gen='initial',id=2) )
            for idx in range(0,self.Popsize):
                num_nodes = np.random.randint(config.Num_Nodes[0],config.Num_Nodes[1]) # +1
                self.Population.append(Individual(num_Nodes=num_nodes,mapping=self.mapping,config=self.config,gen='initial',id=idx))
            self.Pop_fitness = self.Evaluation(self.Population)
            self.set_dir(path='initial')
            self.Save()
        else:
            pathdir = os.path.expandvars(config.Continue_path)[-4]
            curdir = os.path.expandvars(config.Continue_path)[-3]

            latest_file_or_folder = get_latest_folder(config.Continue_path)

            self.gen = int(latest_file_or_folder[-2:])
            self.Population = pickle.load(open(latest_file_or_folder+'/Population.pkl','rb'))
            self.Pop_fitness = np.loadtxt(latest_file_or_folder+'/fitness.txt')
            self.set_dir()

        for x_individual in self.Population:
            self.Archive.append(x_individual.Dec)

    def Evaluation(self,Population):
        if self.config.parallel_evaluation and self.config.n_gpu>1:
            fitness =[]
            for i in range(0,len(Population),self.config.n_gpu):
                # one GPU for one solution executed in one thread
                logging.info('solution:{0:>2d} --- {1:>2d}(Parallel evaluation)'.format(i,i+self.config.n_gpu-1))

                solution_set = Population[i:i+self.config.n_gpu]
                self.Para_Evaluation(solution_set)

            fitness = [x.fitness for x in Population]
            fitness = np.array(fitness)
        else:
            # evaluation in Serial model
            fitness = np.zeros((len(Population),2))
            for i,solution in enumerate(Population):
                # solution = Population[66]
                solution.evaluation(self.config.device_ids)
                fitness[i] = solution.fitness

        return 1.0-fitness

    def Para_Evaluation(self,solution_set):
        thread = [MyThread(solution.evaluation, args=(id,)) for id, solution in enumerate(solution_set)]
        #---------------------------------------
        # (1):execute each thread, but some error(block) may appear due to same dataloader sub-thread are called
        # A = [x.start() for x in thread]
        #---------------------------------
        # (2):wait several seconds after starting each thread
        # to avoid same dataloader sub-thread are used
        for x in thread:
            x.start()
            time.sleep(3)
        # ---------------------------------------
        # synchronize all threads for (returning outputs)/get final outputs
        A = [print(x.is_alive()) for x in thread]
        B = [x.join() for x in thread]
        # C = [x._stop() for x in thread]
        # del A,B,C,thread
        del  A,B,thread
        gc.collect()


    def MatingPoolSelection(self):
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue,
                                                             self.CrowdDistance)
    def Genetic_operation(self):



            Offspring = Generate_crossover_mutation_MOAZ(self.MatingPool,self.gen)   # Crossover


            #--------------------------------Mutation ---------------------------------------

            self.offspring = []
            idx=0
            for Selected_fit_best in Offspring:
                offspring_i_Dec = Generate_AZ(Selected_fit_best,self.gen)
                offspring_i = Individual(Dec=offspring_i_Dec,mapping=self.mapping,config=self.config,gen=self.gen,id=idx)
                idx = idx+1
                self.offspring.append(offspring_i)


            self.Archive.extend(self.offspring)
            self.off_fitness = self.Evaluation(self.offspring)


    def First_Selection(self,Population,Fitness):
        pass



    def EvironmentSelection(self):
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)
        FunctionValue = np.vstack((self.Pop_fitness, self.off_fitness))

        Population, FunctionValue, FrontValue, CrowdDistance, select_index = F_EnvironmentSelect. \
            F_EnvironmentSelect(Population, FunctionValue, self.Popsize)

        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index


    def print_logs(self,since_time=None,initial=False):
        if initial:

            logging.info('********************************************************************Initializing**********************************************')
            print('********************************************************************Initializing**********************************************')
        else:
            used_time = (time.time()-since_time)/60

            logging.info('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                         '*****************************************'.format(self.gen+1,self.Maxi_Gen,used_time))

            print('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                  '*****************************************'.format(self.gen+1,self.Maxi_Gen,used_time))

    def set_dir(self,path=None):
        if path is None:
            path = self.gen
        self.whole_path = "{}/Gen_{}/".format(self.config.exp_name, path)

        if not os.path.exists(self.whole_path):
            os.makedirs(self.whole_path)

    def Save(self):
        # return
        fitness_file = self.whole_path + 'fitness.txt'
        np.savetxt(fitness_file, self.Pop_fitness, delimiter=' ')

        Pop_file = self.whole_path +'Population.txt'
        with open(Pop_file, "w") as file:
            for j,solution in enumerate(self.Population):
                file.write('solution {}: {} \n'.format(j, solution.Dec))

        for i,solution in enumerate(self.Population):
            solution.visualization(self.whole_path+str(i)+'_')

        #------------save as pkl for re-loading------------
        name =  self.whole_path +'Population.pkl'
        f = open(name,'wb')
        pickle.dump(self.Population,f)
        f.close()



    def Plot(self):
        if self.config.parallel_evaluation:
            return
        plt.clf()
        plt.plot(1-self.Pop_fitness[:,0],1-self.Pop_fitness[:,1],'o')
        # plt.xlabel('ACC')
        plt.xlabel('FLOPs')
        plt.ylabel('AUC')
        plt.title('Generation {0}/{1} \n best ACC: {2:.4f}, best AUC: {3:.4f}'.format(self.gen+1,self.Maxi_Gen,max(1-self.Pop_fitness[:,0]), max(1-self.Pop_fitness[:,1])) )
        # plt.show()
        plt.pause(0.2)
        plt.savefig(self.whole_path+'figure.jpg')

    def Main_Loop(self):


        # plt.ion()
        since_time = time.time()
        self.print_logs(initial=True)
        self.Initialization()
        self.Plot()

        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.Popsize)[0]
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)

        while self.gen<self.Maxi_Gen:
            self.set_dir()
            self.print_logs(since_time=since_time)

            self.MatingPoolSelection()
            self.Genetic_operation()
            self.EvironmentSelection()

            self.Save()
            self.Plot()
            self.gen += 1

        # plt.ioff()


if __name__ == '__main__':
    config = get_common_search_config()
    #
    # fix random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    #--------------------------------

    EA = MOAZ(config)
    EA.Main_Loop()


