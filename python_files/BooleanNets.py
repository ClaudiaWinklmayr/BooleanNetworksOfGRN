import numpy as np
from python_files import EntropyMeasures as em

import networkx as nx
from itertools import product
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import json
from functools import partial


#=============================================================================
# THE NODE CLASS
#============================================================================

def update_from_lambda(update_lambda, inputs): 
    return update_lambda(*inputs)

def update_from_dict(update_dictionary, inputs): 
    return  update_dictionary[inputs]

def update_from_list(input_list, output_list, inputs):
    return output_list[input_list.index(inputs)]

class node(): 
    '''
    The node class represents the elements (nodes) in a Boolean network. Each node is characterized by it's ID, 
    it's current value (0 or 1), a list of it's parents and the update function that will change the node value 
    depending on the values of it's parents. The node class is used by the main class 'Boolean network'. 
    '''
    
    ID = None #node id, usually string or int
    Value = None # the node's current state: either 0 or 1 
    InputIDs = [] # list of IDs of the node's parents
    UpdateFunc = {} # how the node's state will change depending on the parent's states. The update function can be 
                    # a lambda expression or a look-up table.
    UpdateType = None #flag specifying the type of update function can be 'dict', 'function' or 'list'

    
    def select_output(self, flag, update_lambda=None, update_dictionary={}, list_update_dictionary={}): 
        if flag == 'dict': 
            self.UpdateFunc = partial(update_from_dict, update_dictionary)
            
        elif flag == 'function':
            self.UpdateFunc = partial(update_from_lambda, update_lambda)
            
        elif flag == 'list':
            self.UpdateFunc = partial(update_from_list, list_update_dictionary['inputs'], list_update_dictionary['outputs'])
        
        else: 
            print('update type unknown')
            
        self.UpdateType = flag
        
    def update(self, inputs):
        # calculates the node's current state depending on the states of its parents. 
        self.Value = self.UpdateFunc(inputs)
        
    def assign_new_update_function(self, output_list): 
        # assigns a new update function to a node using the list of output values. This function presumes 
        # the list to be ordered according to the binary values of the input states. e.g if output_list = [0,0,0,1]
        # the new update function will be {(0,0):0, (0,1):0, (1,0):1, (1,1):1} which corresponds to the AND function.
        
        if self.UpdateType != 'list': 
            print('Assignment for UpdateType {} currently not implemented'.format(self.UpdateType))
            return None
        else: 
            if len(output_list)!= 2**len(self.InputIDs): 
                print('The specified output has length {} but {} values are required!'.format(len(output_list),  2**len(self.InputIDs)))
                return None
                  
            else: 
                Inputs = np.array(list(product((0,1), repeat=len(self.InputIDs))))
                self.UpdateFunc = partial(update_from_list, 
                                          list(tuple(Inputs[i,:]) for i in range(Inputs.shape[0])), 
                                          output_list)
                  
                    
    def __init__(self, ID, Value, InputIDs, UpdateFunc, UpdateType):
        #print(UpdateFunc)
        self.ID = ID
        self.Value = Value

        self.InputIDs = InputIDs
        self.select_output(UpdateType, update_lambda=UpdateFunc, 
                                        update_dictionary=UpdateFunc, 
                                        list_update_dictionary=UpdateFunc)
        

#------------------------------------------------------------------
# Helper Functions for random network creation
#------------------------------------------------------------------



def selectUpdateK1(distribution, parameters=[], FuncFile="FunctionTableK1.csv"):
    '''
    This function will select a Boolean function with k=1 inputs unsing the distribution definition in 
    B.Drossel: "Random Boolean Networks" (2007). 
    '''
    FunctionTableK1 = pd.read_csv(FuncFile)
    functions = ["F1","F2","R1","R2"]

    if distribution == "uniform":
        col = np.random.choice(functions)

    elif distribution == "biased":
        p = parameters[0]
        probs = []
        for func in functions: 
            prob = (p**(sum(FunctionTableK1[func].values)))*((1-p)**(2-sum(FunctionTableK1[func].values)))
            probs.append(prob)
        col = np.random.choice(functions,  p=probs)

    elif distribution == "weighted": 
        delta = parameters[0]
        if delta > 0.5: 
            print('delta must be < 0.5')
            return False
        probs = []
        for func in functions: 
            if func.find("F") > -1: 
                probs.append(delta)
            else:
                probs.append(0.5- delta)
        col = np.random.choice(functions,  p=probs)

    else: 
        print("distribution not known, options are: uniform, biased, weighted")
        return

    update = {eval(FunctionTableK1["Input"][0]):FunctionTableK1[col][0], 
              eval(FunctionTableK1["Input"][1]):FunctionTableK1[col][1]}

    return update, col
    
    
    
def selectUpdateK2(distribution, parameters=[],FuncFile="FunctionTableK2.csv"):
    '''
    This function will select a Boolean function with k=2 inputs unsing the distribution definition in 
    B.Drossel: "Random Boolean Networks" (2007). 
    '''
    FunctionTableK2 = pd.read_csv(FuncFile)
    functions = ['F1', 'F2', 'C1_1', 'C1_2', 'C1_3', 'C1_4', 'C2_1', 'C2_2',
   'C2_3', 'C2_4', 'C2_5', 'C2_6', 'C2_7', 'C2_8', 'R1', 'R2']
    func_types = ["F", "C1", "C2", "R"]

    if distribution == "uniform":
        col = np.random.choice(functions)

    elif distribution == "biased":
        p = parameters[0]
        probs = []
        for func in functions: 
            prob = (p**(sum(FunctionTableK2[func].values)))*((1-p)**(2-sum(FunctionTableK2[func].values)))
            probs.append(prob)
        col = np.random.choice(functions,  p=probs)

    elif distribution == "weighted": 
        function_type_probs = parameters
        func_type = np.random.choice(func_types, p=function_type_probs)
        sub_types = [func for func in func_types if func.find(func_type)>=0]
        col = np.random.choice(sub_types)

    else: 
        print("distribution not known, options are: uniform, biased, weighted")
        return

    update = {eval(FunctionTableK2["Input"][0]):FunctionTableK2[col][0], 
              eval(FunctionTableK2["Input"][1]):FunctionTableK2[col][1], 
              eval(FunctionTableK2["Input"][2]):FunctionTableK2[col][2], 
              eval(FunctionTableK2["Input"][3]):FunctionTableK2[col][3]} 

    return update, col

def check_for_copy_or_const(Output, Controls): 
    '''
    Checks if the Boolean Function specified in the Output array is identical 
    to any of the arrays contained in the list Controls.
    '''
    
    for c in Controls: 
        if sum(Output-c)==0:
            return True
    return False


def select_random_function_under_constraint(n_inputs, p1, notConstant_notCopy=True):
    '''
    Based on the number of input varaibles this ffunction choses a random boolean output function, which will yield
    1 with probability p1. If notConstant_notCopy is True, functions that are either constant or copy one of the inputs 
    will be discarded
    
    n_inputs................ number of input variables
    p1...................... pribability of output taking value 1
    notConstant_notCopy..... if true output functions will be discarded if they are either 
                             constant or simply copy one of the inputs
    '''
    
    Inputs = np.array(list(product((0,1), repeat=n_inputs))) # genearte all possible input value combinations
    output= (np.random.rand(Inputs.shape[0])<p1).astype(int) # choose random output function
    
    if notConstant_notCopy and (n_inputs>1): 
        
        if (p1==1) or (p1==0): #in this cases the output will necessarily be constant
            print('The probability of output value 1 is {}. Cannot make non-constant function'.format(p1))
            return None
        
        controls = [Inputs[:, i] for i in range(Inputs.shape[1])]
        controls.append(np.ones(Inputs.shape[0]))
        controls.append(np.zeros(Inputs.shape[0]))
        
        isConstantOrCopy = check_for_copy_or_const(output, controls)
        counter = 0 # to make sure that the while loop doesn't run forever
        while isConstantOrCopy and (counter<1000): 
            output= (np.random.rand(Inputs.shape[0])<p1).astype(int)
            isConstantOrCopy = check_for_copy_or_const(output, controls)
            counter +=1 
        if counter == 1000: 
            print('I tried {} times to find a non-copy non-constant function. Please check parameters'.format(counter))
            return None
        
            
    return list(tuple(Inputs[i,:]) for i in range(Inputs.shape[0])), list(output)


def get_network_info_from_graph(Graph):  
    '''
    Uses a networkx graph to build a Boolean network. 
    '''
    node_IDs = list(Graph.nodes)
    
    node_parents = {}
    for ni in node_IDs: 
        parents = list(Graph.predecessors(ni))
        if len(parents)== 0: 
            node_parents[ni]=[ni]
        else:     
            node_parents[ni]=parents
    return node_IDs, node_parents

#------------------------------------------------------------------
#------------------------------------------------------------------





class BooleanNetwork(): 
    '''
    The BooleanNetwork class represents a Boolean network as a dictionary where the keys are the node IDs and the 
    values are node-objects (see classs "node"). The main purpose of this class is to build a random Boolean network
    or read in a predefined network from a file. The network can then be drawn, updated or scanned for fixed points.
    '''
    
    nodeDict = {} #dictionary: {node ID: node-object}
    InitialValues = {}  #dictionary: {node ID: intial node value}
    CurrentValues = {} #dictionary: {node ID: current node value}
    Graph = None  # networkx object represnting the connections among the nodes 
    Positions = {} # dictionary: {node ID: (x_position, y_position)}
    UpdateSequences = {} # cache for the calc_all_updates function
    UpdateType = None # can be 'function' or 'dict
    
    def clear(self): 
        self.nodeDict = {} 
        self.nodeIDs = []
        self.InitialValues = {}
        self.CurrentValues = {}
        self.Graph = None 
        self.Positions = {}
        self.UpdateSequences = {}
        self.UpdateType = None
        
    def initialize_from_file(self, filename):
        '''
        Initializes a Boolean Network from a json file specification. A valid specifictaion is a dictionary
        with the following keys:
        
        IDs............. List containing the names of the nodes
        Coonnections.... Dictionary with the node Id as key and a list of the node's parants as value
        Updates......... Dictionary with the node Id as key and the update function as value. 
                         Update function can be a lambda expression in string format or 
                         a dictionary with the input values as keys and the corresponding outputs as values
        Positions....... Dictionary with the node Id as key and a list containing the nodes position in xy-space
                         If Positions is not specified positions will be generated automatically 
                         placing nodes on a circle with radius 1
        '''
        
        if filename.find('json') <0: 
            print('unknown file extension')
            return
        else: 
            loadnet= json.load(open(filename, 'r'))
            IDs = loadnet["IDs"]
            Connections = loadnet["Connections"]
            Updates = {key: eval(loadnet["Updates"][key]) for key in loadnet["Updates"]}
            if "Positions" in loadnet.keys(): 
                Positions = loadnet["Positions"]
            else: 
                Positions = {}
            if type(Updates[list(Updates.keys())[0]]) == dict: 
                UpdateType = 'dict'
            elif hasattr(Updates[list(Updates.keys())[0]], '__call__'):
                UpdateType = 'function'
            else: 
                print("unknown update strategy")
                return
            
            return IDs, Connections, Updates, Positions, UpdateType
        
        
    def initialize_sparse_random_net(self, k, N, ):
        '''
        Initializes a sparse random boolean net following the procedure descirbed in 
        B.Drossel: "Random Boolean Networks" (2007). Each node will have either 1 or 2 parents and the update functions 
        are drawn according to a distribution as specified in UpdateFunctionDistr. The paarmeters are: 
        
        k............. number of parents each node has
        N............. number of nodes in the network
        FuncFile...... file containing the possible update functions (FunctionTableK1.csv or FunctionTableK2.csv)
        UpdateFunctionDistr...string specifying how to choose the input functions. options are: 
                uniform........ input functions are chosen from a uniform distribution
                biased......... input functions are chosen based on the number of ones
                weighted....... input functions are chosen based on the class of functions 
        '''
        
        assert k  in [1,2]
        
        FuncFile = SparseRandomParams['FuncFile']
        UpdateFunctionDistr = SparseRandomParams['UpdateFunctionDistr']
        UpdateFunctionParams = SparseRandomParams['UpdateFunctionParams']
           
        NodeIDs = [i for i in range(N)] 
        
        # randomly choose k inputs for each node. Do not allow self loops
        NodeInputs = [np.random.choice(list(set(NodeIDs).difference({NodeIDs[i]})), k, replace=False) for i in range(N)]
        print("{} inputs per node were chosen at random".format(k))

        if k == 1: 
            ff= FuncFile if len(FuncFile)>0 else "FunctionTableK1.csv"
            NodeFunctions = [selectUpdateK1(UpdateFunctionDistr, UpdateFunctionParams, FuncFile=ff)[0] for i in range(N)]
            UpdateType = 'dict'
            print("update functions were chosen via {} distribution using parameters {}".format(UpdateFunctionDistr, UpdateFunctionParams))
            
        if k == 2: 
            ff= FuncFile if len(FuncFile)>0 else "FunctionTableK2.csv"
            NodeFunctions = [selectUpdateK2(UpdateFunctionDistr, UpdateFunctionParams, FuncFile=ff)[0] for i in range(N)]
            UpdateType = 'dict'
            print("update functions were chosen via {} distribution using parameters {}".format(UpdateFunctionDistr, UpdateFunctionParams))
                        
           
        return NodeIDs, NodeInputs, NodeFunctions, UpdateType
   
    #def select_random_function_under_constraint(M, NoCopyFunctions): 
    
    def initialize_random_net(self, k, N, SigmaNeighbors, RandomNetworkParams, verbose=True): 
        
        '''
        Initializes a random boolean net where the update functions are chosen randomly
        '''
        
        if not RandomNetworkParams['Graph'] == None: 
            if not RandomNetworkParams['Graph'].is_directed():
                print(RandomNetworkParams['Graph'].is_directed())
            #if not type(RandomNetworkParams['Graph']) == nx.classes.digraph.DiGraph: 
                print('The Graph hass to be directed')
                return None
                      
            NodeIDs, NodeInputs = get_network_info_from_graph(RandomNetworkParams['Graph'])
            Nneighbors = [len(NodeInputs[ni]) for ni in NodeInputs]
            
        else: 
            
            NodeIDs= [i for i in range(N)] 

            #generate number of neighbors
            Nneighbors = abs(np.random.normal(k, SigmaNeighbors, N)).astype(int)
            Nneighbors[np.where(Nneighbors>N)] = N
            
            if RandomNetworkParams['NoIsolatedNodes']:  
                Nneighbors[np.where(Nneighbors<1)]=1

            if RandomNetworkParams['NoSelfLoops']:
                Nneighbors[np.where(Nneighbors==N)] = N-1
                NodeInputs = [np.random.choice(list(set(NodeIDs).difference({NodeIDs[i]})), 
                                               Nneighbors[i], replace=False) for i in range(N)]
            else:    
                NodeInputs = [np.random.choice(NodeIDs, Nneighbors[i], replace=False) for i in range(N)]

        #NodeFunctions = [{'inputs': list(k for k in product((0,1), repeat=Nneighbors[i])), 
        #                  'outputs':list((np.random.rand(2**Nneighbors[i])<0.5).astype(int))}
        #                  for i in range(N)]
        NodeFunctions = {}
        for nIDX, nID in enumerate(NodeIDs): 
            inps, outps = select_random_function_under_constraint(Nneighbors[nIDX], 
                          RandomNetworkParams['p1'], RandomNetworkParams['notConstant_notCopy'])
            #print(inps, outps)
            NodeFunctions[nID] = {'inputs':inps , 'outputs':outps}                 
        
        
        if verbose: 
            print("I built a random network: using the following parameters: ")
            for key in RandomNetworkParams: 
                print(key + ':' + str(RandomNetworkParams[key]))

        UpdateType = 'list'

        return NodeIDs, NodeInputs, NodeFunctions, UpdateType


    
    def __init__(self, k=3, N=5, SigmaNeighbors=0, NodeIDs=[], NodeInputs={}, NodeFunctions={}, InitialValues={}, 
                 InitialValuesDistribution=0.5, FileName="", Positions={},
                 SparseRandomParams={'UpdateFunctionDistr':'uniform', 'UpdateFunctionParams':[], 'FuncFile':''}, 
                 RandomNetworkParams={'Graph':None, 'NoIsolatedNodes':True, 
                                      'NoSelfLoops':True, 'p1':0.5, 'notConstant_notCopy':True, 'rootNodesConstant':True}):
        '''
           k.............. number of nearest neighbors. If SigmaNeighbors>0, k is interpreted as the mean of a normal distribution
           N.............. number of nodes in the network
           SigmaNeighbors. variability of each node's number of neighbors
           NodeIDs........ list containing the a unique identifier for each node (typically a string). If the list is empty an integer identifyer is created for each node.
           NodeInputs..... list of lists containing each node's parents
           NodeFunctions.. list of update functions for each node. 
           InitialValues.. list containing the initial values of each node, possible values are 1 and 0
           InitialValuesDistribution...  fraction of 1's in the initialization, default is 0.5
           FileName....... name of a file containing a network specifictaion
           Positions...... dictionary conataining the position of each node if empty nodes will be placed on a ring with radius one 
        '''
        self.clear()
        SparseRandomParams_internal = {'UpdateFunctionDistr':'uniform', 'UpdateFunctionParams':[], 'FuncFile':''}
        SparseRandomParams_internal.update(SparseRandomParams)
        
        RandomNetworkParams_internal = {'Graph':None, 'NoIsolatedNodes':True, 
         'NoSelfLoops':True, 'p1':0.5, 'notConstant_notCopy':True, 'rootNodesConstant':True}
        RandomNetworkParams_internal.update(RandomNetworkParams)
        
        if len(FileName)>0:
            NodeIDs, NodeInputs, NodeFunctions, Positions, UpdateType = self.initialize_from_file(FileName)
            print("initializing network from file: {}".format(FileName)) 
        
        elif (len(NodeIDs) >0):
            if (len(NodeIDs) == len(NodeInputs) == len(NodeFunctions)): 
                UpdateType = type(NodeFunctions[NodeIDs[0]]).__name__
                print('Built network from specifications')
            
        else: 
            if k < 3 and SigmaNeighbors==0: 
                NodeIDs, NodeInputs, NodeFunctions, UpdateType = self.initialize_sparse_random_net(k, N, SparseRandomParams=SparseRandomParams_internal)
                
                print("building random sparse network with {} inputs per node and {} distribution of update functions".format(k, SparseRandomParams_internal['UpdateFunctionDistr']))
                
            else: 
                NodeIDs, NodeInputs, NodeFunctions, UpdateType = self.initialize_random_net(k, N, SigmaNeighbors, RandomNetworkParams_internal)


            
        if len(InitialValues) == 0: 
            InitialValues = list((np.random.rand(len(NodeIDs))<InitialValuesDistribution).astype(int))
            print("initial values were chosen such that P(1)={}".format(InitialValuesDistribution))
            
            
        if len(NodeIDs) == 0: 
            #if not specified otherwise, NodeIds is a list of integers
            NodeIDs = [i for i in range(N)] 
            print("node IDs specified: {}".format(NodeIDs))     
            
        
        G=nx.DiGraph()
        G.add_nodes_from(NodeIDs)
        self.UpdateType=UpdateType
        for nIDX, nID in enumerate(NodeIDs): 
            self.InitialValues[nID] = InitialValues[nIDX]
            self.CurrentValues[nID] = InitialValues[nIDX]
            self.nodeDict[nID] = node(nID, InitialValues[nIDX], NodeInputs[nID], NodeFunctions[nID], UpdateType)
            
            for inp in NodeInputs[nID]: 
                G.add_edge(inp, nID)
        self.Graph = G
        
        if len(Positions) == 0: 
            self.Positions = self.generate_positions(len(NodeIDs))
        else: 
            self.Positions = Positions
            
        self.NodeIDs = NodeIDs

     
    def assign_values_to_nodes(self, value_dict={}, value_list=[], p=0.5): 
        ''' 
        This function assigns values to nodes in the network. Values can be specified deterministically for (i)selected nodes using a dictionary, (ii) for all node,using a list or (iii) randomly by specifying only the parameter p of a Bernoulli distribution. 
        Inputs: 
        value_dict...... a dictionary conataining IDs of specific nodes and the corresponding values
        value list...... list specifying a value for each node in the network
        p............... parameter for a Bernoullli distribution where values are drawn from. p specifies the probability of assigning the value one.
        '''
        
        if bool(value_dict): # if dictionary is given assign values            
            for nodeID, nodeValue in value_dict.items(): 
                self.nodeDict[nodeID].Value = nodeValue
                self.CurrentValues[nodeID] = nodeValue
        
        elif len(value_list)>0:
            if len(value_list) < len(self.nodeDict): 
                print("not enough values specified. {} values were given, but the network has {} nodes".format(len(value_list) , len(self.nodeDict)))
            for nodeIDX, nodeID in enumerate(self.nodeDict):
                self.nodeDict[nodeID].Value = value_list[nodeIDX]
                self.CurrentValues[nodeID] = value_list[nodeIDX]

        else:   # assign values randomly
            for nodeID in self.nodeDict: 
                val = int(np.random.rand()<p)
                self.nodeDict[nodeID].Value = val
                self.CurrentValues[nodeID] = val
                
                
    def assign_update_functions(output_lists): 
        '''
        takes a list of lists containing an outputs for each nodes and assignes each node in the
        network a corresponding output function unsing the assign_new_update_function() propery of the node class.
        '''
        for key in output_lists: 
            self.nodeDict[key].assign_new_update_function(output_lists[key])
                
                
    def update_all(self, FixedNodes={}):
        ''' 
        Performs synchronous update for all nodes in the network. 
        The parameter FixedNodes is a dictionary containing nodeIds and the value these nodes are fixed at. 
        If FixedNodes is not empty the fixed nodes will not be updated
        
        '''
        
        Values_at_timeT = self.CurrentValues.copy()
        
        for nodeID in FixedNodes: 
            self.assign_values_to_nodes(value_dict=FixedNodes)
            
        nodes_to_update = [node_name for node_name in self.nodeDict.keys() if (not node_name in FixedNodes.keys())]
            
        for nID in nodes_to_update:           
            input_vals = tuple([Values_at_timeT[inp] for inp in self.nodeDict[nID].InputIDs])
            self.nodeDict[nID].update(input_vals)
            
        self.getCurrentValues()
            

    def getCurrentValues(self): 
        '''
        Updates the network's CurrentValues dictionary by iterating through all nodes and reading their Value
        '''
        for nID in self.nodeDict.keys():
            self.CurrentValues[nID] = self.nodeDict[nID].Value

        
    
    def generate_positions(self, N, radius = 1):  
        ''' 
        assigns each node a position to allow darwing of the network via the networkx.darw() method.
        If not otherwise specified, nodes are placed on a circle with radius 1
        '''
        pos = {}
        for n, nodeID in enumerate(self.nodeDict): 
            alpha = 2*np.pi*n/float(N)
            pos[nodeID] = np.array([radius*np.cos(alpha), radius*np.sin(alpha)])

        return pos

    
    
    def update_until_cycle(self, FixedNodes={}, verbose=True):
        '''
        Starting from the current values and possibly specified fixed points calculate updates until
        either a fixed point or a cycle is found. If verbose == True the result will be printed on the screen. 
        Returns: Start value, the length of the cycle and the values of the cycle (which might not include the start value).
        '''
        
        find_repeat = False 
        UpdateSequence = []
        self.assign_values_to_nodes(value_dict=FixedNodes)
        UpdateSequence.append(tuple(self.CurrentValues.values()))
        
        while not find_repeat: # update until the sequence starts to repeat
            self.update_all(FixedNodes=FixedNodes)
            UpdateSequence.append(tuple(self.CurrentValues.values()))
            if len(UpdateSequence) != len(list(set(UpdateSequence))): 
                find_repeat = True
        
        cycle_length = 0
        if UpdateSequence[-1] == UpdateSequence[-2]:
            cycle = UpdateSequence[-1]
            if verbose: 
                print('found fixed point: {}'.format(cycle))
        else: 
            cycle_start = UpdateSequence.index(UpdateSequence[-1])
            cycle_length = len(UpdateSequence)- 1- cycle_start
            cycle = UpdateSequence[cycle_start:]
            if verbose:
                print('found circle of length {}:\n{}'.format(cycle_length, cycle))  
                
        return UpdateSequence[0], cycle_length, cycle
    

    
    def scan_state_space(self, FixedNodes={}, StartNodes={}):
        '''
        Scans the state space for fixed points and cycles. Nodes listed in the FixedNodes dictionary will be kept
        fixed throughout the scan i.e the connections to their parents will be cut. Nodes listed in the StartNodes dictionary 
        will at the start of each scan round be set to the same value but are allowed to change during updates. This is 
        equivalent to a conditional scan, effecctively reducing the numbers of start values to check
        
        Output: 
        cycle_dict.... a dictionary where the keys are the lengths of the cycles and 
                       the values are lists of tuples ccorresponding to the values of the cycle.
        
        '''

        N = len(self.nodeDict)-len(FixedNodes)-len(StartNodes)# number of varying nodes
        names = [node_name for node_name in self.nodeDict.keys() 
                 if ((not node_name in FixedNodes.keys()) and (not node_name in StartNodes.keys()))]
        #all possible combinations of node values except those fixed or with a predefned start value
        start_values = list(product(range(2), repeat=N)) 
        
        value_dict = FixedNodes.copy() # empty is nothing is fixed, containing fixed nodes otherwise
        value_dict.update(StartNodes)
        fixed_point_list = []
        cycle_dict = {}
            
        for sv_idx, sv in enumerate(start_values): 
            # counter 
            print(str(round(sv_idx/len(start_values),2)), end='\r')            
            #set the current start values and assign them to the nodes
            for i, x in enumerate(names):
                value_dict[x] = sv[i]
            self.assign_values_to_nodes(value_dict)
            
            start, cycle_length, cycle = self.update_until_cycle(verbose=False, FixedNodes=FixedNodes)
            
            if cycle_length == 0: 
                fixed_point_list.append(cycle)
            else: 
                if cycle_length in cycle_dict.keys():
                    #if a cycle of length k is found we check if it had been discovered before
                    #by checking whether teh first element of the current cycle is an element of any known cycle
                    rep = np.array([cycle[0] in cyc for cyc in cycle_dict[cycle_length]])
                    if not rep.any():
                        cycle_dict[cycle_length].append(cycle)
                else: 
                    cycle_dict[cycle_length] = []
                    cycle_dict[cycle_length].append(cycle)

        cycle_dict[0] = [[fp] for fp in list(set(fixed_point_list))]
        
#        if display: 
#            self.make_phase_space_info(cycle_dict)
        return cycle_dict
    
    
    
 #   def make_phase_space_info(self, cycle_dict):
 #       
 #       for cyc_length in np.sort(list(cycle_dict.keys())): 
 #           if (cyc_length ==0):
 #               if (len(cycle_dict[cyc_length])>0): 
 #                    print("{} fixed points".format(len(cycle_dict[0])))#
 #                   for noID in self.nodeDict.keys():
 #                       print(str(noID) + '\t', end ='')
 #                   print("")
 #                  for entry in cycle_dict[cyc_length]: 
 #                       for fval in entry[0]:
 #                           print(str(fval) + '\t', end ='')
 #                       print("")
 #           else: 
 #               print("{} cycles of length {}: \n {}".format(len(cycle_dict[cyc_length]), cyc_length, cycle_dict[cyc_length]))
 #   
 #       if len(cycle_dict[0]) >0:
 #           FPs = []
 #           for entry in cycle_dict[0]: 
 #               FPs.append(list(entry[0]))
 #           FPs = np.array(FPs)
 #           plt.imshow(FPs, cmap='gray')
 #           NFPs, Nnodes =  FPs.shape
 #
 #           for f in range(NFPs+1): 
 #               plt.plot([-0.5, Nnodes - 0.5], [f-1.5, f-1.5], color='grey')
 #
 #           for nn in range(Nnodes):
 #               plt.vlines(x=nn-0.5, ymin=-0.5, ymax= NFPs -.5, color='grey')
 #
 #           plt.xlim(-0.5, Nnodes - 0.5)
 #          plt.ylim(-0.5,NFPs -.5)
 #           plt.yticks(np.arange(0, NFPs), ['FP{}'.format(x) for x in np.arange(1, NFPs+1)], size=14)
 #           plt.gca().invert_yaxis()
 #           plt.xticks(np.arange(Nnodes), self.NodeIDs, rotation=90, size=14)
 #
 #          plt.show()
        
        
    
 
    def calc_all_updates(self, rounds=1, FixedNodes={}, cheap=False):
        '''
        This function generates all possible network states and calculates the subsequent stae for each of them. 
        Will return two lists containing the input and output states. If the parameter "cheap" is set to True, 
        the function checks if the updates have already been calcualted in which case they are retrieved form the chache.
        Otherwise they will be calculated anew and saved to the cache.
        
        Inputs: 
        rounds ........ number of updates to calculate
        FixedNodes..... dictionary, specifying nodes that will not be changed in an update (i.e nodes whos input links are virtually cut)
        cheap.......... if True the resulating input and output values will be cashed to speed up future calls to this procedure. 
        
        Output: 
        start.......... list of all possible network states as tuples
        stop........... list of the network staes after updating as may times as specified by "rounds" parameter
        '''
        if cheap == False: 
            return self.calc_all_updates_new(rounds=rounds)
        
        start, stop = self.retrieve_update_sequence(rounds, FixedNodes)
        if type(start) == np.ndarray: 
            return start, stop


        # in case the values are not yet saved, retrieve_update_sequence returns the neccessary 
        # identifyers to be used for saving the new results
        res_identifyer = stop
        start_identifyer = start


        N = len(self.nodeDict)-len(FixedNodes) # number of varying nodes
        names = [node_name for node_name in self.nodeDict.keys() if not node_name in FixedNodes.keys()]
        #all possible combinations of node values except those chosen fixed
        start_values = list(product(range(2), repeat=N)) 
        self.assign_values_to_nodes(FixedNodes)# make sure fixed noded take on the fiex value

        result_values = []
        full_start_values = []

        for sv_idx, sv in enumerate(start_values): 
            value_dict = {x:sv[i] for i, x in enumerate(names)}
            self.assign_values_to_nodes(value_dict)
            full_start_values.append(list(self.CurrentValues.values()))

            for i in range(rounds):
                self.update_all(FixedNodes=FixedNodes)

            result_values.append(list(self.CurrentValues.values()))

        start = np.array(full_start_values)
        stop = np.array(result_values) 
        
        
        self.UpdateSequences[start_identifyer]= start
        self.UpdateSequences[res_identifyer]=stop
            
        return start, stop  
    
    

    def retrieve_update_sequence(self, rounds, FixedNodes): 
        '''
        Helper function for calc_all_updates which checks if a full set of updates has been already 
        calculated in which case it is retrieved form storage. For documentation on the functionality of calc_all_updates
        see the corresponding docstring. 
        '''
        
        res_identifyer='Result_rounds{}'.format(rounds)
        start_identifyer = 'Start'
        sorted_names = list(FixedNodes.keys())
        sorted_names.sort()
        
        if len(FixedNodes)>0:
            res_identifyer +='_fixed_'
            start_identifyer +='_fixed_'
            for node in sorted_names: 
                res_identifyer += str(node) + '_'
                start_identifyer+= str(node) + '_'
                
        if (res_identifyer in self.UpdateSequences.keys()) and (start_identifyer in self.UpdateSequences.keys()): 
            start = self.UpdateSequences[start_identifyer]
            stop = self.UpdateSequences[res_identifyer]            
            return start, stop 
        
        else: 
            return start_identifyer, res_identifyer
        
    def calc_all_updates_new(self, rounds=1, FixedNodes={}): 
        '''
        Same functionality as calc_all_updates but without the ability to save and retrieve previously        
        calculated values form cache. For documentation see the doctring of calc_all_updates.
        '''
        
        N = len(self.nodeDict)-len(FixedNodes) # number of varying nodes
        names = [node_name for node_name in self.nodeDict.keys() if not node_name in FixedNodes.keys()]
        #all possible combinations of node values except those chosen fixed
        start_values = list(product(range(2), repeat=N)) 
        self.assign_values_to_nodes(FixedNodes)# make sure fixed noded take on the fixed value

        result_values = []
        full_start_values = []

        for sv_idx, sv in enumerate(start_values): 
            value_dict = {x:sv[i] for i, x in enumerate(names)}
            self.assign_values_to_nodes(value_dict)
            full_start_values.append(list(self.CurrentValues.values()))

            for i in range(rounds):
                self.update_all(FixedNodes=FixedNodes)

            result_values.append(list(self.CurrentValues.values()))

        start = np.array(full_start_values)
        stop = np.array(result_values) 
        
        return start, stop  

    

   
    def draw_pretty(self, FixedNodes={}, SinglePlot=True, SaveName=''): 
        colors = ['lightsteelblue', 'salmon']
        colorlist=[]
        lws = []
        
        for key, val in self.CurrentValues.items():
            colorlist.append(colors[val])
            if key in FixedNodes: 
                lws.append(3)
            else: 
                lws.append(0)

        if SinglePlot: 
            plt.figure()
            
        nx.draw_networkx_nodes(self.Graph, pos=self.Positions, linewidths=lws, 
                               edgecolors='k',node_color=colorlist, node_size=600)#, linewidth=lws)

        for idx, pair in enumerate(self.Graph.edges):
            if not pair[1] in FixedNodes: 
                edge = nx.draw_networkx_edges(self.Graph, pos=self.Positions, width=1.50, arrows=True, arrowsize=15, 
                                         edgelist=[pair], edge_color='grey', style='solid', alpha=1, node_size=600)

                edge[0].set_connectionstyle("arc3,rad=0.1")
        nx.draw_networkx_labels(self.Graph, pos=self.Positions,  font_size=12, font_color='k')  
            
        plt.axis("off")
        plt.tight_layout()
        
        #
        if len(SaveName)>0: 
            plt.savefig(SaveName)
        elif SinglePlot: 
            plt.show()
            
            
    def draw(self, RemoveNodes=[], positions={}): 
        colors = ['lightsteelblue', 'salmon']
        colorlist=[]
        for key, val in self.CurrentValues.items():
            if key in RemoveNodes: 
                colorlist.append('black')
            else: 
                colorlist.append('lightgrey')
                #colorlist.append(colors[val])

        bi_dir = []
        for edge in self.Graph.edges: 
            if edge[::-1] in self.Graph.edges: 
                bi_dir.append(edge)
                
        if (len(positions)==0):
            if len(self.Positions)==0:
                print('Node positions were not provided so I generated positions randomly!')
                pos = self.generate_positions(len(self.nodeDict))
            else: 
                pos=self.Positions
                
        

        #nx.draw_networkx_nodes(self.Graph, pos=self.Positions,edgecolors='k',node_color=colorlist, node_size=600)
        #nx.draw_networkx_nodes(self.Graph, pos=pos,edgecolors='k',node_color=colorlist, node_size=600)
        nx.draw_networkx_nodes(self.Graph, pos=pos,edgecolors='k', node_size=600)


        for idx, pair in enumerate(self.Graph.edges):
            #edge = nx.draw_networkx_edges(self.Graph, pos=self.Positions, width=1.50, arrows=True, arrowsize=15, 
            #                             edgelist=[pair], edge_color='grey', style='solid', alpha=1, node_size=600)
            edge = nx.draw_networkx_edges(self.Graph, pos=pos, width=1.50, arrows=True, arrowsize=15, 
                                         edgelist=[pair], edge_color='black', style='solid', alpha=1, node_size=600)
                                 
            if pair in bi_dir: 
                edge[0].set_connectionstyle("arc3,rad=0.1")
                
        nx.draw_networkx_labels(self.Graph, pos=pos,  font_size=12, font_color='k')  
        
        plt.axis('off')


        
#    def draw_VelizCuba_net(self, FixedNodes={}): 
#        colors = ['lightsteelblue', 'salmon']
#        plt.figure()
#        colorlist=[]
#        lws = []
#        
#        for key, val in self.CurrentValues.items():
#            colorlist.append(colors[val])
#            if key in FixedNodes: 
#                lws.append(3)
#            else: 
#                lws.append(0)
#
#        nx.draw_networkx_nodes(self.Graph, pos=self.Positions, linewidths=lws, 
#                               edgecolors='k',node_color=colorlist, node_size=600)#, linewidth=lws)
#
#        C_styles = {('C', 'M'):"bar,angle=180,fraction=0", ('Ge', 'C'):"arc3,rad=-0.2", 
#                    ('Rm', 'M'):"arc3,rad=-0.2", ('R', 'M'):"arc3,rad=-0.2", 
#                    ('R', 'Rm'):"arc3,rad=-3", ('L', 'A'):"arc3,rad=-0.1", 
#                    ('L', 'Am'):"arc3,rad=-0.15", ('Ge', 'L'):"arc3,rad=0.2", 
#                    ('Ge', 'Lm'):"arc3,rad=0.2", ('Lm', 'Am'):"arc3,rad=0.3", 
#                    ('Am', 'R'):"arc3,rad=1.5", ('Am', 'Rm'):"arc3,rad=-1.5"}
#                          
#        for idx, pair in enumerate(self.Graph.edges):
#            if not pair[1] in FixedNodes: 
#                edge = nx.draw_networkx_edges(self.Graph, pos=self.Positions, width=1.50, arrows=True, arrowsize=15, 
#                                         edgelist=[pair], edge_color='grey', style='solid', alpha=1, node_size=600)
#                if pair in C_styles.keys(): 
#                    edge[0].set_connectionstyle(C_styles[pair])
#                else: 
#                    edge[0].set_connectionstyle("arc3,rad=0.1")
#
#        nx.draw_networkx_labels(self.Graph, pos=self.Positions,  font_size=12, font_color='k')  
#        plt.axis("off")
#        plt.tight_layout()


    
    
    
    

