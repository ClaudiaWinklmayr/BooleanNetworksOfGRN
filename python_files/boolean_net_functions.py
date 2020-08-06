from python_files import BooleanNets as bn
from python_files import EntropyMeasures as em
import numpy as np
import networkx as nx
import dit
from itertools import product, combinations
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def tup2str(tup): 
    ''' Converts a tuple to a string to make labels for the statespace graph'''
    string = ''
    for t in tup: 
        string += str(t)
        
    return string

def generate_positions(Graph,radius = 1):  

    pos = {}
    N = len(Graph.nodes)
    for n, nodeID in enumerate(Graph.nodes): 
        alpha = 2*np.pi*n/float(N)
        pos[nodeID] = np.array([radius*np.cos(alpha), radius*np.sin(alpha)])

    return pos




#=============================================================================
# ATTRACTOR FUNCTIONS
#=============================================================================

def update_to_attractor(Net, start_vals = []): 
    ''' Updates a Boolean network until it hits an attractor and returns the resulting update sequence, 
    can either start from the networks current state or explicitly specify a start value'''
    
    if len(start_vals)!=0: 
        Net.assign_values_to_nodes(value_list=start_vals)
    update_seq = []
    stop = False
    while not stop: 
        update_seq.append(tuple(Net.CurrentValues.values()))
        Net.update_all()
        
        if len(update_seq) > len(list(set(update_seq))): 
            stop = True
            return update_seq



def identify_attactors(Net): 
    ''' finds the attractors of a boolean network by recursively removing all nodes without parents i.e 
    all states that cannot be reached from other states.'''
    
    done = False 
    start, stop = Net.calc_all_updates()
    start = [tuple(s) for s in start]
    stop = [tuple(s) for s in stop]
    
    while not done: 
        
        attr_start = []
        attr_stop = []

        for state_idx, state in enumerate(start): 
            if state in stop: 
                attr_start.append(state)
                attr_stop.append(stop[state_idx])

        if len(list(set(attr_start))) == len(list(set(attr_stop))): 
            done =True
        else: 
            start = attr_start
            stop = attr_stop

    return attr_start,  attr_stop


def find_basins_of_attraction(Net, verbose=False):
    
    '''Finds the basin of attraction for each attractor by recursively traversing all incoming states. 
    Returns a dictionary where the keys are the attractors and the values are lists of incoming states.'''
    
    #make attractor list
    cycles = Net.scan_state_space()
    start, stop = Net.calc_all_updates()
    start = [tuple(s) for s in start]
    stop = [tuple(s) for s in stop]
    attractors = []

    for length in cycles.keys(): 
        if length==0: 
            for attractor in cycles[length]: 
                attractors.append((length, attractor[0]))
        else: 
            for attractor in cycles[length]: 
                print(attractor)
                attractors.append((length, attractor[0]))
    
    
    # find basin of attraction for each attractor
    Basins = {}
    for attractor in attractors: 
        done = False
        Basin = [] # list of elements in the basin of attraction
        Inters = [attractor[1]] # each attractor is itsef a member of its basin of attraction
        

        while not done: 
            # add the parents of each node in the basin 
            inter_val = Inters[0]
            indices = [i for i, x in enumerate(stop) if x == inter_val]

            if len(indices) > 0: 
                parents = [start[i] for i in indices]
                Inters.extend(parents)#[p for p in parents if p!= inter_val])
                
            # if the node's parents have been discovered, the node is added to the Basin list and the searh
            # continues with it's parents
            Basin.append(inter_val)    
            Inters = [i for i in Inters if (i != inter_val) and (i not in Basin)]

            if len(Inters) == 0: 
                done = True

        Basins[attractor] = Basin
    if verbose: 
        print('(Attractor length, Attractor elements), Size of Basin')
        for key in Basins: 
            print(key, len(Basins[key]))
        
    return Basins






#=============================================================================
# SENSITIVITY FUNCTIONS
#=============================================================================

def get_Hamming_neighbors(x):
    '''The input x is a binary list or tuple of length N. The function finds all N tuples y, 
    that differ from x in exactly one position'''
    
    x = np.array(x)
    neighbors = []
    for i in range(len(x)):
        y  = x.copy()
        y[i] = 1-x[i]
        neighbors.append(tuple(y))
        
    return neighbors


def Sensitivity(func_dict):
    ''' Calaculates the Sensitivity of a Boolean Function i.e the average number of changes in 
    the output if one of the inputs changes"
    '''
    S = np.zeros(len(func_dict.keys()))   
    for x_idx, x  in enumerate(func_dict.keys()): 
        x_neighbors = get_Hamming_neighbors(x)
        S[x_idx] = sum([abs(func_dict[x]-func_dict[y])for y in x_neighbors])

    return np.mean(S)


def NetworkSensitivity(Net):
    ''' Calculates the Sensitivity of each node in a Boolean Network and returns the result as a dictionary
    where the keys are the nodeIDs and the values are the Sensitivity.
    '''
    Sens = {}
    for nodeID, node in Net.nodeDict.items(): 
        Ninps = len(node.InputIDs)
        func_dict = {}
        inputs = list(product((0,1), repeat=Ninps))
        for inp in inputs: 
            func_dict[inp] = node.UpdateFunc(inp)
        Sens[nodeID] = Sensitivity(func_dict)
    return Sens


#=============================================================================
# CANALYZING FUNCTIONS
#=============================================================================


def is_canalyzing(func_dict):
    ''' checks if a Boolean function in dictionary format is canalzing and returns the index of the canalyzed variable.
    If the function is not canalyzing is_canalyzing will return -1'''
    
    Inputs = list(func_dict.keys())
    Ninps = len(Inputs[0])
    func_array = np.zeros((2**Ninps, Ninps+1))
    
    for inp_idx, inp in enumerate(Inputs): 
        func_array[inp_idx, :Ninps] = inp
        func_array[inp_idx, Ninps] = func_dict[inp]
    #print(func_array)
        
    for k in range(Ninps): 
        idx0 = np.where(func_array[:, k]==0)[0]
        idx1 = np.where(func_array[:, k]==1)[0]
        
        if (func_array[idx0, Ninps]==1).all() or (func_array[idx0, Ninps]==0).all(): 
            return k
        
        elif (func_array[idx1, Ninps]==1).all() or (func_array[idx1, Ninps]==0).all(): 
            return k
        
    return -1

def NetworkCanalyzing(Net):
    ''' Determines for each node in the network if its update function is canalyzing and returns a dictionary where the 
    keys are the nodeIDs and the values are -1 if the update function is not canalyzing, or the index of the canalyzed varaible otherwise.
    '''
    Cana = {}
    for nodeID, node in Net.nodeDict.items(): 
        Ninps = len(node.InputIDs)
        func_dict = {}
        inputs = list(product((0,1), repeat=Ninps))
        for inp in inputs: 
            func_dict[inp] = node.UpdateFunc(inp)
        Cana[nodeID] = is_canalyzing(func_dict)
    return Cana

#=============================================================================
# CLUSTERING FUNCTIONS
#=============================================================================

def find_clusters(Net): 
    '''
    Finds clusters in the network by first calclating the attractor states and then finding sets of nodes that have low
    joint entropy. 
    
    Output: list of tuples, where the first entry is a tuple of node indices and the second entry is the jount entropy
    '''

    a_start, a_stop = identify_attactors(Net)
    nodes = Net.NodeIDs
    # calculate the joint attaractor distribution
    d = dit.Distribution(a_start, [1/len(a_start)]*len(a_start))
    
    entropies = []
    clusters = []
    for pair in combinations(range(len(nodes)), 2): 
        #find all pairs that have entropy lower than 1 
        H = dit.shannon.entropy(d.marginal(pair))
        if H <= 1: 
            entropies.append((pair, dit.shannon.entropy(d.marginal(pair))))

    for pair in entropies: 
        # successively add nodes to the pairs to find larger clausters with low entropy
        a = find_cluster_containing_pair(Net, a_start, pair[0])
        clusters.append(tuple(a[0][0]))
        clusters = list(set(clusters))
        
    return [(c, dit.shannon.entropy(d.marginal(c))) for c in clusters]       

def find_cluster_containing_pair(Net, a_start, pair): 
    '''
    This is a helper function for the find_clusters() function. The funtion takes a pair of nodes as input and 
    successively adds other nodes to find larger clusters with low entropy. 
    '''
    #pair is a list
    nodes = Net.NodeIDs
    d = dit.Distribution(a_start, [1/len(a_start)]*len(a_start))
    stop = False
    entropies = [tuple((pair, dit.shannon.entropy(d.marginal(pair))))]
    length_counter = 3
    
    while not stop: 
        
        print('\r Checking tuples of length: {}   '.format(length_counter), end = '')
        tuple_ent = []
        for H_tuple_idx, H_tuple in enumerate(entropies): 
            for idx in [x for x in range(len(nodes)) if x not in H_tuple[0]]: 
                new_tup = [x for x in H_tuple[0]] 
                new_tup.append(idx)
                new_tup.sort()
                
                H = dit.shannon.entropy(d.marginal(new_tup))
                if H <= 1: 
                    tuple_ent.append(tuple((tuple(new_tup), H)))
                    
        if len(tuple_ent)== 0: 
            stop = True
            return entropies
        
        else: 
            length_counter += 1
            unique_tups = list(set([tuple(x[0]) for x in tuple_ent]))
            tuple_ent = [tuple((ut, dit.shannon.entropy(d.marginal(ut)))) for ut in unique_tups] 
            entropies = tuple_ent



#=============================================================================
# DIRECTED INFORMATION ALL PAIRS 
# this is equivalent to mutual information
#=============================================================================
def DI_matrix(net,  FixedNodes={}, RemoveNodes=[]): 
    ''' This function clauclated the directed information flow between all pairs of nodes unding 
    the expression defined in Mathai et. al., 2007 As discussed in the repost, 
    in Boolean networks of the sort generated with the BooleaNetwork class, 
    this is equivalent to calculating the mutual infromation
    '''
    
    NodeIDs = [str(s) for s in net.nodeDict.keys()]
    N = len(NodeIDs)

    start, stop = net.calc_all_updates(rounds=1, FixedNodes=FixedNodes)
    YY={}
    X={}
    XYY={}


    NodeIDs_working=NodeIDs.copy()
    for x in RemoveNodes: 
        NodeIDs_working.remove(x)
        
    normalizer = float(start.shape[0])

    for x in NodeIDs_working: 
        i = NodeIDs.index(x)
        c = Counter(start[:,i])
        X[NodeIDs[i]]=[c[k]/normalizer for k in range(2)]

        Y1Y2 = start[:, i] + 2*stop[:, i]
        c = Counter(Y1Y2)
        YY[NodeIDs[i]]=[c[k]/normalizer for k in range(4)]

        for y in NodeIDs_working: 
            j = NodeIDs.index(y)
            c = Counter(4*start[:, j]+Y1Y2)
            XYY[NodeIDs[i]+'_'+NodeIDs[j]] = [c[k]/normalizer for k in range(8)]

    DI_Matrix = np.zeros((len(NodeIDs_working),len(NodeIDs_working)))
    MI_Matrix = DI_Matrix

    for i, node_i in enumerate(NodeIDs_working): 
        for j, node_j in enumerate(NodeIDs_working): 
                #MI_Matrix[i, j] = em.MutualInformation(pairs[node_i+'_'+node_j], singlesX[node_i], singlesY[node_j])
            DI_Matrix[i, j] = em.Entropy(YY[node_j])+em.Entropy(X[node_i])-em.Entropy(XYY[NodeIDs_working[j]+'_'+NodeIDs_working[i]])

    for i in range(len(NodeIDs_working)):
        DI_Matrix[i, i] = -100
        
    return DI_Matrix


def calc_mutual_information(self, X, Y, shift, verbose=False):
    '''
    Calculate Mutual Information between two nodes. 

    X,Y....... Ids of the nodes between which MI should be calculated
    shift..... temporal distance between X and Y
    '''
    x_idx = list(Net.nodeDict.keys()).index(X)
    y_idx = list(Net.nodeDict.keys()).index(Y)

    start, stop = Net.calc_all_updates(rounds=max(1, shift))

    if shift==0: 
        XX = stop[:, x_idx]
        YY = stop[:, y_idx]
    else: 
        XX = start[:, x_idx]
        YY = stop[:, y_idx]

    px = [1- np.mean(XX), np.mean(XX)]
    py = [1- np.mean(YY), np.mean(YY)]

    c = Counter(XX+2*YY)
    p_joint = np.array([c[k] for k in range(4)])
    p_joint = p_joint/sum(p_joint)
    mi = em.MutualInformation(p_joint, px, py)
    #p_joint = [len(np.where(XX+YY==2)[0]), len(np.where(XX-YY==1)[0]),  len(np.where(YY-XX==1)[0]), len(np.where(XX+YY==0)[0])]
    #p_joint = np.array(p_joint)/float(len(XX))

    if verbose: 
        print("px = \t{}".format(px))
        print("py = \t{}".format(py))
        print("pxy = \t{}".format(p_joint))

        print("MI = {}".format(mi))

    return mi

def calc_mutual_information_all_pairs(Net, rounds=1, plot=True, save_name='',RemoveNodes=[]): 
    '''
    Calculates all updates for as many steps as specified by the rounds parameter.  The claucluates mutual information
    at subsequent timesteps between all pairs of nodes independent of wheteher or not they are directly connected. 
    If the plot parameter is set to true, an imshow plot will be produced from the the result matrix.
    '''

    NodeIDs = list(Net.nodeDict.keys())
    NodeIDs_working = NodeIDs
    for x in RemoveNodes: 
        NodeIDs_working.remove(x)

    N = len(NodeIDs_working)
    MIs = np.zeros((N,N))
    start, stop = Net.calc_all_updates(rounds=max(1, rounds))
    if rounds==0: 
        start=stop

    for idx_i, node_i in enumerate(NodeIDs_working):
        for idx_j, node_j in enumerate(NodeIDs_working):

            XX = start[:, idx_i]
            YY = stop[:, idx_j]
            px = [1- np.mean(XX), np.mean(XX)]
            py = [1- np.mean(YY), np.mean(YY)]              
            p_joint = [len(np.where(XX+YY==2)[0]), len(np.where(XX-YY==1)[0]),  len(np.where(YY-XX==1)[0]), len(np.where(XX+YY==0)[0])]

            p_joint = np.array(p_joint)/float(len(XX))
            MIs[idx_i, idx_j] =  em.MutualInformation(p_joint, px, py)

    if plot: 
        #plt.figure(figsize=(5,5))        
        plt.imshow(MIs, aspect='auto')
        plt.xticks(np.arange(0, N,1), list(NodeIDs_working), rotation=90)  
        plt.yticks(np.arange(0, N,1), list(NodeIDs_working))  
        plt.ylabel("t", rotation=0, size=20)
        plt.xlabel("t+{}".format(rounds), size=20)
        plt.colorbar()

        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        plt.gca().yaxis.set_minor_locator(minor_locator)
        plt.grid(which='minor')

        if len(save_name)>0: 
            plt.savefig(save_name)
    return MIs



#=============================================================================
# INFORMATION FLOW FUNCTIONS
#=============================================================================

def get_parent_child_tuples(Net, Ninps):
    '''
    Finds all subsets of nodes where Ninps inputs feed into one output. 
    In particular, when a node has more than Ninps inputs, all relevant subgroups of inputs will be returned.
    '''
    tuples = []
    for node in Net.nodeDict:
        parents = Net.nodeDict[node].InputIDs
        if len(parents) < Ninps: 
            parent_combinations=[]
        else:
            parent_combinations = list(combinations(parents, r=Ninps))

        for pc in parent_combinations: 
            tuples.append((list(pc), [node]))
    return tuples


def info_flow_in_tuples_and_triplets(Net, Ninps):
    ''' Calculates the information flow within pairs and triplets of nodes. 
    If Ninps = 1 the function will find all nodes with at least one input and calculate 
    the mutual information between input and output.
    If Ninps = 2 the fnction will find all nodes with at least 2 inputs and calculates mutual information and 
    information decompostion between parents and child. 
    '''
    
    if Ninps not in [1,2]:
        print('number of inputs can only be 1 or 2')
        return 

    if Ninps == 1:
        tuples = get_parent_child_tuples(Net, 1)
 
        if not type(tuples[0][0][0]).__name__ == 'int': 
            tuples_new = []
            IDs = list(Net.nodeDict.keys())
            for tup in tuples: 
                tuples_new.append(tuple([[IDs.index(tup[0][0])],[IDs.index(tup[1][0])]]))
            tuples = tuples_new
            
        print('Calculating info flow for {} pairs....'.format(len(tuples)))
        start, stop = Net.calc_all_updates()
        mi = []
        
        for tup in tuples: 
            states = [(start[i,tup[0][0]], stop[i,tup[1][0]]) for i in range(start.shape[0])]
            c = Counter(states)
            unique_states = list(c.keys())
            state_probs = np.array(list(c.values()))
            dist = dit.Distribution(unique_states, state_probs/sum(state_probs))
            
            mi.append(dit.shannon.mutual_information(dist, [0],[1]))
        return mi

        
    if Ninps == 2:
        tuples = get_parent_child_tuples(Net, 2)
        print('Calculating info flow for {} triplets....'.format(len(tuples)))

        if not type(tuples[0][0][0]).__name__ == 'int': 
            tuples_new = []
            IDs = list(Net.nodeDict.keys())
            for tup in tuples: 
                tuples_new.append(tuple([[IDs.index(tup[0][0]), IDs.index(tup[0][1])],[IDs.index(tup[1][0])]]))
            tuples = tuples_new
        
        
        start, stop = Net.calc_all_updates()
        syn = []
        red = []
        unique = []
        mi = []
        mi_pair = []

        for tup in tuples: 
            states = [(start[i,tup[0][0]], start[i,tup[0][1]], stop[i,tup[1][0]]) for i in range(start.shape[0])]
            c = Counter(states)
            unique_states = list(c.keys())
            state_probs = np.array(list(c.values()))
            dist = dit.Distribution(unique_states, state_probs/sum(state_probs))
            a = dit.pid.PID_BROJA(dist)

            syn.append(a.get_partial(((0, 1),)))
            red.append(a.get_partial(((0,), (1,))))
            unique.append(a.get_partial(((0,),)))
            unique.append(a.get_partial(((1,),)))
            mi.append(dit.shannon.mutual_information(dist, [0],[2]))
            mi.append(dit.shannon.mutual_information(dist, [1],[2]))
            mi_pair.append(dit.shannon.mutual_information(dist, [0,1],[2]))
            
        return syn, red, unique, mi, mi_pair
    return syn, red, unique, mi, mi_pair


def info_decomposition_clusters(Net, InputsX, InputsY, Outputs, set_names=False):
    ''' 
    Calculates information decomposition and mutual information between subgroups of nodes. 
    For this the subgroups are first reduced to triplets by calculating the joint dirstribution of all
    nodes in a subgroup
    '''
    res, dist = prepare_cluster_distribution(Net, InputsX, InputsY, Outputs, set_names=False)

    pid = dit.pid.PID_BROJA(dist)

    uni0 = pid.get_partial(((0,),))
    uni1 = pid.get_partial(((1,),))
    syn = pid.get_partial((((0, 1),)))
    red = pid.get_partial((((0,), (1,))))

    mi = dit.shannon.mutual_information(dist, [0,1],[2])
    mi0 = dit.shannon.mutual_information(dist, [0],[2])
    mi1 = dit.shannon.mutual_information(dist, [1],[2])

    return {'UNI0':uni0, 'UNI1':uni1, 'SYN':syn, 'RED':red, 'MI0':mi0, 'MI1':mi1, 'MI':mi}


def prepare_cluster_distribution(Net, InputsX, InputsY, Outputs, set_names=False): 
    '''Generates a dit Distribution of three random varaibles where each varaible represents a group of N>=1 original varaibles.'''
    start, stop = Net.calc_all_updates()
    Nodes = list(Net.nodeDict.keys())
    M = start.shape[0]
    
    inpX = np.zeros(M)
    for node_idx, node in enumerate(InputsX): 
        inpX+= (node_idx+1)*start[:,Nodes.index(node)]
        
    inpY = np.zeros(M)
    for node_idx, node in enumerate(InputsY): 
        inpY+= (node_idx+1)*start[:,Nodes.index(node)]   
        
    outp = np.zeros(M)
    for node_idx, node in enumerate(Outputs): 
        outp+= (node_idx+1)*stop[:,Nodes.index(node)]   
        
    res = np.vstack((np.vstack((inpX, inpY)), outp))
    res = [tuple(res[:, i].astype(int)) for i in range(M)]
    c = Counter(res)
    states = list(c.keys())
    probs = list(c.values())
    d = dit.Distribution(states, [p/sum(probs) for p in probs])
    
    if set_names:
        d.set_rv_names(["Inp1","Inp2","Outp" ])
        
    return res, d



#=============================================================================
# PLOT FUNCTIONS
#=============================================================================
def make_state_diagram(BooleanNetwork, FixedNodes={}, savename='', return_graph=False):
    '''
    Calculates all macro states of a Boolean network and their relationship. The resulting set of states 
    is represented as a graph where node A is a parent of node B when one updating 
    step leads from macro-state A to macro-state B. 
    '''
    N = len(BooleanNetwork.nodeDict)-len(FixedNodes) # number of varying nodes
    names = [node_name for node_name in BooleanNetwork.nodeDict.keys() if not node_name in FixedNodes.keys()]
    start_values = list(product(range(2), repeat=N))
     
    G=nx.DiGraph()
    G.add_nodes_from(start_values)
    value_dict = FixedNodes # empty is nothing is fixed, containing fixed nodes otherwise
     
    for sv_idx, sv in enumerate(start_values): 
        print(sv_idx/len(start_values), end='\r')
        value_dict = {x:sv[i] for i, x in enumerate(names)}
        BooleanNetwork.assign_values_to_nodes(value_dict)
     
        start = tuple(BooleanNetwork.CurrentValues.values())
        BooleanNetwork.update_all()
        stop = tuple(BooleanNetwork.CurrentValues.values())
        G.add_edge(start, stop)
    #return G 
     
    print('\nwaiting for plot')
    pos = nx.planar_layout(G, scale=2)
    #pos = nx.kamada_kawai_layout(G, scale=100)
    #pos = nx.spiral_layout(G)
    labels = {a:''.join(tuple(map(str , a ))) for a in start_values}
    #nx.draw_networkx(G, pos=pos, node_color='thistle',node_shape='s',node_size=800,labels=labels,
    #edge_color='black',  font_weight='bold', arrowprops=dict(arrowstyle="->",  max_arrow_width=0.3))
    nx.draw_networkx(G, pos=pos, node_color='thistle', node_size=300,labels=labels,
    edge_color='black' , arrowprops=dict(arrowstyle="->",  max_arrow_width=0.3))
    plt.axis("off")

    if len(savename)>0: 
        plt.savefig(savename)    
    if return_graph: 
        return G
    
    
    
def make_attractor_graph(Net, draw =True): 
    '''Identifies the attractors of a Boolean network and generates  a graph representinng theri relations'''
    a_start, a_stop = identify_attactors(Net)
    G = nx.DiGraph()
    G.add_nodes_from(a_start)
    G.add_edges_from([a_start[i],a_stop[i]] for i in range(len(a_start)))
    pos = generate_positions(G)

    if draw: 
        nx.draw_networkx_nodes(G, pos=pos, node_size=300, node_color='lightgrey')
        nx.draw_networkx_edges(G, pos=pos )
        nx.draw_networkx_labels(G, pos=pos, labels={a:tup2str(a) for a in a_stop} )
        plt.axis('off')
    
    return G
# =============================================================================
