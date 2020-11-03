import sys
from pyspark import SparkContext
from collections import deque
import copy
import time

#FUNCTIONS
def make_pairs(item, u_list):
  pairs = []
  for u in u_list:
    if u[0] > item[0]:
      pair = [item[1], u[1]]
      pair.sort()
      pair_t = tuple(pair)
      pairs.append(pair)
  return pairs

def bfs(graph, root_node):
  queue = deque([root_node])
  level = {root_node: 0}
  parent = {root_node: []}
  while queue:
    v = queue.popleft()
    for n in graph[v]:
      if n not in level:            
        queue.append(n)
        level[n] = level[v] + 1
      #Add to parent
      if n not in parent:
        parent[n] = [v]
      #Accounts for multiple parents
      elif n in parent:
        n_level = level.get(n)
        v_level = level.get(v)
        if v_level == n_level - 1:
          parent[n].append(v)
  dag = {}
  for node in level.keys():
    dag[node] = (level.get(node), parent.get(node))
  return dag

def gm_compute_edge_scores(root_dag_dict, root):
  '''For a given node's DAG, output the DAG's edges and GM score.
  input: dag - a given node's direct acyclic graph (DAG)
  output: dag_edges_score_l - a list of the dag's edges and their GM score'''
  levels = list(root_dag_dict.keys())
  levels.sort(key = lambda x: x[1], reverse=True)
  dag_edge_scores = {}

  for level in levels:
    neighbors = root_dag_dict.get(level)
    for idx, neighbor in enumerate(neighbors):
      #NODE SCORE
      #Check if there are child nodes
      child_level = (level[0], level[1] + 1)
      child_nodes = root_dag_dict.get(child_level)

      #if there are no child nodes for current level, add 1 to dag node score
      if child_nodes == None:
        root_dag_dict[level][idx][1].append(1)
      
      #if there are child nodes for current level
      elif child_nodes != None:
        node_base_score = 1
        child_edges_base_score = 0
        #check each child node to see if neighbor is in the child's parents
        for child in child_nodes:
          #if the neighbor is not the child's parent
          if neighbor[0] not in child[1][0]:
            pass
          #if the neighbor is the child's parent 
          elif neighbor[0] in child[1][0]:
            child_edge = [neighbor[0], child[0]]
            child_edge.sort()
            child_edge_t = tuple(child_edge)
            child_edge_score = dag_edge_scores.get(child_edge_t)
            child_edges_base_score += child_edge_score
        node_final_score = node_base_score + child_edges_base_score
        root_dag_dict[level][idx][1].append(node_final_score)

      #EDGE SCORE
      #Assign points to upper edge
      parent_nodes = neighbor[1][0]
      upper_level = (level[0], level[1] - 1)
      upper_neighbors = root_dag_dict.get(upper_level)

      #For all upper_neighbors, find parents and count how many parent paths exist
      if upper_neighbors != None:
        tot_parent_paths = 0
        parent_paths = {}
        for parent in parent_nodes:
          for upper_neighbor in upper_neighbors:
            if parent == upper_neighbor[0]:
              tot_parent_paths += len(upper_neighbor[1][0])
              parent_paths[parent] = len(upper_neighbor[1][0])

        #For each parent node, assign points to upper edge proportional to the # parent_paths of the parent node
        for parent in parent_nodes:
          d_edge = [neighbor[0], parent]
          d_edge.sort()
          d_edge_t = tuple(d_edge)
          node_score = neighbor[1][1]
          # print('node_score: ', node_score)

          #In this scenario, your parent is the root so give all node point to the upper edge
          if tot_parent_paths == 0:
            dag_edge_scores[d_edge_t] = node_score
          
          else:
            upper_edge_score = node_score * (parent_paths.get(parent) / tot_parent_paths)
            dag_edge_scores[d_edge_t] = upper_edge_score

  dag_edge_scores_l = list(dag_edge_scores.items())
  return dag_edge_scores_l

def modularity(communities_l, m, graph_reduced, graph_initial):
  '''Calculate modularity score for all communities.
  input: communities_l - a list of communities
  input: m - number of edges of initial graph
  input: graph_reduced - graph with removed edges currently evaluating. Used to get k_i, k_j.
  inptu: graph_initial - used to get a_ij'''
  Q = 0
  for comm in communities_l:
    for i in comm:
      k_i = len(graph_reduced.get(i))
      for j in comm:
        k_j = len(graph_reduced.get(j))
        if j in graph_initial[i]: a_ij = 1
        else: a_ij = 0
        q = a_ij - ((k_i * k_j)/ (2*m) )
        Q += q
  Q = (1/(2*m)) * Q
  return Q

#PARAMETERS
filter_thresh = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]

sc = SparkContext(appName="inf553")
sc.setLogLevel("ERROR")

#DRIVER

#Load the u-b data and create reference objects ub_set_d and u_list
ub_set = sc.textFile(input_file_path)\
.map(lambda x: tuple(x.split(',')))\
.filter(lambda x: x[0] != 'user_id')\
.map(lambda x: (x[0], [x[1]]) )\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], set(x[1])))\
.filter(lambda x: len(x[1]) >= filter_thresh)\
.persist()

ub_set_d = dict(ub_set.collect())
u_list = list(enumerate(ub_set_d.keys()))

#Use flatmap to get all user pairs (~ 5.6M)
#For each user pair, check if business intersection >= thresh
# For valid user pairs, output key = u1 and value = u2
u_edges_base = sc.parallelize(u_list)\
.flatMap(lambda x: [pair for pair in make_pairs(x, u_list)])\
.map(lambda x: (x, ub_set_d.get(x[0]) & ub_set_d.get(x[1])))\
.filter(lambda x: len(x[1]) >= filter_thresh)\
.map(lambda x: (x[0][0], [x[0][1]]))\
.persist()

u_edges_a = u_edges_base.reduceByKey(lambda x, y: x + y).persist()
u_edges_b = u_edges_base.map(lambda x: (x[1][0], [x[0]]))\
.reduceByKey(lambda x, y: x + y).persist()

#Save the graph to a dict, with key = user, value = {users connected to key user}
u_edges_full = sc.union([u_edges_a, u_edges_b])\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], set(x[1])))
graph_initial = dict(u_edges_full.collect())
m = len(u_edges_base.collect())

#NODES 
u_nodes_a = u_edges_a.reduceByKey(lambda x, y: 1)\
.map(lambda x: (x[0], 1))\
.persist()

u_nodes_b = u_edges_b.reduceByKey(lambda x, y: 1)\
.map(lambda x: (x[0], 1))\
.persist()

u_nodes_full = sc.union([u_nodes_a, u_nodes_b])\
.reduceByKey(lambda x, y: 1)\
.map(lambda x: str(x[0]))\
.persist()
u_nodes_full_l = u_nodes_full.collect()

#FIRST PASS
#For each node, contruct the node's direct acyclic graph (dag)
#output will have key = root_node and value = {(root_node, level0): [(neighbor_node, [[parent1, parent2, ...]])],
                                             # (root_node, level1): [('5', [['4']]), ('3', [['4']])], ... }
dag_base = sc.parallelize(u_nodes_full_l)\
.map(lambda x: (x, bfs(graph_initial, x))).persist()

dag = dag_base.flatMap(lambda x: [((x[0], neighbor[1][0]), [(neighbor[0], [neighbor[1][1]])]) for neighbor in list(x[1].items())])\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0][0], [(x[0], x[1])]))\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], dict(x[1]))).persist()

#For each node's DAG get the edges and betweenness score
dag_edges_scores = dag.flatMap(lambda x: [edge_score for edge_score in gm_compute_edge_scores(x[1], x[0])])\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], x[1]/2))

#Select the edge with the maximum GM score
dag_edges_scores_l = dag_edges_scores.collect()
dag_edges_scores_l_s = sorted(dag_edges_scores_l, key = lambda x: (-x[1], x[0][0]))
max_score = dag_edges_scores_l_s[0][1]
edges_to_remove = list(filter(lambda x: x[1] == max_score, dag_edges_scores_l_s))

#Write the betweenneess result to the outfile
with open(betweenness_output_file_path, 'w') as outfile:
  for line in dag_edges_scores_l_s:
    line_str = str(line)[1:-1]
    outfile.write(line_str)
    outfile.write('\n')

#ALL SUBSEQUENT PASSES
#Remove the most between edge(s) from the graph 
start = time.time()
graph_reduced = copy.deepcopy(graph_initial)
Q_tracker = {} #Will look like {idx: {'Q': 0.23, 'communities': [(u0, u1, u2), (u3, u4, u5, u6)]}}

max_Q = 0
current_Q = 0
iter = 1
while current_Q >= max_Q:
# for iter in range(1, 8):
  max_Q = current_Q
  print('iter: ', iter)

  #Calculate the betweenness scores
  dag = dag_base.flatMap(lambda x: [((x[0], neighbor[1][0]), [(neighbor[0], [neighbor[1][1]])]) for neighbor in list(x[1].items())])\
  .reduceByKey(lambda x, y: x + y)\
  .map(lambda x: (x[0][0], [(x[0], x[1])]))\
  .reduceByKey(lambda x, y: x + y)\
  .map(lambda x: (x[0], dict(x[1]))).persist()

  #For each node's DAG get the edges and betweenness score
  dag_edges_scores = dag.flatMap(lambda x: [edge_score for edge_score in gm_compute_edge_scores(x[1], x[0])])\
  .reduceByKey(lambda x, y: x + y)\
  .map(lambda x: (x[0], x[1]/2))

  #Select the edge(s) with the maximum betweenness score
  dag_edges_scores_l = dag_edges_scores.collect()
  dag_edges_scores_l_s = sorted(dag_edges_scores_l, key = lambda x: -x[1])
  max_score = dag_edges_scores_l_s[0][1]
  edges_to_remove = list(filter(lambda x: x[1] == max_score, dag_edges_scores_l_s))

  #Remove the edges from the graph
  for edge in edges_to_remove:
    graph_reduced[edge[0][0]].remove(edge[0][1])
    graph_reduced[edge[0][1]].remove(edge[0][0])

  #Recalculate dags for all ndoes with the updated graph
  dag_base = sc.parallelize(u_nodes_full_l)\
  .map(lambda x: (x, bfs(graph_reduced, x))).persist()

  #Extract communities from each dag
  communities = dag_base.map(lambda x: (list(x[1].keys()), 1))\
  .map(lambda x: (tuple(sorted(x[0])), x[1]))\
  .reduceByKey(lambda x, y: 1)\
  .map(lambda x: x[0])
  communities_l = communities.collect()

  #Calculate modularity from the communities
  current_Q = modularity(communities_l, m, graph_reduced, graph_initial)
  print('Q: ', current_Q)

  #Save the iter, Q, and communities
  Q_tracker[iter] = {'Q': current_Q, 'communities': communities_l}

  iter += 1

end = time.time()
t_time = end-start
print('find max modularity time: ', t_time)

#Select the community pairing with the maximum modularity
communities_final = Q_tracker.get(iter-2).get('communities')
communities_final_s = sorted(communities_final, key = lambda x: (len(x), x[0]))

#Write the community result to the outfile
with open(community_output_file_path, 'w') as outfile:
  for line in communities_final_s:
    line_str = str(line)[1:-1]
    outfile.write(line_str)
    outfile.write('\n')

