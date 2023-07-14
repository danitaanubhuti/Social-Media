"""
Social Media Project

Name: Danita Anubhuti Prakash
student ID: s4745511

Tasks:
    1. Calculate the Betweenness Centrality for nodes in the Facebook dataset.
    Overview: write code to load the Facebook social network data and construct an undirected 
    and unweighted graph. Based on the constructed graph, you are required to write a program 
    to calculate the betweenness centralities for the graph vertices.
    Input: The provided Facebook social network data.
    Output: The top-10 nodes with the highest betweenness centralities.
    
    2. Calculate PageRank Centrality for nodes in the Facebook dataset. 
    Overview: write code to load the Facebook social network data and construct an undirected 
    and unweighted graph. Based on the constructed graph, you are required to write a program 
    to calculate the PageRank (with alpha=0.85,beta=0.15) centralities for the graph vertices.
    Input: The provided Facebook social network data.
    Output: The top-10 nodes with the highest PageRank centralities.
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import networkx as nx


def construct_graph(dataframe):
    """
     Converts the DataFrame to a Graph dictionary and 
     then converts it to undirected and unweighted graph
     Input Parameter: dataframe
     Return : Returns the undirected and unweighted graph
    """
    graph = {}  
    
    for index, row in df.iterrows(): #Coverting the dataframe by iterating over the rows
        source_node, target_node = row['Source'], row['Target'] #Soring the coloumn values in variables
        if source_node not in graph: # creating key vales in the dictionary
            graph[source_node] = [target_node]
        else:#appending target nodes as values to the keys
            graph[source_node].append(target_node)
            
    g = nx.Graph()#creates an undirected graph using networkx matrix
    for key, value in graph.items():
        #for each key value pair the loop adds an edge to the graph.
        #this results in creating an undirected and unnweighted graph.
        for neighbor in value:
            g.add_edge(key,neighbor)
            
    return g

def top_n(dic_centrality, n=10):
    """
     Finds the top n nodes having highest values in a dictionary
     Input Parameter: dic_centrality: Input dictionary used for sorting
                n: the total elements needed to be retirved from the dictionary
     Return : returns top n nodes having hihest values in a dictionary
    """
    sorted_dictionary = sorted(dic_centrality.items(), key=lambda x: x[1], reverse=True) #sorts the input dictionary
    top_n_elements = dict(sorted_dictionary[:n])#stores the top n nodes in the dictionary
    
    top_n_list = list(top_n_elements.keys())#retieves the node values
    top_n_nodes = [int(x) for x in top_n_list]
    
    return top_n_nodes


def shortestpath_bfs(graph, start_node):
    """
    Performs a BFS traversal of the graph starting from node s.
    Input Parameters: graph: Graph Dictionary
                      start_node: the node to start the bfs traversal from
    Return: A set of data structures as visited_nodes, parent_nodes, sigma, D
    """
    
    #Initilization og different data structures
    visited_nodes = []#keeps track of the visited nodes
    parent_nodes = {current_vertex: [] for current_vertex in graph} #tracks the parent nodes
    sigma = {current_vertex: 0 for current_vertex in graph} #tracks number of shortest paths
    sigma[start_node] = 1
    D = {current_vertex: -1 for current_vertex in graph}# keeps track of the shortest path os each node
    D[start_node] = 0 
    Q = [start_node]#queue to keep track of the traversal

    #Finding the shortest path from bfs traversal
    while Q:
        
        current_vertex = Q.pop(0) # v represents the current vertex used for the traversal
        visited_nodes.append(current_vertex)
        
        for neigh_vertex in graph[current_vertex]:# w represents the neighbouring vertex that are being procesed
            if neigh_vertex not in graph:#if neighbouring vetex not there continue
                continue
            if D[neigh_vertex] == -1: #neighbour is not visited then add to the queue
                Q.append(neigh_vertex)
                D[neigh_vertex] = D[current_vertex] + 1 #distance from current node is noted
            if D[neigh_vertex] == D[current_vertex] + 1:
                sigma[neigh_vertex] += sigma[current_vertex]
                parent_nodes[neigh_vertex].append(current_vertex)
                
    return visited_nodes, parent_nodes, sigma, D

def accumulate_dependencies(graph, nodes_visited, parent_nodes, sigma):
    """
    Accumulates the dependencies for each node based on the BFS tree.
    Input Paramteres: graph: Graph Dictionary
                      nodes_visited: output from bfs method
                      parent_nodes: output from bfs method
                      sigma: number of shortest paths
    Return: The dependency value stored in Delta dictionary
    """
    
    delta = {parent_vertex: 0 for parent_vertex in graph} # dictionary containing key as vetex and inital value as 0

    #Calculating dependency for each node by looping
    while nodes_visited:
        visited_node = nodes_visited.pop()
        for parent_vertex in parent_nodes[visited_node]:
            if sigma[visited_node] == 0:#no shortest paths to the visited node
                continue
            #otherwise calculating the dependancy
            delta[parent_vertex] += sigma[parent_vertex]/sigma[visited_node]*(1+delta[visited_node])
            
    return delta

def create_matrices(graph, eps, length_graph):
    """
    Creates the adjacency matrix and inverse degree matrix from the input graph.
    Input Parameters: graph: Adictionary containing nodes as scoure as key and value as target node lists
    Return: Adjacency Matrix and Inverse Degree Matrix
    """
    #sorting nodes and creating an matrix using numpy
    nodes = sorted(list(graph.nodes()))
    adj_matrix = np.zeros((length_graph, length_graph))# Adjacency  matrix- initialized with zeros 

    # Populate the adjacency matrix
    for row in range(length_graph):
        for column in range(length_graph):
            if nodes[column] in graph[nodes[row]]:
                adj_matrix[row][column] = 1

    # Construct inverse degree matrix
    #to avoid division by zero, values with 0 are changed to eps
    D_inv = np.diag(1 / np.maximum(adj_matrix.sum(axis=1), eps))
    
    return adj_matrix, D_inv

def centrality_networkx(graph):
    """
    To understand the centralities obatined from networkx function
    Input Paramter: Graph: Undirected and Unweighted graph
    Return : The measures obtained from networkx pakage
    """
    betweeness =  top_n(nx.betweenness_centrality(graph, normalized= False))
    pagerank = top_n(nx.pagerank(graph,alpha=0.85))
    return betweeness, pagerank
    
def output_textfile(betweeness, pagerank):
    """
    Saving the outputs to a file.
    Input Parameter: betweeness:The betweeness centrality measure
                     pagerank:The pagerank centrality measure
    Return: None
    """
    with open('47455119.txt.txt', 'w') as file:#saving output file- 47455119.txt
        #converting the output to a string
        for node in betweeness:
            file.write(str(node) + " ")
        file.write("\n")
        for node in pagerank:
            file.write(str(node) + " ")

def betweenness_centrality(graph):
    """
    #CODE FOR TASK ONE 
    Calculates the betweenness centrality for each node in the graph.
    Input Parameter: graph: Graph Dictionary
    Return: Centrality dictionary containg key as node and item as centrality value
    """
    # making a dictionary and initializing it with 0.0 for each node
    centrality = {v: 0.0 for v in graph}
    
    for node in graph:
        visited_node, parent_node, sigma, D = shortestpath_bfs(graph, node)# finding shortest path with help of bfs
        #finding dependency of the nodes stores in delta
        delta = accumulate_dependencies(graph, visited_node, parent_node, sigma)
        for vertex in delta:
            if vertex != node:
                # formula to find centrality
                centrality[vertex] += delta[vertex]/2 
    
    return centrality

def pagerank_centrality(graph, alpha=0.85, beta=0.15):
    """
    #CODE FOR TASK TWO 
    Computes the PageRank centrality of each node in a graph.
    Input Parameter: Graph: A dictionary graph
                     Alpha: Specified in Task2 as 0.85
                     Beta: Specified in Task2 as 0.15
    Return :Centrality Values of each node in the form of a dictionary
    """
    eps = 1e-6
    len_graph = len(graph)
    # Create adjacency and degree matrices
    adj_matrix, D_inv = create_matrices(graph, eps, len_graph)

    # Creating two vectors for PageRank
    prev_PR = np.zeros(len_graph)
    page_rank = np.ones(len_graph) / len_graph
    
    #Calculating the Pagerank for a given pagr reank till it's different with previous pagerank is very small
    while np.sum(np.abs(prev_PR-page_rank)) > eps:
        prev_PR = page_rank #previous page rank
        #Current Page Rank
        page_rank = alpha * adj_matrix.T @ D_inv @ page_rank + (1 - alpha) * beta * np.ones(len_graph) + (1 - alpha) * (1 - beta) * np.sum(page_rank) / len_graph
        page_rank = page_rank / np.sum(page_rank)
        
    page_rank = dict(zip([index for index in range(len(page_rank))], page_rank))
    return page_rank

def main():
    """
    The main function. Execution starts here.
    Input Parameter: None
    Return : None
    """
    #Construction a graph from the data frame
    graph = construct_graph(df)
        
    #Finding the node betweeness centrality for the graph
    node_betweeness = betweenness_centrality(graph)
    top_10_betweeness = top_n(node_betweeness, n=10)
    print("The top 10 nodes having highest betweeness centrality are",top_10_betweeness)
    
    #Finding the pagerank centrality for the graph
    pagerank =  pagerank_centrality(graph)
    top_10_pagerank = top_n(pagerank, n=10)
    print("The Top 10 nodes having pagerank centrality are",top_10_pagerank) 
    
    #Saving the output in a textfile
    output_textfile(top_10_betweeness, top_10_pagerank)
    
    #Finding the Centralities from networkx pakage on the graph
    networkx_betweeness, networkx_pagerank = centrality_networkx(graph)
    print("The Top 10 nodes having highest betweeness from networkx pakage",networkx_betweeness) 
    print("The Top 10 nodes having highest pagerank from networkx pakage",top_10_pagerank)     
    
if __name__ == '__main__':
    """
    Call back the code to load the data and generate the results.
    """
    #Importing the data from text file to dataframe
    #Here the data is stored in a text file called 'Facebook_data.txt'
    df = pd.read_csv('data.txt', delimiter=' ', header=None, names=['Source', 'Target'])
    main()# Calling main to execute the function
