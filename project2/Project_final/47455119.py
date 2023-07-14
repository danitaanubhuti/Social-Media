"""
Social Media Project 2
Name- Danita Anubhuti Praakash
StudentID - s4745511

Overview: The provided co-author network has 5,240 nodes, 11,696 edges. The edges
of the whole co-author network are then split into three parts, which are E_train, E_validation, 
and E_test. The missing 100 positive edges are formed among the core node. 
Based on the given training and validation sets of the co-author network, 
you are required to write a program to rank the unlabeled edges in the 
test set. For each pair of nodes in the test set, your program should compute a proximity 
score. Rank the 10,100 pairs of nodes according to your computed proximity score in 
descending order and the Top-100 pairs of nodes will be compared with the 
ground truth to compute accuracy. 

Input: The provided network datasets.
Output: The predicted Top-100 edges
"""

#importing libraries
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score
import itertools
import math
import itertools
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def construct_graph(dataframe):
    """
     Converts the DataFrame to a Graph dictionary and 
     then converts it to undirected and unweighted graph
     Input Parameter: dataframe
     Return : Returns the undirected and unweighted graph
    """
    graph = {}  
    
    for index, row in dataframe.iterrows(): #Coverting the dataframe by iterating over the rows
        source_node, target_node = row['Node1'], row['Node2'] #Soring the coloumn values in variables
        if source_node not in graph: # creating key vales in the dictionary
            graph[source_node] = [target_node]
        else:#appending target nodes as values to the keys
            graph[source_node].append(target_node)
            
    g = nx.Graph()
    for key, value in graph.items():
        #for each key value pair the loop adds an edge to the graph.
        #this results in creating an undirected and unnweighted graph.
        for neighbor in value:
            g.add_edge(key,neighbor)
            
    return g

def adamic_adar_index(G, node1, node2):
    """
    Find the adamic adar score for the given nodes in the graph.
    Input: Graph and the nodes in graph to find score.
    Return: The score between the two nodes in a graph
    """
    #finding the common neighbours between the two nodes
    common_neighbors = set(G.neighbors(node1)) & set(G.neighbors(node2))
    adamic_score = 0

    for neighbor in common_neighbors:#iterating over the common neighbours
        degree = G.degree(neighbor)
        adamic_score += 1 / math.log(degree) if degree > 1 else 0

    return adamic_score

def calculate_total_neighbors_in_common(G, node1, node2):
    """
    Find the total number of common neighbours for the nodes in the given graph.
    Input: Graph and the nodes in graph to find score.
    Return: The total number of common neighbours in pair of nodes.
    """
    #if the neighbours are not in graph
    if node1 not in G.nodes() or node2 not in G.nodes():
        return 0
    #finding the common neighbours
    neighbors_node1 = set(G.neighbors(node1))
    neighbors_node2 = set(G.neighbors(node2))
    total_neighbors_in_common = len(neighbors_node1.intersection(neighbors_node2))
    return total_neighbors_in_common

def calculate_jaccard_similarity(G, node1, node2):
    """
    Find the jaccard similarity for the nodes in the given graph.
    Input: Graph and the nodes in graph to find score.
    Return: The jaccard similarity in pair of nodes.
    """
    #if the neighbours are not in graph
    if node1 not in G.nodes() or node2 not in G.nodes():
        return 0
    #finding the neighbours for each pair.
    neighbors_node1 = set(G.neighbors(node1))
    neighbors_node2 = set(G.neighbors(node2))
    
    intersection = neighbors_node1.intersection(neighbors_node2)
    union = neighbors_node1.union(neighbors_node2)
    
    if len(union) == 0:
        return 0
    #finding the jaccard similarity
    jaccard_similarity = len(intersection) / len(union)
    return jaccard_similarity

def calculate_preferential_attachment(G, node1, node2):
    """
    Find the preferential attachment for the nodes in the given graph.
    Input: Graph and the nodes in graph to find score.
    Return: The preferential attachment in pair of nodes.
    """
    if node1 not in G.nodes() or node2 not in G.nodes():
        return 0
    #finding the degree of the nodes
    degree_node1 = G.degree(node1)
    degree_node2 = G.degree(node2)
    #finding the preferential attachment score
    preferential_attachment = degree_node1 * degree_node2
    return preferential_attachment

def sample_negative_edges(graph, num_samples):
    """
    Sampled pair of nodes with no edges from the graph.
    Input: The graph and number of nodes to be sampled.
    Return: Dataframe containing the pair of nodes, with num_samples as the row.
    """
    # Get all nodes in the graph
    nodes = graph.nodes()

    # Find pairs of nodes that do not have any edges between them
    negative_edges = []
    sampled_count = 0

    #looping over pairs of nodes
    for pair in itertools.combinations(nodes, 2):
        if not graph.has_edge(pair[0], pair[1]):
            negative_edges.append(pair)
            sampled_count += 1

        if sampled_count == num_samples:
            break

    # Randomly shuffle the missing edges
    random.shuffle(negative_edges)

    # Create a DataFrame with the sampled negative edges
    #Dataframe contains 3 rows Node1, Node2, Edge(0)
    df = pd.DataFrame(negative_edges[:num_samples], columns=['Node1', 'Node2'])
    df = df.assign(Edge=0)
    return df

def build_model(train_df):
    """
    Building the Logistic Regression model for Link Prediction.
    Input: The training set
    Return: The Model after it is fit on the train set.
    """
    # Separate the features and target variable in the training data
    X = train_df.drop(columns=['Edge', 'Node1', 'Node2'])
    y = train_df['Edge']
    
    # Training the logistic regression model
    log_model = LogisticRegression(random_state = 42)
    log_model.fit(X, y)
    
    #Returning the fitted model
    return log_model

def evaluate_model(model, df):
    """
    The logistic regression model is tested on different datasets.
    Input: Logistic Model build and the dataframe to be tested (validation sets)
    Return: 
    """
    # Predict on the dataframe
    X_test = df.drop(columns=['Edge', 'Node1', 'Node2'])
    pred_test = model.predict(X_test)

    # The true edges in the dataframe
    true_edges = df['Edge']

    # Convert predictions to integers
    predicted_edges = pred_test.astype(int)

    # Calculate the confusion matrix
    print('Confusion Matrix is: \n', confusion_matrix(true_edges, predicted_edges))

    # Print the accuracy score
    print("Accuracy is: \n", accuracy_score(true_edges, pred_test))
    
def predict_top100_edges(df_test, model):
    #drop the source and target
    
    X_test = df_test.drop(columns=['Node1', 'Node2'])
    
    pred_val_prob = model.predict_proba(X_test)
    # Assuming you have a dataframe named 'df_test' and the probabilities in 'positive_prob' variable
    df_test['Probability'] = pred_val_prob[:, 1]
    
    df_test_sorted = df_test.sort_values(by='Probability', ascending=False)
    top_100 = df_test_sorted[:100]
    
    with open('4745511.txt', 'w') as file:
        for _, row in top_100.iterrows():
            file.write(f"{int(row['Node1'])} {int(row['Node2'])}\n")
            
    print("\nSuccessfully saved the top-100 edges in file")

def main():
    """
    The main function. Execution starts here.
    Input Parameter: None
    Return : None
    """
    #Importing the data from text file to dataframe
    df_test = pd.read_csv('test.txt', delimiter=' ', header=None, names=['Node1', 'Node2'])
    df_train = pd.read_csv('training.txt', delimiter=' ', header=None, names=['Node1', 'Node2'])
    val_negative = pd.read_csv('val_negative.txt', delimiter=' ', header=None, names=['Node1', 'Node2'])
    val_positive = pd.read_csv('val_positive.txt', delimiter=' ', header=None, names=['Node1', 'Node2'])
    
    # create a directed graph from the dataframe
    G = construct_graph(df_train)
    
    #sampling the data
    df_train['Edge'] = 1
    # Find pairs of nodes without edges
    negative_edges_sample = sample_negative_edges(G, 22992)
    # Concatenate the sample with df_train
    df_train = pd.concat([df_train, negative_edges_sample], ignore_index=True)
    
    #Calculating the proximity scores for training data
    df_train['AdamicAdar'] = df_train.apply(lambda row: adamic_adar_index(G, row['Node1'], row['Node2']), axis=1)
    df_train['TotalCommonNeighbors'] = df_train.apply(lambda row: calculate_total_neighbors_in_common(G, row['Node1'], row['Node2']), axis=1)
    df_train['JaccardSimilarity'] = df_train.apply(lambda row: calculate_jaccard_similarity(G, row['Node1'], row['Node2']), axis=1)
    df_train['PreferentialAttachment'] = df_train.apply(lambda row: calculate_preferential_attachment(G, row['Node1'], row['Node2']), axis=1)  
    
    #Combining the two validation dataframes
    val_positive['Edge'] = 1
    val_negative['Edge'] = 0
    validation_df = pd.concat([val_positive, val_negative], ignore_index=True)
    
    #Calculating the proximity scores for validation data
    validation_df['AdamicAdar'] = validation_df.apply(lambda row: adamic_adar_index(G, row['Node1'], row['Node2']), axis=1)
    validation_df['TotalCommonNeighbors'] = validation_df.apply(lambda row: calculate_total_neighbors_in_common(G, row['Node1'], row['Node2']), axis=1)
    validation_df['JaccardSimilarity'] = validation_df.apply(lambda row: calculate_jaccard_similarity(G, row['Node1'], row['Node2']), axis=1)
    validation_df['PreferentialAttachment'] = validation_df.apply(lambda row: calculate_preferential_attachment(G, row['Node1'], row['Node2']), axis=1)
    
    # Split the validation dataset into positive and negative datasets containing the scores
    val_positive = validation_df[validation_df['Edge'] == 1]
    val_negative = validation_df[validation_df['Edge'] == 0]
    
    # Build the clustering model using the training data
    model = build_model(df_train)

    #Evaluate the clustering model on the test data
    print("\nThe results from combined validation set")
    evaluate_model(model, validation_df)
    
    print("\nThe results from positive validation set")
    evaluate_model(model, val_positive)
    
    print("\nThe results from negative validation set")
    evaluate_model(model, val_negative)
        
    #performing the model on the test set
    df_test['AdamicAdar'] = df_test.apply(lambda row: adamic_adar_index(G, row['Node1'], row['Node2']), axis=1)
    df_test['TotalCommonNeighbors'] = df_test.apply(lambda row: calculate_total_neighbors_in_common(G, row['Node1'], row['Node2']), axis=1)
    df_test['JaccardSimilarity'] = df_test.apply(lambda row: calculate_jaccard_similarity(G, row['Node1'], row['Node2']), axis=1)
    df_test['PreferentialAttachment'] = df_test.apply(lambda row: calculate_preferential_attachment(G, row['Node1'], row['Node2']), axis=1)
    
    #performing predictions on the test set and outputs the result to text file
    predict_top100_edges(df_test, model)
            
if __name__ == '__main__':
    main()
