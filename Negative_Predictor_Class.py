_Author_ = "Karthik Vaidhyanathan"
_Institute_ = "Gran Sasso Science Institute"

# Script to predict the negative interactions in a positive signed network using random generated data set

import networkx as nx # The graph library, Can be downloaded from  : https://github.com/networkx/networkx
from ConfigParser import SafeConfigParser
from Logging import logger
from collections import defaultdict
import numpy as np
import json
import scipy.sparse
from scipy.sparse import csr_matrix

import itertools
import triadic # Custom library for getting the number of triads for each node

import svmpy   # For Soft margin SVM
import pickle

import dill

import traceback

from sklearn.svm import SVC

CONFIG_FILE = "settings.conf"
#CONFIG_SECTION = "random"
CONFIG_SECTION = "generator"

def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv
def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv

class Negative_Link_Predictor():
    data_path = ""
    trust_file = ""
    rating_file = ""
    author_file = ""
    json_path = ""
    num_users = 0
    num_posts = 0
    feature_path = ""
    def __init__(self):
        logger.info("Initializing the orchestrator")
        parser = SafeConfigParser()
        parser.read(CONFIG_FILE)
        try:
            self.data_path = parser.get(CONFIG_SECTION, "data_path")
            self.trust_file = parser.get(CONFIG_SECTION,"trust_file")      # This will form the signed network
            self.rating_file = parser.get(CONFIG_SECTION,"rating_file")    # For the user opinion matrix O
            self.author_file = parser.get(CONFIG_SECTION,"author_file")    # For the author content matrix A
            self.json_path = parser.get(CONFIG_SECTION,"json_path")    # For the author content matrix A
            self.num_users = int(parser.get(CONFIG_SECTION,"users"))
            self.num_posts = int(parser.get(CONFIG_SECTION,"posts"))
            self.feature_path = parser.get(CONFIG_SECTION,"feature_path")
            self.model_path = parser.get(CONFIG_SECTION,"model_path")
        except Exception as e:
            logger.error(e)

    def generate_user_author_matrix(self):
        # This function creates the matrix A
        # Gets the label from the jsons created and accordingly creates the matrix
        logger.info("Generating the user author matrix")
        try:
            file_path = self.data_path  + self.author_file
            f = open(file_path,"r")
            user_post_json = defaultdict(list)

            # for generating the csr matrix
            row_list = []
            column_list = []
            data_list  =[]

            m = self.num_users + 1
            M = self.num_posts + 1
            shape = (m,M)
            matrix = np.zeros(shape,dtype=np.int8)

            for lines in f.readlines():
                component = lines.strip("\n").split(" ")
                user = int(component[1]) # Retrive the key
                post = int(component[0]) # Retrieve the post id
                user_post_json[user].append(post)
                row_list.append(user)
                column_list.append(post)
                data_list.append(1)
                matrix[user][post] = 1


            # Get the number of users and posts

            user_pair_file = open(self.feature_path + "user_post.json", "w")
            # file_dir = self.data_path + "encoding.json"
            # user_dict = json.dumps(user_dict)
            json.dump(user_post_json, user_pair_file)

            np.save(self.json_path + "author.npy",matrix)

            # Use sparse matrix to save in case of large data sets
            sparse_author = csr_matrix((data_list, (row_list, column_list)), shape=(m, M))
            scipy.sparse.save_npz(self.json_path + "sparse_author.npz", sparse_author)
        except Exception as e:
            logger.error(e)
        logger.info("User Author matrix generated")


    def user_opinion_matrix(self):
        # This creates the user opinion matrix
        # This can be a simple json file dentoing a matrix
        logger.info("Generating user opinion matrix")
        try:
            file_path = self.data_path + self.rating_file
            f = open(file_path, "r")

            # Load the user json and post json into the memory
            m = self.num_users + 1
            M = self.num_posts + 1
            row_list = []
            column_list = []
            data_list = []
            shape   =  (m,M)
            matrix = np.zeros(shape,dtype=np.int8)

            for lines in f.readlines():
                component = lines.strip("\n").split(" ")
                post = int(component[1])
                user = int(component[0])
                rating = int(component[2])

                row_list.append(user)
                column_list.append(post)

                if rating > 3:
                    rating = 1
                else:
                    rating = -1
                new_rating = rating
                abs_rating =  (new_rating - abs(new_rating))/2
                matrix[user][post] = abs_rating
                data_list.append(abs_rating)

            np.save(self.json_path + "opnion.npy",matrix)
            sparse_opinion = csr_matrix((data_list,(row_list,column_list)),shape=(m,M))
            scipy.sparse.save_npz(self.json_path + "sparse_opnion.npz", sparse_opinion)
        except Exception as e:
            logger.error(e)
        logger.info("User opinion matrix generated")

    def generate_Negative_Interaction_matrix(self):
        # Return the inital negative sample set
        # First find the transpose of the sparse opinion and perform dot product with author matrix

        logger.info("Generating Negative Interaction matrix")
        try:
            sparse_author = scipy.sparse.load_npz(self.json_path + "sparse_author.npz")
            sparse_opinion = scipy.sparse.load_npz(self.json_path + "sparse_opnion.npz")

            #############################
            author_matrix = np.load(self.json_path + "author.npy")
            opinion_matrix = np.load(self.json_path + "opnion.npy")

            numpy_transpose_opinion = opinion_matrix.transpose()
            numpy_negative_author = author_matrix.dot(-1)
            numpy_product_matrix = numpy_negative_author.dot(numpy_transpose_opinion)
            np.save(self.json_path + "negativeInteraction_nump.npy",numpy_product_matrix)
            ################################
            transpose_opinion = sparse_opinion.transpose()
            negative_sparse_author = sparse_author.dot(-1)

            product_matrix = negative_sparse_author.dot(transpose_opinion) # store this matrix so that it can be easily accessed

            scipy.sparse.save_npz(self.json_path + "negativeInteraction_matrix.npz", product_matrix) # Save for later (input to the algo)
        except Exception as e:
            logger.error(e)
        logger.info("Negative Interaction matrix generated")

    def generate_Negative_Sample_Set(self):
        # This is to generate the set NS from the Negative_Interaction matrix
        logger.info("Generating Negative Sample set")
        negative_interaction = scipy.sparse.load_npz(self.json_path + "negativeInteraction_matrix.npz")
        #negative_interaction_numpy = np.load(self.json_path + "negativeInteraction_nump.npy")
        NS = set()  # To add all the non zero elemnts to the set
        row_column_list = zip(*negative_interaction.nonzero()) # Add the elements in a row column way to the list
        for data in row_column_list:

               if data[0]==data[1]:
                   continue
               NS.add(data)

        logger.info("Generated Negative Sample set")
        return NS

    def generate_graph(self):
        # Read the file and generate the graph
        logger.info("Generating the trust graph")
        try:
            full_file_path  = self.data_path + self.trust_file
            f=open(full_file_path,"r")

            trust_graph = nx.DiGraph() # Initialize the graph
            nodes = list(range(self.num_users))
            nodes = [str(node) for node in nodes]
            trust_graph.add_nodes_from(nodes)
            for line in f.readlines():
                graph_component = line.strip("\n").split(" ")
                user1 = graph_component[0]
                user2 = graph_component[1]
                weight = graph_component[2]
                if weight == "1":
                    trust_graph.add_edge(str(user1),str(user2),weight=1)

            negative_interaction_matrix = scipy.sparse.load_npz(self.json_path + "negativeInteraction_matrix.npz")
            NS = self.generate_Negative_Sample_Set()
            list_NS = list(NS)  # Convert the set to a list for easy access, orignal set can be used as list to avoid this confusion
            print len(list_NS)
            for row_column_pair in list_NS:
                row_index = row_column_pair[0]
                column_index = row_column_pair[1]
                if trust_graph.has_edge(row_index,column_index):
                    continue
                else:
                    trust_graph.add_edge(str(row_index),str(column_index),weight=-1)

            nx.write_gml(trust_graph.to_undirected(), self.json_path + "graph.gml")
            nx.write_gml(trust_graph, self.json_path + "graph_directed.gml")
        except Exception as e:
            logger.error(e)
        logger.info("Generated the Trust Graph")

    def generate_triads(self):
        # Load the graph.gml file and generate the triads
        logger.info("Generating Triads")
        triad_list = []
        non_status_list = []
        try:
            NS = self.generate_Negative_Sample_Set()
            trust_graph = nx.read_gml(self.json_path + "graph.gml")
            list_cliques = nx.enumerate_all_cliques(trust_graph)
            triad_cliques = [triad for triad in list_cliques if len(triad) == 3]
            small_graph = nx.DiGraph()
            trust_graph = nx.read_gml(self.json_path + "graph_directed.gml")
            for triads in triad_cliques:
                combinations = itertools.permutations(triads, 2)
                for pair in combinations:
                    # Find between whom the negative link exists
                    if trust_graph.has_edge(pair[0],pair[1]):
                        edge_data = trust_graph.get_edge_data(pair[0], pair[1])
                        if edge_data["weight"] == -1:
                            # Flip the direction change the weight and add to the small graph
                            small_graph.add_edge(pair[1],pair[0],weight=1)
                        else:
                            small_graph.add_edge(pair[0],pair[1],weight=1)
                try:
                    cycle = nx.find_cycle(small_graph)
                    non_status_list.append(triads)
                except:
                    triad_list.append(triads)
                    pass

                small_graph.clear()
        except Exception as e:
            logger.error(e)

        logger.info("Generated Triad List")
        return triad_list,non_status_list


    def remove_set_elements_NS(self,non_status_list,status_list):
        NS = list(self.generate_Negative_Sample_Set())
        try:
            trust_graph = nx.read_gml(self.json_path + "graph_directed.gml")  # Load the graph in memory
            for triads in non_status_list:
                combinations = itertools.permutations(triads, 2)
                for data in combinations:
                    if trust_graph.has_edge(data[0],data[1]):
                        edge_data = trust_graph.get_edge_data(data[0], data[1])
                        if edge_data["weight"] == -1:
                            pair = (int(data[0]),int(data[1]))
                            if pair in NS:
                                NS.remove(pair)
            # Add elements to NS if any of the triads
            NS = set(NS)
            for triads in status_list:
                combinations = itertools.permutations(triads, 2)
                for data in combinations:
                    if trust_graph.has_edge(data[0],data[1]):
                        edge_data = trust_graph.get_edge_data(data[0], data[1])
                        if edge_data["weight"] == -1:
                            pair = (int(data[0]),int(data[1]))
                            #print pair
                            NS.add(pair)
        except Exception as e:
            logger.error(e)
        return NS  # Returns the new NS after addition and removal of pairs

    def generate_reliablity_matrix(self,NS):
        reliability_dict = {}
        logger.info("Generating reliability matrix")
        try:
            negative_interaction_matrix = scipy.sparse.load_npz(self.json_path + "negativeInteraction_matrix.npz")
            for user1_user2 in NS:
                user1 = user1_user2[0]
                user2 = user1_user2[1]
                negative_interaction = negative_interaction_matrix[user1,user2]
                #print negative_interaction
                if negative_interaction == 0:
                    weight_user1_user2 = 2
                else:
                    weight_user1_user2 = negative_interaction/15.0
                #3  print weight_user1_user2
                #print round(weight_user1_user2,2)
                reliability_dict[(user1,user2)] = round(weight_user1_user2,1)
        except Exception as e:
            logger.error(e)

    def generate_user_pairs(self):

        # Generate all the possible pairs of users and add that to a json dict and all the manipulations can be done on this
        logger.info("Generating user pairs")
        user_dict = {}
        try:
            trust_graph = nx.read_gml(self.json_path + "graph_directed.gml")
            users= trust_graph.nodes()
            combinations = itertools.permutations(range(self.num_users), 2)

            for user1_user2 in combinations:
                user1 = int(user1_user2[0])
                user2 = int(user1_user2[1])
                if (user1!=user2):
                    key = str(user1) +"_"+str(user2)
                    user_dict[key] = []
            user_pair_file = open(self.feature_path + "user_pair.json", "w")
            json.dump(user_dict, user_pair_file)
        except Exception as e:
            logger.error(e)

        logger.info("User pairs Generated")




    def user_feature_generator(self,triad_list):
        # Add all the user pairs in a json
        logger.info("Generating user features")
        # For a pair of users we need to generate the user features
        # Load the user_pair json and for each pair generate the corresponding features
        try:
            trust_graph = nx.read_gml(self.json_path + "graph_directed.gml")
            #census = nx.triadic_census(trust_graph)

            user_pair_file = open(self.feature_path + "user_pair.json", "r")
            user_pair_json = json.load(user_pair_file, object_hook=_decode_dict)
            user_pair_file.close()

            user_post_file = open(self.feature_path + "user_post.json", "r")
            user_post_json = json.load(user_post_file, object_hook=_decode_dict)

            # For getting the number of posetive and negative reviews that user has expressed
            #sparse_author = scipy.sparse.load_npz(self.json_path + "sparse_author.npz")
            #sparse_opinion = scipy.sparse.load_npz(self.json_path + "sparse_opnion.npz")
            author_matrix = np.load(self.json_path + "author.npy")
            opnion_matrix = np.load(self.json_path + "opnion.npy")
            #census,node_census = triadic.triadic_census(trust_graph)
            count = 0
            for pair in user_pair_json:
                users = pair.split("_")
                user1 = int(users[0])
                user2 = int(users[1])

                # outdegree and indegree of users in terms of the number of positive links
                user1_outdegree = trust_graph.out_degree(str(user1),weight = 1)
                user1_indegree = trust_graph.in_degree(str(user1),weight = 1)
                user2_outdegree = trust_graph.out_degree(str(user2),weight = 1)
                user2_indegree = trust_graph.in_degree(str(user2), weight = 1)

                if isinstance(user1_outdegree,nx.classes.reportviews.OutDegreeView):
                    user1_outdegree = 0
                #[print user1_outdegree
                if isinstance(user2_outdegree,nx.classes.reportviews.OutDegreeView):
                    user2_outdegree = 0

                if isinstance(user1_indegree,nx.classes.reportviews.InDegreeView):
                    user1_indegree = 0

                if isinstance(user2_indegree, nx.classes.reportviews.InDegreeView):
                    user2_indegree = 0

                # Get the count of the number of triads in which user1 and user2 is a part of

                user1_triad_count = 0
                user2_triad_count = 0
                #user1_census = node_census[str(user1)]

                for triad in triad_list:
                    if str(user1) in triad:
                        user1_triad_count = user1_triad_count + 1
                    if str(user2) in triad:
                        user2_triad_count = user2_triad_count + 1

                user1_content_count = 0

                user1_content_row = author_matrix[user1,:]
                for columns in user1_content_row:
                    if columns > 0:
                        user1_content_count += 1

                user2_content_count = 0

                user2_content_row = author_matrix[user2, :]
                for columns in user2_content_row:
                    if columns > 0:
                        user2_content_count += 1

                user1_receive_postive_count = 0
                user1_receive_negative_count = 0
                if str(user1) in user_post_json:
                # For user 1 and user 2 calculate the number of positive and negative counts for each post
                    for post in user_post_json[str(user1)]:
                        # Check the total posetive count received by the post
                        post_count = opnion_matrix[:,post]
                        for rating in post_count:
                            if rating == 1:
                                user1_receive_postive_count += 1
                            elif rating == -1:
                                user1_receive_negative_count += 1

                # user 2
                user2_receive_postive_count = 0
                user2_receive_negative_count = 0

                if str(user2) in user_post_json:
                    for post in user_post_json[str(user2)]:
                        # Check the total posetive count received by the post
                        post_count = opnion_matrix[:,post]
                        for rating in post_count:
                            if rating == 1:
                                user2_receive_postive_count += 1
                            elif rating == -1:
                                user2_receive_negative_count += 1

                # Count the number of posts for which user 1 gave either 1 or -1

                user1_give_positive_count = 0
                user1_give_negative_cont = 0


                user1_express = opnion_matrix[user1,:]

                for rating in user1_express:
                    if rating == 1:
                        user1_give_positive_count += 1
                    elif rating == -1:
                        user1_give_negative_cont += 1



                user2_give_positive_count = 0
                user2_give_negative_cont = 0

                user2_express = opnion_matrix[user2, :]

                for rating in user2_express:
                    if rating == 1:
                        user2_give_positive_count += 1
                    elif rating == -1:
                        user2_give_negative_cont += 1



                # for each user start inserting all the data in the list
                user_feature_list = []

                user_feature_list.append(user1_outdegree)
                user_feature_list.append(user1_indegree)
                user_feature_list.append(user2_outdegree)
                user_feature_list.append(user2_indegree)
                user_feature_list.append(user1_triad_count)
                user_feature_list.append(user2_triad_count)
                user_feature_list.append(user1_content_count)
                user_feature_list.append(user2_content_count)
                user_feature_list.append(user1_receive_postive_count)
                user_feature_list.append(user1_receive_negative_count)
                user_feature_list.append(user2_receive_postive_count)
                user_feature_list.append(user2_receive_negative_count)
                user_feature_list.append(user1_give_positive_count)
                user_feature_list.append(user1_give_negative_cont)
                user_feature_list.append(user2_give_positive_count)
                user_feature_list.append(user2_give_negative_cont)
                user_pair_json[pair] = user_feature_list
                count+=1

            user_pair_file = open(self.feature_path + "user_feature.json", "w")
            json.dump(user_pair_json, user_pair_file)
            user_pair_file.close()

        except Exception as e:
            logger.error(e)

        logger.info("Generated User Features")

    def user_pair_feature_generator(self):
        # generate the features for each pair of users

        logger.info("Generating feature for user pairs")
        try:
            trust_graph = nx.read_gml(self.json_path + "graph_directed.gml")

            # Load the pairs of the users from the json file
            user_pair_file = open(self.feature_path + "user_pair.json", "r")
            user_pair_json = json.load(user_pair_file, object_hook=_decode_dict)



            user_pair_file.close()

            user_post_file = open(self.feature_path + "user_post.json","r")
            user_post_json = json.load(user_post_file,object_hook=_decode_dict)

            opnion_matrix = np.load(self.json_path + "opnion.npy")     # Load the opinion matrix

            count = 0
            pair_feature_json = {}     # Keep adding this to save into the json file

            for pair in user_pair_json:
                users = pair.split("_")
                user1 = int(users[0])
                user2 = int(users[1])

                count_positive_u1_u2 = 0        # Count of the posetive interaction from user 1 to user 2
                count_negative_u1_u2 = 0        # Count of the negative interaction from user 1 to user 2

                #print user1
                #print user2
                for posts in user_post_json[str(user2)]:
                    if opnion_matrix[user1,posts] == 1:
                        count_positive_u1_u2 += 1
                    elif opnion_matrix[user1,posts] == -1:
                        count_negative_u1_u2 += 1



                count_positive_u2_u1 = 0
                count_negative_u2_u1 = 0

                for posts in user_post_json[str(user1)]:
                    if opnion_matrix[user2, posts] == 1:
                        count_positive_u2_u1 += 1
                    elif opnion_matrix[user2, posts] == -1:
                        count_negative_u2_u1 += 1

                # Find Jaccard Coefficients

                in_edges_user1 = []
                in_edges_user2 = []
                out_edges_user1 = []
                out_edges_user2 = []  # To find the jaccard coefficients

                in_user1 = trust_graph.in_edges(str(user1),data=True)
                in_user2 = trust_graph.in_edges(str(user2),data=True)


                out_user1 = trust_graph.out_edges(str(user2),data=True)
                out_user2 = trust_graph.out_edges(str(user2),data=True)

                #print in_edges
                for edges in in_user1:
                    if edges[2]["weight"] == 1:
                        # Adding the positive links
                        in_edges_user1.append((edges[0],edges[1]))


                for edges2 in in_user2:
                    if edges2[2]["weight"] == 1:
                        # Adding the positive links
                        in_edges_user2.append((edges2[0],edges2[1]))


                jaccard_indegree_positive = 0.0
                if not(len(in_edges_user1) == 0 or len(in_edges_user2) == 0):
                    jaccard_indegree_positive = len(set(list(in_edges_user1)).intersection(set(in_edges_user2)))/float(len(set(list(in_edges_user1)).union(set(in_edges_user2))))
                    jaccard_indegree_positive = round(jaccard_indegree_positive,1)


                for edges in out_user1:
                    if edges[2]["weight"] == 1:
                        # Adding the positive links
                        out_edges_user1.append((edges[0],edges[1]))

                for edges2 in out_user2:
                    if edges2[2]["weight"] == 1:
                        # Adding the positive links
                        out_edges_user2.append((edges2[0],edges2[1]))

                jaccard_outdegree_positive = 0.0
                if not(len(out_edges_user1) == 0 or len(out_edges_user2) == 0):
                    jaccard_outdegree_positive = len(set(list(out_edges_user1)).intersection(set(out_edges_user2)))/float(len(set(list(out_edges_user1)).union(set(out_edges_user2))))
                    jaccard_outdegree_positive = round(jaccard_outdegree_positive,1)

                shortest_path_distance = 100     # Set this to a much higher value
                if nx.has_path(trust_graph,str(user1),str(user2)):
                    shortest_path_distance = nx.shortest_path_length(trust_graph,str(user1),str(user2))

                pair_feature_list = []
                pair_feature_list.append(count_positive_u1_u2)
                pair_feature_list.append(count_negative_u1_u2)
                pair_feature_list.append(count_positive_u2_u1)
                pair_feature_list.append(count_negative_u2_u1)
                pair_feature_list.append(jaccard_indegree_positive)
                pair_feature_list.append(jaccard_outdegree_positive)
                pair_feature_list.append(shortest_path_distance)

                pair_feature_json[pair] = pair_feature_list

            # Save eveything to a json file

            pair_feature_file = open(self.feature_path + "pair_feature.json", "w")
            json.dump(pair_feature_json, pair_feature_file)
            pair_feature_file.close()

        except Exception as e:
            logger.error(e)

        logger.info("Generated Paired Features")

    def signed_feature_generator(self,triad_list):
        logger.info("Generating signed feature for user pairs")
        try:
            trust_graph = nx.read_gml(self.json_path + "graph_directed.gml")

            # Load the pairs of the users from the json file
            user_pair_file = open(self.feature_path + "user_pair.json", "r")
            user_pair_json = json.load(user_pair_file, object_hook=_decode_dict)

            user_pair_file.close()
            census,nodes= triadic.triadic_census(trust_graph)
            #print nodes
            signed_feature_json = {}
            for pair in user_pair_json:
                users = pair.split("_")
                user1 = int(users[0])
                user2 = int(users[1])

                signed_feature_list = []
                # To calculate the weighted indegree and outdegree in terms of negative links of user1 and user2
                in_user1_count = 0
                out_user1_count = 0

                in_user2_count = 0
                out_user2_count = 0

                in_user1 = trust_graph.in_edges(str(user1), data=True)
                out_user1 = trust_graph.out_edges(str(user1),data=True)

                in_user2 = trust_graph.in_edges(str(user2), data=True)
                out_user2 = trust_graph.out_edges(str(user2), data=True)

                in_edges_user1 = []
                in_edges_user2 = []
                out_edges_user1 = []
                out_edges_user2 = []  # To find the jaccard coefficients

                for edges in in_user1:
                    if edges[2]["weight"] == -1:
                        # Adding the negative links for jaccard coefficients and increment to the weighted indgreee
                        in_edges_user1.append((edges[0],edges[1]))
                        in_user1_count += 1

                for edges2 in in_user2:
                    if edges2[2]["weight"] == -1:
                        # Adding the positive links
                        in_edges_user2.append((edges2[0],edges2[1]))
                        in_user2_count += 1


                for edges in out_user1:
                    if edges[2]["weight"] == -1:
                        # Adding the positive links
                        out_edges_user1.append((edges[0],edges[1]))
                        out_user1_count += 1

                for edges2 in out_user2:
                    if edges2[2]["weight"] == -11:
                        # Adding the positive links
                        out_edges_user2.append((edges2[0],edges2[1]))
                        out_user2_count += 1


                jaccard_indegree_negative = 0.0

                if not (len(in_edges_user1) == 0 or len(in_edges_user2) == 0):
                    jaccard_indegree_negative = len(set(list(in_edges_user1)).intersection(set(in_edges_user2))) / float(
                        len(set(list(in_edges_user1)).union(set(in_edges_user2))))
                    jaccard_indegree_negative = round(jaccard_indegree_negative, 1)


                jaccard_outdegree_negative = 0.0
                if not (len(out_edges_user1) == 0 or len(out_edges_user2) == 0):
                    jaccard_outdegree_positive = len(set(list(out_edges_user1)).intersection(set(out_edges_user2))) / float(
                        len(set(list(out_edges_user1)).union(set(out_edges_user2))))
                    jaccard_outdegree_negative= round(jaccard_outdegree_negative, 1)


                #print nodes[str(user1)]
                #print nodes[str(user2)]
                signed_feature_list.append(in_user1_count)
                signed_feature_list.append(out_user1_count)
                signed_feature_list.append(in_user2_count)
                signed_feature_list.append(out_user2_count)
                signed_feature_list.append(jaccard_indegree_negative)
                signed_feature_list.append(jaccard_outdegree_negative)

                signed_triad_list = self.pairwise_triadic_census(str(user1),str(user2),trust_graph,triad_list)
                #print signed_triad_list

                signed_feature_list.extend(signed_triad_list)
                #print signed_feature_list
                signed_feature_json[pair] = signed_feature_list



            # Create the sigened feature json
            signed_feature_file = open(self.feature_path + "signed_feature.json", "w")
            json.dump(signed_feature_json, signed_feature_file)
            signed_feature_file.close()

        except Exception as e:
            logger.error(e)

        logger.info("Generated Signed features")

    def pairwise_triadic_census(self,node_u,node_v, graph,triadlist):
        # Gives the count of each type of triads the nodes are part of
        triad_type_dict = {}
        signed_triad_list = []
        logger.info("Generating Pairwise Triadic Census")
        '''
        For the second class of feature we consider each triad involving the edge (u,v), 
        consisting of a node w such that w has an edge either to or from u and also an edge either to or from v. 
        There are 16 distinct types of triads involving (u, v): the edge between w and u can be in either direction and 
        of either sign, and the edge between w and v can also be in either direction and of either sign; this leads to
        2.2.2.2 = 16 possibilities. Each of these 16 triad types may provide different evidence about the 
        sign of the edge from u to v, some favoring a negative sign and some favoring a positive sign.We encode this 
        information in a 16-dimensional vector specifying the number of triads of each type that (u, v) is involved in.
        '''
        try:
            count_wu_posetive_wv_posetive = 0
            count_wu_posetive_wv_negative = 0
            count_wu_negative_wv_posetive = 0
            count_wu_negative_wv_negative = 0

            count_uw_posetive_vw_posetive = 0
            count_uw_posetive_vw_negative = 0
            count_uw_negative_vw_posetive = 0
            count_uw_negative_vw_negative = 0

            count_uw_posetive_wv_posetive = 0
            count_uw_posetive_wv_negative = 0
            count_uw_negative_wv_posetive = 0
            count_uw_negative_wv_negative= 0

            count_wu_posetive_vw_posetive = 0
            count_wu_posetive_vw_negative = 0
            count_wu_negative_vw_posetive = 0
            count_wu_negative_vw_negative = 0

            for triad in triadlist:
                if node_u in triad and node_v in triad :
                    set_w = set(triad).difference([str(node_u),str(node_v)])
                    node_w = str(set_w.pop())
                    if graph.has_edge(str(node_w),node_u) and graph.has_edge(str(node_w),node_v):
                        edge_data_wu = graph.get_edge_data(node_w,node_u)
                        edge_data_wv = graph.get_edge_data(node_w, node_v)
                        if edge_data_wu["weight"] == 1 and edge_data_wv["weight"] == 1:
                            count_wu_posetive_wv_posetive +=1
                        if edge_data_wu["weight"] == 1 and edge_data_wv["weight"] == -1:
                            count_wu_posetive_wv_negative += 1
                        if edge_data_wu["weight"] == -1 and edge_data_wv["weight"] == 1:
                            count_wu_negative_wv_posetive += 1
                        if edge_data_wu["weight"] == -1 and edge_data_wv["weight"] == -1:
                            count_wu_negative_wv_negative += 1


                    if graph.has_edge(node_u,str(node_w)) and graph.has_edge(node_v,str(node_w)):
                        edge_data_uw = graph.get_edge_data(node_u, node_w)
                        edge_data_vw = graph.get_edge_data(node_v, node_w)

                        if edge_data_uw["weight"] == 1 and edge_data_vw["weight"] == 1:
                            count_uw_posetive_vw_posetive += 1
                        if edge_data_uw["weight"] == 1 and edge_data_vw["weight"] == -1:
                            count_uw_posetive_vw_negative += 1
                        if edge_data_uw["weight"] == -1 and edge_data_vw["weight"] == 1:
                            count_uw_negative_vw_posetive += 1
                        if edge_data_uw["weight"] == -1 and edge_data_vw["weight"] == -1:
                            count_uw_negative_vw_negative += 1

                    if graph.has_edge(node_u,str(node_w)) and graph.has_edge(str(node_w),node_v):
                        edge_data_uw = graph.get_edge_data(node_u, node_w)
                        edge_data_wv = graph.get_edge_data(node_w, node_v)

                        if edge_data_uw["weight"] == 1 and edge_data_wv["weight"] == 1:
                            count_uw_posetive_wv_posetive += 1
                        if edge_data_uw["weight"] == 1 and edge_data_wv["weight"] == -1:
                            count_uw_posetive_wv_negative += 1
                        if edge_data_uw["weight"] == -1 and edge_data_wv["weight"] == 1:
                            count_uw_negative_wv_posetive += 1
                        if edge_data_uw["weight"] == -1 and edge_data_wv["weight"] == -1:
                            count_uw_negative_wv_negative += 1

                    if graph.has_edge(str(node_w),node_u) and graph.has_edge(node_v,str(node_w)):
                        edge_data_wu = graph.get_edge_data(node_w, node_u)
                        edge_data_vw = graph.get_edge_data(node_v, node_w)

                        if edge_data_wu["weight"] == 1 and edge_data_vw["weight"] == 1:
                            count_wu_posetive_vw_posetive += 1
                        if edge_data_wu["weight"] == 1 and edge_data_vw["weight"] == -1:
                            count_wu_posetive_vw_negative += 1
                        if edge_data_wu["weight"] == -1 and edge_data_vw["weight"] == 1:
                            count_wu_negative_vw_posetive += 1
                        if edge_data_wu["weight"] == -1 and edge_data_vw["weight"] == -1:
                            count_wu_negative_vw_negative += 1

            signed_triad_list = []
            signed_triad_list.append(count_wu_posetive_wv_posetive)
            signed_triad_list.append(count_wu_posetive_wv_negative)
            signed_triad_list.append(count_wu_negative_wv_posetive)
            signed_triad_list.append(count_wu_negative_wv_negative)

            signed_triad_list.append(count_uw_posetive_vw_posetive)
            signed_triad_list.append(count_uw_posetive_vw_negative)
            signed_triad_list.append(count_uw_negative_vw_posetive)
            signed_triad_list.append(count_uw_negative_vw_negative)

            signed_triad_list.append(count_uw_posetive_wv_posetive)
            signed_triad_list.append(count_uw_posetive_wv_negative)
            signed_triad_list.append(count_uw_negative_wv_posetive)
            signed_triad_list.append(count_uw_negative_wv_negative)

            signed_triad_list.append(count_wu_posetive_vw_posetive)
            signed_triad_list.append(count_wu_posetive_vw_negative)
            signed_triad_list.append(count_wu_negative_vw_posetive)
            signed_triad_list.append(count_wu_negative_vw_negative)

        except Exception as e:
            logger.error(e)

        logger.info("Generated 16 triad features")
        return signed_triad_list


    def feature_combiner(self,NS):
        # This will combine all the features for a user pair and then this is used for SVM
        # Get the negative sample set and generate labels
        logger.info("Combining Features")
        features = []
        transpose_labels = []
        label_list = []
        try:
            # Get all user pairs
            user_pair_file = open(self.feature_path + "user_pair.json", "r")
            user_pair_json = json.load(user_pair_file, object_hook=_decode_dict)
            user_pair_file.close()

            # Get all the user features created
            user_feature_file = open(self.feature_path + "user_feature.json", "r")
            user_feature_json = json.load(user_feature_file, object_hook=_decode_dict)
            user_feature_file.close()

            # Get all the pair features created
            pair_feature_file = open(self.feature_path + "pair_feature.json", "r")
            pair_feature_json = json.load(pair_feature_file, object_hook=_decode_dict)
            pair_feature_file.close()

            # Get all the signed features created
            signed_feature_file = open(self.feature_path + "signed_feature.json", "r")
            signed_feature_json = json.load(signed_feature_file,object_hook=_decode_dict)
            signed_feature_file.close()

            # Load the trust graph
            trust_graph = nx.read_gml(self.json_path + "graph_directed.gml")



            final_feature_list =[]
            label_list = []
            main_feature_json  = {}
            for user in user_pair_json:
                feature_list = []
                #if user in user_feature_json:
                user1 = int(user.split("_")[0])
                user2 = int(user.split("_")[1])
                if trust_graph.has_edge(str(user1), str(user2)):
                    edge_data = trust_graph.get_edge_data(str(user1), str(user2))
                    if edge_data["weight"] == -1:
                        if (user1,user2) in NS:
                            feature_list.extend(user_feature_json[user])
                            feature_list.extend(pair_feature_json[user])
                            feature_list.extend(signed_feature_json[user])
                            final_feature_list.append(feature_list)
                            main_feature_json[user] = feature_list
                            label_list.append(-1)
                    else:
                        feature_list.extend(user_feature_json[user])
                        feature_list.extend(pair_feature_json[user])
                        feature_list.extend(signed_feature_json[user])
                        final_feature_list.append(feature_list)
                        main_feature_json[user] = feature_list
                        label_list.append(edge_data["weight"])


            # Create the final feature json
            final_feature_file = open(self.feature_path + "final_feature.json", "w")
            json.dump(main_feature_json, final_feature_file)
            final_feature_file.close()

            #$print len(main_feature_json.keys())
            features = np.asmatrix(final_feature_list).astype(np.double) # To make it ready to be processed for SVM
            labels = np.asmatrix(label_list).astype(np.double)
            transpose_labels = labels.transpose()
        except Exception as e:
            logger.error(e)

        return features,transpose_labels,label_list


    def link_predictor(self,features,labels):
        logger.info("SVM Training processing")
        #trainer = svmpy.SVMTrainer(svmpy.Kernel.gaussian(0.5),0.05)
        trainer = svmpy.SVMTrainer(svmpy.Kernel.radial_basis(2.0),1.0)
        predictor = trainer.train(features, labels)


        #print predictor

        # Save the predictor object

        with open(self.model_path + "predictor.pkl", 'wb') as output:  # Overwrites any existing file.
            pickle.dump(predictor, output, pickle.HIGHEST_PROTOCOL)   # Dump as a pickle object
        logger.info("Training completed")

    def link_predictor_normalSVM(self,features,labels):
        logger.info("SVM Normal SVC Classification")
        clf = SVC(C=1.0,gamma=2.0,kernel="rbf")
        predictor = clf.fit(features, labels)
        with open(self.model_path + "predictor_scipySVM.pkl", 'wb') as output:  # Overwrites any existing file.
            pickle.dump(predictor, output, pickle.HIGHEST_PROTOCOL)   # Dump as a pickle object




    def predict_users(self,user1,user2):
        # Get the user 1 and user 2 for which the link has to been predicted

        # Get the feature vector for the given user
        category  = []
        logger.info("Generating predictions for " + str(user1) + " " + str(user2))
        try:
            feature_json_file =  open(self.feature_path + "final_feature.json", "r")
            feature_json = json.load(feature_json_file,object_hook=_decode_dict)
            feature_json_file.close()
            predictor = None
            with open(self.model_path + 'predictor.pkl', 'rb') as input:
                predictor = pickle.load(input)

            category = None
            #print len(feature_json.keys())
            user_pair = str(user1) + "_" + str(user2)
            if user_pair not in feature_json:
                user_feature_file = open(self.feature_path + "user_feature.json", "r")
                user_feature_json = json.load(user_feature_file, object_hook=_decode_dict)
                user_feature_file.close()

                # Get all the pair features created
                pair_feature_file = open(self.feature_path + "pair_feature.json", "r")
                pair_feature_json = json.load(pair_feature_file, object_hook=_decode_dict)
                pair_feature_file.close()

                # Get all the signed features created
                signed_feature_file = open(self.feature_path + "signed_feature.json", "r")
                signed_feature_json = json.load(signed_feature_file, object_hook=_decode_dict)
                signed_feature_file.close()

                feature_list = []
                feature_list.extend(user_feature_json[user_pair])
                feature_list.extend(pair_feature_json[user_pair])
                feature_list.extend(signed_feature_json[user_pair])
                feature_matrix = np.asmatrix(feature_list, np.double)
                category = predictor.predict(feature_matrix)
            else:
                feature_vector = feature_json[user_pair]
                feature_matrix = np.asmatrix(feature_vector,np.double)
                category = predictor.predict(feature_matrix)
        except Exception as e:
            logger.error(e)
        return [category]

    def predict_users_normal_svm(self,user1,user2):
        # Get the user 1 and user 2 for which the link has to been predicted

        # Get the feature vector for the given user
        feature_json_file =  open(self.feature_path + "final_feature.json", "r")
        feature_json = json.load(feature_json_file,object_hook=_decode_dict)

        with open(self.model_path + 'predictor_scipySVM.pkl', 'rb') as input:
            predictor = pickle.load(input)
        category = None
        #print len(feature_json.keys())
        user_pair = str(user1) + "_" + str(user2)
        if user_pair not in feature_json:
            user_feature_file = open(self.feature_path + "user_feature.json", "r")
            user_feature_json = json.load(user_feature_file, object_hook=_decode_dict)
            user_feature_file.close()

            # Get all the pair features created
            pair_feature_file = open(self.feature_path + "pair_feature.json", "r")
            pair_feature_json = json.load(pair_feature_file, object_hook=_decode_dict)
            pair_feature_file.close()

            # Get all the signed features created
            signed_feature_file = open(self.feature_path + "signed_feature.json", "r")
            signed_feature_json = json.load(signed_feature_file, object_hook=_decode_dict)
            signed_feature_file.close()

            feature_list = []
            feature_list.extend(user_feature_json[user_pair])
            feature_list.extend(pair_feature_json[user_pair])
            feature_list.extend(signed_feature_json[user_pair])
            feature_matrix = np.asmatrix(feature_list, np.double)
            category = predictor.predict(feature_matrix)
        else:
            feature_vector = feature_json[user_pair]
            feature_matrix = np.asmatrix(feature_vector,np.double)
            category = predictor.predict(feature_matrix)
        #print category
        return category


    def accuracy_check(self):
        # Read the trust file and check for false posetives and false negatives
        tp = 0  # True Positive
        tn = 0  # True Negative
        fp = 0  # False Positve
        fn = 0  # False Negative
        logger.info("Generating accuracy")
        try:
            full_file_path = self.data_path + self.trust_file
            f = open(full_file_path, "r")
            for line in f.readlines():
                graph_component = line.strip("\n").split(" ")
                #print graph_component
                #break
                user1 = graph_component[0]
                user2 = graph_component[1]
                weight = int(graph_component[2])
                if user1 == user2:
                    continue
                #cat = negative_link_predictor_object.predict_users_normal_svm(user1,user2)
                cat = negative_link_predictor_object.predict_users(user1,user2)
                if cat is None:
                    continue
                if weight == -1 and cat[0] == -1.0:
                    tp += 1
                elif weight == 1 and cat[0] == -1.0:
                    fp += 1

                elif weight == 1 and cat[0] == 1.0:
                    tn+= 1
                elif weight == -1 and cat[0] == 1.0:
                    fn += 1
            logger.info("Confusion matrix in the order tp, tn, fp, fn " + str(tp) + " " + str(tn) + " " +  str(fp) + " " + str(fn))
        except Exception as e:
            logger.error(e)
            traceback.print_exc()
        #print tp,tn,fp,fn



    def recommendor(self,posts_dict):
        # Creates the final dense graph between all pairs of users using the negative link prediction
        try:
            logger.info("Generating recommendations")
            # For each user generate the recommendations
            user_pair_file = open(self.feature_path + "user_pair.json", "r")
            user_pair_json = json.load(user_pair_file, object_hook=_decode_dict)
            user_pair_file.close()
            user_count = 0
            user_recommendor = {}
            for user in range(0,self.num_users,1):
                user_recommendor[str(user)] = []
            for user1 in range(0,self.num_users,1):
                user_list = []
                for user2 in range(0,self.num_users,1):
                    if user1 == user2:
                        continue
                    cat = negative_link_predictor_object.predict_users(user1, user2)
                    #print cat
                    if cat[0] == 1.0:
                        user_list.append(str(user2))

                user_recommendor[str(user1)].extend(self.find_top_users(posts_dict,user_list))
                user_count += 1
                print "processing user : " + str(user_count)
            user_recommendor_file = open(self.feature_path + "recommendation.json", "w")
            json.dump(user_recommendor, user_recommendor_file)
            user_recommendor_file.close()
            logger.info("Recommendations generated")
        except Exception as e:
            traceback.print_exc()
            logger.error(e)



    def generate_item_recommendations(self,user_id,top_n=10):
        # This genrates the item reccomendations based on the user recommendations generated
        # Top n is the number of recommendations to be generated by default 10 recoomendations are generated
        try:
            logger.info("Generating item based recommendations")
            user_pair_file = open(self.feature_path + "user_pair.json", "r")
            user_pair_json = json.load(user_pair_file, object_hook=_decode_dict)
            user_pair_file.close()

            recommendation_file  = open(self.feature_path + "recommendation.json", "r")
            recommendation_json = json.load(recommendation_file,object_hook=_decode_dict)
            #print len(recommendation_json.keys())
            if str(user_id) in recommendation_json:
                print recommendation_json[str(user_id)]
                print len(recommendation_json[str(user_id)])
        except Exception as e:
            traceback.print_exc()
            logger.error(e)


    def find_top_users(self,post_dict,users):
        # print sorted_ratings
        # Get the set of users for which the ranking has to be done
        top_user_keys = []
        logger.info("Generating the top 10 users")
        try:
            user_post_file = open(self.feature_path + "user_post.json", "r")
            user_post_json = json.load(user_post_file, object_hook=_decode_dict)
            user_post_file.close()

            post_file = open(self.json_path +"post.json","r")
            post_json = json.load(post_file,object_hook=_decode_dict)
            post_file.close()
            inv_post = {v: k for k, v in post_json.iteritems()} # We need to find the post id of a post label

            user_popularity_dict = {}
            for user in users:
                #print user
                ratings_sum = 0
                for posts in user_post_json[user]:
                    if posts in post_dict:
                        ratings_avg = post_dict[posts]
                        ratings_sum = ratings_sum + ratings_avg
                user_popularity_dict[user] = ratings_sum

            top_users_dict = sorted(user_popularity_dict.items(), key=lambda x: x[1],reverse=True)
            top_user_keys = [pair[0] for pair in top_users_dict[:10]]
        except Exception as e:
            logger.error(e)
        return top_user_keys


    def ratings_mean_generator(self):
        # This is used in recommendation to give the ranking of the recommendations
        # First read the ratings file
        file_path = self.data_path + self.rating_file
        f = open(file_path,"r")
        dict_post  = {}


        for lines in f.readlines():
            component = lines.strip("\n").split(" ")
            post = int(component[1])
            user = int(component[0])
            rating = int(component[2])
            if post in dict_post:
                dict_post[post] = dict_post[post] + rating
            else:
                dict_post[post] = rating
        print dict_post
        return dict_post


    def find_MAP(self):
        # This basically finds the MAP score of the recommendor system for the given dataset
        # Using the trust data set for validation
        # For every user the recommendations are generated and checked if there is any negative samples in them
        # For each user find the list of recommendations
        # Precision - is a percentage of correct items among first i recommendations where i = 10 in our case
        # Recalli equals 1/n if the ith item is correct and 0 otherwise
        # If all items retrieved are correct then precision equals 1 and total recall equals 1 and total AP is 1
        # MAP is the average of all AP's

        logger.info("Generating MAP")
        # Find all the unique users in the trusted data. Get the recommendations from reccommendation json
        try:
            reccommendation_file = open(self.feature_path + "recommendation.json","r")
            reccommendation_json = json.load(reccommendation_file,object_hook=_decode_dict)


            f = open(self.data_path+self.trust_file,"r")
            users = set()  # To ensure that the users are not repeated
            user_map = defaultdict(list) # For every user add the users which he/she has liked
            for line in f.readlines():
                graph_component = line.strip("\n").split(" ")
                #print graph_component
                #break
                user1 = graph_component[0]
                user2 = graph_component[1]
                weight = int(graph_component[2])

                users.add(user1)
                users.add(user2)
                if weight == -1:
                    user_map[user1].append(user2)

            ap_list = []  # for each of the recommendations performed find the average precision
            user_count  = 0
            for user in list(users):
                if user in user_map:
                    user_count += 1
                    reccommendation_list = reccommendation_json[user]
                    total_count = len(reccommendation_list)      # By default this is 10
                    n_count = 0
                    negative_list = user_map[user]
                    #print set(reccommendation_list).intersection(set(original_recommendation))
                    p_count = 0 # For checking the precision
                    rel_list = []
                    for item in reccommendation_list:
                        if item in negative_list:
                            rel_list.append(0.0)  # says that the recommendtion at that point was irrelavant
                            n_count += 1
                        else:
                            p_count += 1
                            n_count += 1
                            rel_list.append(p_count/float(n_count))
                    sum_precision = 0
                    for data in rel_list:
                        sum_precision = sum_precision+data

                    ap_user = round((1/float(total_count)) * sum_precision,2)
                    ap_list.append(ap_user)


            map_score = 0
            for data in ap_list:
                map_score = map_score + data

            map_score = round((1/float(user_count)) * map_score,3)
            print map_score
            logger.info("Mean Average Precision "+ str(map_score))
        except Exception as e:
            logger.error(e)
            traceback.print_exc()







if __name__ == '__main__':
    negative_link_predictor_object = Negative_Link_Predictor()
    negative_link_predictor_object.generate_user_author_matrix()
    #negative_link_predictor_object.user_opinion_matrix()
    #negative_link_predictor_object.generate_Negative_Interaction_matrix()
    #NS = negative_link_predictor_object.generate_Negative_Sample_Set()
    #print len(NS)
    #negative_link_predictor_object.generate_graph()
    #triad_list,non_status_list = negative_link_predictor_object.generate_triads()
    #print triad_list
    #print non_status_list
    #print len(triad_list)
    #print non_status_list
    #print len(non_status_list)
    #updated_NS = negative_link_predictor_object.remove_set_elements_NS(non_status_list,triad_list)
    #print updated_NS
    #print len(updated_NS)
    #negative_link_predictor_object.generate_user_pairs()

    #negative_link_predictor_object.generate_reliablity_matrix(updated_NS)
    #triad_list.extend(non_status_list)


    #negative_link_predictor_object.user_feature_generator(triad_list)
    #negative_link_predictor_object.user_pair_feature_generator()
    #negative_link_predictor_object.signed_feature_generator(triad_list)

    #features,labels,label_list = negative_link_predictor_object.feature_combiner(updated_NS)
    #for label in label_list:
    #    if label==-1:
    #        print label
    #print features.shape
    #print labels.shape

    #print "starting svm"
    #negative_link_predictor_object.link_predictor(features,labels)
    #cat = negative_link_predictor_object.predict_users(75,86)
    #print cat
    #negative_link_predictor_object.link_predictor_normalSVM(features, label_list)  # SVM in SCIpy

    #negative_link_predictor_object.accuracy_check()
    #negative_link_predictor_object.recommendor()
    #negative_link_predictor_object.generate_item_recommendations(20)
    #features = features.transpose()
    #labels = labels.transpose()
    #negative_link_predictor_object.find_top_users()
    #post_rating_dict = negative_link_predictor_object.ratings_mean_generator()
    #negative_link_predictor_object.find_top_users(post_rating_dict)
    #negative_link_predictor_object.recommendor(post_rating_dict)

    negative_link_predictor_object.find_MAP()


    '''
    Results
    Confusion matrix in the order tp, tn, fp, fn
    C =  1, 142, 0, 158, 0  0.47
    C =  0.1, 110 30 128 32 0.46
    C =  0.5 141,4,154,1   0.47
    
    
    
    C = 1, Sigma = 0.5  13 424 32 40
    C = 0.5, Sigma = 0.5 13 424 32 40
    C = 1, sigma = 0.6
    
    
    '''




