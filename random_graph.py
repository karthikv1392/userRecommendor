_Author_ = "Karthik Vaidhyanathan"
# Generating random graph to simulate the real world networks

import networkx as nx
import random

def generate_graph(nodes,edges):
    random_graph = nx.dense_gnm_random_graph(nodes,edges)
    possible_weight = [-1,1]
    for (u, v, w) in random_graph.edges(data=True):
        w['weight'] = random.choice(possible_weight)

    for (u, v, w) in random_graph.edges(data=True):
        print u,v,w


    return random_graph


def find_triads(graph):
    trust_graph = graph.to_undirected()
    list_cliques = nx.enumerate_all_cliques(trust_graph)



    triad_cliques = [triad for triad in list_cliques if len(triad) == 3]

    small_graph = nx.DiGraph()

    for triads in triad_cliques:
        # Find all the triads that eliminates the status theory

        if not trust_graph.has_edge(triads[0], triads[1]):
            print "True"

        if not trust_graph.has_edge(triads[1], triads[2]):
            print "False"

        if not trust_graph.has_edge(triads[2], triads[0]):
            print "False"

        # Change the direction of the edges and add in another graph

        edge01 = trust_graph.get_edge_data(triads[0], triads[1])
        edge12 = trust_graph.get_edge_data(triads[1], triads[2])
        edge20 = trust_graph.get_edge_data(triads[2], triads[0])



        if edge01["weight"] == -1:
            small_graph.add_edge(triads[1], triads[0], weight=1)
        else:
            small_graph.add_edge(triads[0], triads[1], weight=1)

        if edge12["weight"] == -1:
            small_graph.add_edge(triads[2], triads[1], weight=1)
        else:
            small_graph.add_edge(triads[1], triads[2], weight=1)

        if edge20["weight"] == -1:
            small_graph.add_edge(triads[0], triads[2], weight=1)
        else:
            small_graph.add_edge(triads[2], triads[0], weight=1)

        try:
            cycle = nx.find_cycle(small_graph)
        except:
            print triads
            pass

        small_graph.clear()

def generate_rating_data(users,posts,path):
    # Create a fake graph with ratings saying each user rated a particular object with a rating between 1 and 5
    f = open(path + "ratings.txt","w")
    for index in range(0,users,1):
        my_list = [] # To ensure that the users does not rate a post two times

        max_posts = random.randint(8,30)   # Just to simulate that a user might rate n posts atleast 5 posts

        for num_post in range(0,max_posts,1):
            while(True):
                post_id = random.randint(0,posts)  # Randonmly select a post id
                if post_id not in my_list:   # Ensures that same user does not give a rating again
                    break
            if post_id == index:
                post_id = post_id + 1
            my_list.append(post_id)

            rating = random.randint(1,5)       # Give a rating between 1 and 5

            form_string = str(index) + " " + str(post_id) + " " + str(rating) + "\n"
            f.write(form_string)
    f.close()

def generate_user_author_data(users,posts,path):
    # Create a dataset with users owning the posts
    # This should be done in the reverse way
    f = open(path + "author.txt","w")
    for index in range(0,posts,1):
        post_id = index
        user_id = random.randint(0,users)
        form_String =  str(post_id) + " " + str(user_id) + "\n"
        f.write(form_String)      # Keep writing the post and users

    f.close()




if __name__ == '__main__':
    nodes = 100
    edges = 300
    graph = generate_graph(nodes,edges)
    random_dir_path  = "/Users/karthik/PycharmProjects/RecommendorSystem/random_data/"
    f= open(random_dir_path + "trust_data.txt","w")

    nx.write_weighted_edgelist(graph,f)
    #find_triads(graph)
    generate_rating_data(100,800,random_dir_path)
    generate_user_author_data(100,800,random_dir_path)
