_Author_ = "Karthik Vaidhyanthan"
_Institute_ = "Gran Sasso Science Institute"

from ConfigParser import SafeConfigParser
from Logging import logger
import json

# To Generate the data of the top n users from the given data set

# As a part of the recommendation system module

CONFIG_FILE = "settings.conf"
CONFIG_SECTION = "generator"  # The section for specifying the configurations of the data generator

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


class DataGenerator():
    data_path = ""
    trust_file = ""
    rating_file = ""
    author_file = ""
    json_path = ""
    num_users = 0
    num_posts = 0
    feature_path = ""
    generated_data_path = ""
    def __init__(self):
        # Initialize the data configurations module
        logger.info("Initializing the orchestrator")
        parser = SafeConfigParser()
        parser.read(CONFIG_FILE)
        self.data_path = parser.get(CONFIG_SECTION, "data_path")
        self.trust_file = parser.get(CONFIG_SECTION, "trust_file")  # This will form the signed network
        self.rating_file = parser.get(CONFIG_SECTION, "rating_file")  # For the user opinion matrix O
        self.author_file = parser.get(CONFIG_SECTION, "author_file")  # For the author content matrix A
        self.json_path = parser.get(CONFIG_SECTION, "json_path")  # For the author content matrix A
        self.num_users = int(parser.get(CONFIG_SECTION, "users"))
        self.num_posts = int(parser.get(CONFIG_SECTION, "posts"))
        self.feature_path = parser.get(CONFIG_SECTION, "feature_path")
        self.model_path = parser.get(CONFIG_SECTION, "model_path")
        self.generated_data_path = parser.get(CONFIG_SECTION,"generated_data_path")

    def label_generator(self):
        # Generate the labels for the users and the contents as the numbering might not be in an increasing order
        # store the label as a json file and return the name of json file in output
        file_path = self.data_path + self.author_file
        f = open(file_path, "r")
        user_label = {}
        post_label = {}
        user_key = 0
        post_key = 0
        users =set()
        posts = set()
        count = 0
        start_count  = 0

        for lines in f.readlines():
            if start_count < 10000: # To skip the first 10000 users
                start_count += 1
                continue
            component = lines.split("|")
            user = int(component[1])
            post = int(component[0])

            if user not in user_label:
                user_label[user] = user_key
                user_key = user_key + 1

            if post not in post_label:
                post_label[post] = post_key
                post_key = post_key + 1



            if user_key >= self.num_users:
                break

            if start_count >= 10000:
                count += 1


        # Store both the labels generated in a json file
        user_json_file = open(self.json_path + "user.json", "w")
        # file_dir = self.data_path + "encoding.json"
        json.dump(user_label, user_json_file)

        post_json_file = open(self.json_path + "post.json", "w")
        json.dump(post_label,post_json_file)

        print len(post_label.keys())
        print len(user_label.keys())


    def load_data(self):
        # Load the json data into the memory and return post and user json
        # loads both the label into the memory
        user_file = open(self.json_path + "user.json", "r")
        post_file = open(self.json_path + "post.json", "r")

        user_json = json.load(user_file, object_hook=_decode_dict)
        post_json = json.load(post_file, object_hook=_decode_dict)

        return user_json,post_json


    def generate_selected_data(self):
        save_file_path = self.generated_data_path + "author.txt"
        save_file = open(save_file_path, "w")

        user_file = open(self.json_path + "user.json", "r")
        post_file = open(self.json_path + "post.json", "r")

        user_json = json.load(user_file, object_hook=_decode_dict)
        post_json = json.load(post_file, object_hook=_decode_dict)

        user_file.close()
        post_file.close()
        file_path = self.data_path + self.author_file
        f = open(file_path, "r")


        for lines in f.readlines():
            component = lines.split("|")
            if str(component[1]) in user_json and str(component[0]) in post_json:
                user = user_json[str(component[1])]
                post = post_json[str(component[0])]
                print user,post
                form_String = str(post) + " " + str(user) + "\n"
                save_file.write(form_String)  # Keep writing the post and users

        save_file.close()


    def generate_selected_rating_data(self):
        # To generate all the rating data for the users and posts key that has been created
        file_path = self.data_path + self.rating_file
        f = open(file_path, "r")
        user_json, post_json = self.load_data()
        save_file_path = self.generated_data_path + "ratings.txt"
        save_file = open(save_file_path, "w")
        rating_above = 0
        rating_below = 0
        for lines in f.readlines():
            component = lines.split("\t")
            # print component

            if component[1] in user_json and component[0] in post_json:
                post = post_json[component[0]]
                user = user_json[component[1]]
                rating = int(component[2])
                if rating > 3:
                    rating_above += 1
                else:
                    rating_below +=1
                form_string = str(user) + " " + str(post) + " " + str(rating) + "\n"
                save_file.write(form_string)

        print rating_above
        print rating_below
        save_file.close()


    def generate_selected_trust_data(self):
        # Generate the test data for the number of users required
        full_file_path = self.data_path + self.trust_file
        f = open(full_file_path, "r")
        user_json, post_json = self.load_data()

        save_file_path = self.generated_data_path + "trust_data.txt"
        save_file = open(save_file_path, "w")

        for line in f.readlines():
            component = line.strip("\n").split("\t")
            user1 = component[0]
            user2 = component[1]
            if user1 in user_json and user2 in user_json:
                node1 = user_json[user1]
                node2 = user_json[user2]
                weight = component[2]

                form_string = str(node1) + " " + str(node2) + " " + str(weight) + "\n"
                save_file.write(form_string)

        save_file.close()



    def find_users_max_posts(self,min_count,max_count):
        # To find the top n users based on the number of content they have written
        # min_count is for telling the range between which the generation has to be done
        file_path = self.data_path + self.author_file
        f = open(file_path, "r")
        user_post_count = {}
        user_list = []
        for lines in f.readlines():
            component = lines.split("|")
            user = int(component[1])
            post = int(component[0])

            if user in user_post_count:
                user_post_count[user] += 1
            else:
                user_post_count[user] = 1

        limit_count = 0
        count = 0
        for key, value in sorted(user_post_count.iteritems(), key=lambda (k, v): (v, k), reverse=True):
            #print "%s: %s" % (key, value)
            if count < min_count:
                count += 1
                continue
            limit_count += 1
            #print limit_count
            user_list.append(key)
            if limit_count>=max_count:
                break

        return user_list

    def top_users_label_generator(self,user_list):
        # Generate the users for the top 100 users
        file_path = self.data_path + self.author_file
        print file_path
        f = open(file_path, "r")
        user_label = {}
        post_label = {}
        user_key = 0
        post_key = 0
        count = 0
        print user_list
        for lines in f.readlines():
            component = lines.split("|")
            user = int(component[1])
            post = int(component[0])
            if user in user_list:

                if user not in user_label:
                    user_label[user] = user_key
                    user_key = user_key + 1

                if post not in post_label:
                    post_label[post] = post_key
                    post_key = post_key + 1

        print len(user_label.keys())
        print len(post_label.keys())


        # Store both the labels generated in a json file
        user_json_file = open(self.json_path + "user.json", "w")
        # file_dir = self.data_path + "encoding.json"
        json.dump(user_label, user_json_file)

        post_json_file = open(self.json_path + "post.json", "w")
        json.dump(post_label, post_json_file)


    def generate_users_basedon_ratings(self,num_users,count_pos=50,count_neg=50):
        # Open the ratings txt file and try to generate a data set with equal number of 1 and -1
        file_path = self.data_path + self.rating_file
        print file_path
        f = open(file_path, "r")
        count_1 = 0
        user_list = []
        count_neg1 = 0
        user_count = 0
        save_file_path = self.generated_data_path + "ratings.txt"
        save_file = open(save_file_path, "w")
        for lines in f.readlines():
            #print lines
            component = lines.strip("\n").split("\t")
            #print component
            post = int(component[0])
            user = int(component[1])
            rating = int(component[2])
            if rating < 4 and count_neg1 < count_neg:
                if user not in user_list:
                    form_string = str(user) + " " + str(post) + " " + str(rating) + "\n"
                    save_file.write(form_string)
                    count_neg1 += 1
                    user_count += 1
                    user_list.append(user)
                    #print user

            if rating > 3 and count_1 < count_pos:
                if user not in user_list:
                    form_string = str(user) + " " + str(post) + " " + str(rating) + "\n"
                    save_file.write(form_string)
                    count_1+= 1
                    user_count += 1
                    user_list.append(user)
                    #print user

            if user_count >= num_users:
                break



        save_file.close()
        return user_list

if __name__ == '__main__':
    data_generator_object = DataGenerator()
    #user_list = data_generator_object.generate_users_basedon_ratings(100,50,50)
    #data_generator_object.top_users_label_generator(user_list)
    #data_generator_object.generate_selected_data()
    #data_generator_object.generate_selected_rating_data()
    #data_generator_object.generate_selected_trust_data()
    




