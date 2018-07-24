_Author_ = "Karthik Vaidhyanathan"
_Institute_ = "Gran Sasso Science Institute"

# Wrapper Script to call the recommendor object
# The logging for induvidual Function is performed in the Negative_Predictor_class

from  Negative_Predictor_Class import Negative_Link_Predictor

def label_generator(negative_link_predictor_object):
    # The Label Generation module
    negative_link_predictor_object.generate_user_author_matrix()
    negative_link_predictor_object.user_opinion_matrix()
    negative_link_predictor_object.generate_Negative_Interaction_matrix()
    NS = negative_link_predictor_object.generate_Negative_Sample_Set()
    negative_link_predictor_object.generate_graph()
    triad_list, non_status_list = negative_link_predictor_object.generate_triads()
    updated_NS = negative_link_predictor_object.remove_set_elements_NS(non_status_list, triad_list)
    negative_link_predictor_object.generate_user_pairs()

    negative_link_predictor_object.generate_reliablity_matrix(updated_NS)
    triad_list.extend(non_status_list)
    return triad_list,updated_NS

def feature_generator(negative_link_predictor_object, triad_list):
    # Featiure Genarator Module
    negative_link_predictor_object.user_feature_generator(triad_list)
    negative_link_predictor_object.user_pair_feature_generator()
    negative_link_predictor_object.signed_feature_generator(triad_list)
    return True

def predictor(negative_link_predictor_object,updated_NS):
    # The predictor module
    features, labels, label_list = negative_link_predictor_object.feature_combiner(updated_NS)
    negative_link_predictor_object.link_predictor(features, labels)
    return True


def recommender(negative_link_predictor_object):
    # Generate recommendation for each user
    post_rating_dict = negative_link_predictor_object.ratings_mean_generator()
    negative_link_predictor_object.recommendor(post_rating_dict)
    return True

def validation(negative_link_predictor_object):
    # Calculate the accuracy and MAP scores of the recommendation produced
    negative_link_predictor_object.accuracy_check()
    negative_link_predictor_object.find_MAP()

    return True


if __name__ == '__main__':
    negative_link_predictor_object =  Negative_Link_Predictor()
    triad_list, updated_NS = label_generator(negative_link_predictor_object)   # Lable Generation
    print "Label Generation Completed"
    feature_status = feature_generator(negative_link_predictor_object,triad_list) # Feature Generation
    if feature_status ==  True:
        print "Feature Geneartion Completed"
    prediction_status = predictor(negative_link_predictor_object,updated_NS)     # Prediction
    if prediction_status == True:
        print "Prediction Completed"
    recommendation_status  = recommender(negative_link_predictor_object)
    if recommendation_status == True:
        print "Recommendations Generated"
    validation_status = validation(negative_link_predictor_object)
    if validation_status == True:
        print "Validation Completed"

