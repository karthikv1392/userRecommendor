# Recommeder system for Positive Signed network using Negative Link Prediction
This work was done as a part of the coursework on Large Graph Mining and Algorithm Engineering at the Gran Sasso Science Institute ([GSSI](gssi.it))

## Required Packages ##
The tool is built using Python 2.7 and it depends on the following packages for it to run :
1) [NetworkX](http://networkx.github.io) 
2) scipy
3) numpy
4) pickle

## Instructions to use the tool ##

1) Once you have the above installed, clone this repository and make sure that folder hierarchy is maintained
2) The data folder contains the complete [epinions dataset](http://www.trustlet.org/extended_epinions.html)
3) The file Data_Generator.py contains the script to generate subset data from the main data set. The generated data are 
   then saved in the folder generated_data. Inorder to generate new data set :
   1) Go to settings.conf (This file contains the configurations to run the whole tool)
   2) Uncomment the first key "data_path" and comment the second data_path 
   3) Set the key "author_file" to mc.txt 
   4) Go to the Python file Data_Generator.py, uncomment all the lines under the main function
   5) Set the parameters for the function "generate_users_basedon_ratings" as the number of users, number of posetive ratings
      number of negative ratings to be considered for generation and run the file "Data_Generator.py"
4) The main file for running the recommendations is "Recommender.py". If you just want to see the recommendations for all the users
   in the dataset, run this file.
5) All the recommendations will be generated in the folder "generated_features" in the file "recommendation.json".
6) As of now the tool works based on json file based data store. It can be further extended to support database based data store.
7) The MAP score will be printed on the console as well as all the acccuracy measures and the MAP score will be generated in the log file under the logs folder
8) All the settings of the tool can be configured in settings.conf file
9) The folder "generated_features" contains all the features generated during the run of the tool, "genarated_model" stores the SVM model generated,
   "genarated_json" basically stores the label of the nodes and the post and other matrices generated during the tool run.
10) The work was done based on the paper "Negative Link Prediction in Social Media" by Jiliang Tang, Shiyu Chang, Charu Aggarwal, and Huan Liu. 2015, (In Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM '15))

For any queries contact [karthikv1392@gmail.com](karthikv1392@gmail.com)
