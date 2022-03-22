#!/bin/sh

folder_project="/home/christovis/InternetGov/proj1_3gpp_and_comp"
folder_src="${folder_project}/src"

# Step 0.
# Create a very broad  set of queries in unsupervised fashion
#python3 unsupervised_query_expansion.py

# Step 1.
# Using the set of queries create in step 0 crop the search set into a more
# manageable size
query_file="${folder_project}/keywords/bigrams_unsupervised_verified.csv"
python3 find_target_documents.py --query_file $query_file

# Step 2.
# Having a reasonable sized search set (not too computational expensive and 
# memory heavy), there are multiple methods to perform supervised
# 'Concept Expansion'.
#python3 supervised_query_expansion.py
