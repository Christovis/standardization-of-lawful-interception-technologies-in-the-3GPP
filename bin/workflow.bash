#!/bin/sh

#TODO: this is doublicating the CONFIG settings
folder_project="/Users/christovis/Documents/InternetGovernance/proj1_3gpp_and_comp/standardization-of-lawful-interception-technologies-in-the-3GPP/"
folder_src="${folder_project}src/tgpp/"
folder_bin="${folder_project}bin/"

# Step 0.
#python3 prepare_source.py

# Step 1.
# Create a very broad set of queries in unsupervised fashion
#python3 ${folder_bin}unsupervised_query_expansion.py

# Step 1.
# Using the set of queries create in step 0 crop the search set into a more
# manageable size: 3GPP_TSG_SA_WG3, 3GPP_TSG_SA_WG3_LI
search_set="3GPP_TSG_SA_WG3_LI"
python3 ${folder_bin}find_target_documents.py --search_set $search_set

# Step 2.
# Having a reasonable sized search set (not too computational expensive and
# memory heavy), there are multiple methods to perform supervised
# 'Concept Expansion'.
#python3 supervised_query_expansion.py
