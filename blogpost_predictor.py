# -*- coding: utf-8 -*-
"""
Blogpost Predictor

This script reads in blogpost data and do various stuff to it to give guidelines
to content writing and to form a model to predict popular posts so that 
we can expose ourselves more in those posts

Created on Fri Aug 25 22:34:58 2017
Final Version: Sun Sept 3 11:08:35 2017

Note: May require up to 30 minutes to run

@author: William Lee
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingRegressor

"""-------------------------------------------------------------
Function: read_train_data()

Read training data from the training data file
Create column name lists and load file into seperate dataframes
for easier processing and preliminary analysis
Note that column specification of the training file was found on 
website where data was downloaded
Return 11 dataframes: df_stats: Site Stats Data
                      df_total_comment: Total comment over 72hr
                      df_prev_comments: Comment history data
                      df_prev_trackback: Link history data
                      df_time: Basetime - publish time
                      df_length: Length of Blogpost
                      df_bow: Bag of word data
                      df_weekday_B: Basetime day of week
                      df_weekday_P: Publish day of week
                      df_parents: Parent data
                      df_target: comments next 24hrs
Return list of column names as name_list_final
-----------------------------------------------------------------"""  
def read_train_data():
    
	# Create Column name list according to data descriptions
    name_list1 = []
    for i in range(50,60,1):
        name_list1.append('mean_'+ str(i))
        name_list1.append('stdev_'+ str(i))
        name_list1.append('min_'+ str(i))
        name_list1.append('max_'+ str(i))
        name_list1.append('median_'+ str(i))
    
    df_stats = pd.read_csv('blogData_train.csv', header = None, names = name_list1, usecols = list(range(50)))    
    
    name_list2 = ['total_c']
    df_total_comment = pd.read_csv('blogData_train.csv', header = None, names = name_list2, usecols = [50]) 

    name_list3 = ['prev_24hr', 'prev_24hr_to_48hr', 'first_24hr', 'diff_c']
    df_prev_comments = pd.read_csv('blogData_train.csv', header = None, names = name_list3, usecols = [51, 52, 53, 54])
     
    name_list4 = ['total_l', 'prev_24hr_l', 'prev_24hr_to_48hr_l', 'first_24hr_l', 'diff_l']
    df_prev_trackback = pd.read_csv('blogData_train.csv', header = None, names = name_list4, usecols = [55, 56, 57, 58, 59])
    
    name_list5 = ['time_diff']
    df_time = pd.read_csv('blogData_train.csv', header = None, names = ['time_diff'], usecols = [60])
    
    name_list6 = ['length']
    df_length = pd.read_csv('blogData_train.csv', header = None, names = ['length'], usecols = [61])
    
    # Create Columns names for the 200 bag of word features
    name_list7 = []
    for i in range(200):
        name_list7.append('Keyword' + str(i))

    df_bow = pd.read_csv('blogData_train.csv', header = None, names = name_list7, usecols = list(range(62,262,1)))
    
	name_list8 = ['Monday_B', 'Tuesday_B', 'Wednesday_B', 'Thursday_B', 'Friday_B', 'Saturaday_B', 'Sunday_B']
    df_weekday_B = pd.read_csv('blogData_train.csv', header = None, names = name_list8, usecols = list(range(262,269,1))) 
    
    name_list9 = ['Monday_P', 'Tuesday_P', 'Wednesday_P', 'Thursday_P', 'Friday_P', 'Saturaday_P', 'Sunday_P']
    df_weekday_P = pd.read_csv('blogData_train.csv', header = None, names = name_list9, usecols = list(range(269,276,1)))
    
    name_list10 = ['parents', 'parent_min', 'parent_max', 'parent_avg']
    df_parents = pd.read_csv('blogData_train.csv', header = None, names = name_list10, usecols = [276, 277, 278, 279])
    
    name_list11 = ['target_reg']
    df_target = pd.read_csv('blogData_train.csv', header = None, names = name_list11, usecols = [280])

    #Combine all name_list into 1
    name_list_final = name_list1 + name_list2 + name_list3 + name_list4 + name_list5 \
                + name_list6 + name_list7 + name_list8 + name_list9 + name_list10 + name_list11  
    
    return df_stats, df_total_comment, df_prev_comments, df_prev_trackback, df_time, df_length, df_bow, df_weekday_B,\
        df_weekday_P, df_parents, df_target, name_list_final
        
"""-------------------------------------------------------------
Function: process_days(df_test)

Convert the one-hot encoded day of week columns in df_test into a 
two single columns (one for publishing day, one for basetime day)
that has day of weeks labels instead

return processed dataframe 
-----------------------------------------------------------------"""
def process_days(df_test):
    
    name_list8 = ['Monday_B', 'Tuesday_B', 'Wednesday_B', 'Thursday_B', 'Friday_B', 'Saturaday_B', 'Sunday_B']
    name_list9 = ['Monday_P', 'Tuesday_P', 'Wednesday_P', 'Thursday_P', 'Friday_P', 'Saturaday_P', 'Sunday_P']
    df_test['Day_B'] = 'Monday'
    df_test.loc[df_test.Tuesday_B == 1, 'Day_B'] = 'Tuesday'
    df_test.loc[df_test.Wednesday_B == 1, 'Day_B'] = 'Wednesday'
    df_test.loc[df_test.Thursday_B == 1, 'Day_B'] = 'Thursday'
    df_test.loc[df_test.Friday_B == 1, 'Day_B'] = 'Friday'
    df_test.loc[df_test.Saturaday_B == 1, 'Day_B'] = 'Saturday'
    df_test.loc[df_test.Sunday_B == 1, 'Day_B'] = 'Sunday'    
    
    df_test['Day_P'] = 'Monday'
    df_test.loc[df_test.Tuesday_P == 1, 'Day_P'] = 'Tuesday'
    df_test.loc[df_test.Wednesday_P == 1, 'Day_P'] = 'Wednesday'
    df_test.loc[df_test.Thursday_P == 1, 'Day_P'] = 'Thursday'
    df_test.loc[df_test.Friday_P == 1, 'Day_P'] = 'Friday'
    df_test.loc[df_test.Saturaday_P == 1, 'Day_P'] = 'Saturday'
    df_test.loc[df_test.Sunday_P == 1, 'Day_P'] = 'Sunday'    
    
    df_test.drop(name_list8, axis = 1, inplace = True)
    df_test.drop(name_list9, axis = 1, inplace = True)
    
    return df_test


"""-------------------------------------------------------------
Function: load_test_data(name_list_final)

Load test data with columns specified in name_list_final. 
Data from 01 Feb 2012 to 15 Feb 2012 is used as testing data
Load each file and merge into one single dataframe, then seperate
out the site stats and parent data.
Return three dataframes: df_stats - site stats data
                         df_test - bulk of the test data
                         df_parents - blog parent data
-----------------------------------------------------------------"""
def load_test_data(name_list_final):
    test_filelist = []
    #find all files that belongs to the test set, as specified by their date
	for filename in os.listdir('.'):
        if 'test' in filename:
            if ((filename.split('.')[1] == '02') and (int(filename.split('.')[2]) < 16)):
                test_filelist.append(filename)
    
    df_test = pd.DataFrame(columns = name_list_final)
    
    #concat all test files into a single dataframe	
    for testfile in test_filelist:
        df_temp = pd.read_csv(testfile, header = None, names = name_list_final)
        df_test = pd.concat([df_test, df_temp], ignore_index = True)

    #Process Day of week data into a single column
    
    df_test = process_days(df_test)
    
    # Reconstruct namelist so to remove appropiate columns
    name_list10 = ['parents', 'parent_min', 'parent_max', 'parent_avg']
    name_list1 = []
    for i in range(50,60,1):
        name_list1.append('mean_'+ str(i))
        name_list1.append('stdev_'+ str(i))
        name_list1.append('min_'+ str(i))
        name_list1.append('max_'+ str(i))
        name_list1.append('median_'+ str(i))
    
    #Add column for identifying successful posts    
    df_test['target_clf'] = df_test.target_reg > 50
    
	#Save the stats and the blog parent data into seperate dataframes 
    df_stats_test = df_test[name_list1]
    df_test.drop(name_list1, axis = 1, inplace = True)
    df_parents_test = df_test[name_list10]
    df_test.drop(name_list10, axis = 1, inplace = True)
    
    return df_stats_test, df_test, df_parents_test


"""-------------------------------------------------------------
Function: load_daily_validation(filename, name_list_final)

Load data in a single file with filename with columns specified in 
name_list_fianl. Used in evaluating daily operation. Site Stats
and Parents data are dropped as they are not used in model 

Return one dataframes: df_test - bulk of the test data
-----------------------------------------------------------------"""
def load_daily_validation(filename, name_list_final):
    
    df_test = pd.read_csv(filename, header = None, names = name_list_final)
    
    #Process Day of week data into a single column
    df_test = process_days(df_test)
    # Reconstruct namelist so to remove appropiate columns    
    name_list10 = ['parents', 'parent_min', 'parent_max', 'parent_avg']
    name_list1 = []
    for i in range(50,60,1):
        name_list1.append('mean_'+ str(i))
        name_list1.append('stdev_'+ str(i))
        name_list1.append('min_'+ str(i))
        name_list1.append('max_'+ str(i))
        name_list1.append('median_'+ str(i))
    #Add column for identifying successful posts   
    df_test['target_clf'] = df_test.target_reg > 50

	#remove stats data and parent data
    df_test.drop(name_list1, axis = 1, inplace = True)
    df_test.drop(name_list10, axis = 1, inplace = True)
    
    return df_test


"""-------------------------------------------------------------
Function: load_validation_all(name_list_final)

Load validation data with columns specified in name_list_fianl. 
Data from 01 Mar 2012 to 30 Mar 2012 is used as validation data
Load each file and merge into one single dataframe, then seperate
out the site stats and parent data which are dropped
Return one dataframes: df_test - bulk of the test data
-----------------------------------------------------------------"""    
def load_validation_all(name_list_final):
    
	#extract filenames of all files with test in the file name and dates specifed
    validation_filelist = []
    for filename in os.listdir('.'):
        if 'test' in filename:
            if (filename.split('.')[1] == '03'):
                validation_filelist.append(filename)
            elif ((filename.split('.')[1] == '02') and (int(filename.split('.')[2]) >= 16)):
                validation_filelist.append(filename)
    
    df_test = pd.DataFrame(columns = name_list_final)
  
    #combine all files into a single dataframe
    for testfile in validation_filelist:
        df_temp = pd.read_csv(testfile, header = None, names = name_list_final)
        df_test = pd.concat([df_test, df_temp], ignore_index = True)

    #Process day of week data
    
    df_test = process_days(df_test)
    
    name_list10 = ['parents', 'parent_min', 'parent_max', 'parent_avg']
    name_list1 = []
    for i in range(50,60,1):
        name_list1.append('mean_'+ str(i))
        name_list1.append('stdev_'+ str(i))
        name_list1.append('min_'+ str(i))
        name_list1.append('max_'+ str(i))
        name_list1.append('median_'+ str(i))
    
    df_test['target_clf'] = df_test.target_reg > 50
    
	#remove stats and parent data
    df_test.drop(name_list1, axis = 1, inplace = True)
    df_test.drop(name_list10, axis = 1, inplace = True)
    
    return df_test


    
"""-------------------------------------------------------------
Function: resample_data(df_data, factor)

Resample data in df_data to combat class imbalance
Assumes that Positive (True) class is the minority class
Use Bootstrap sampling of the minority class data to select samples
to add into the data. Then remove the same number of majority class
data randomly selected from the sample set. The amount to add/subtract 
is calculated by factor * number of minority class samples. 
Return two dataframes: df_data_sampled - resampled data
                       Y - target for prediction
-----------------------------------------------------------------"""    
def resample_data(df_data, factor):
    
    #Bootstrap sampling of minority class data to add into dataset
    true_indices = df_data[df_data['target_clf'] == True].index
    replicate_indices = np.random.choice(true_indices, len(true_indices)*factor, replace = True)
    df_data_sampled = df_data.loc[replicate_indices]
    
    #Random Removal of the same number of majority class sample 
    false_indices = df_data[df_data['target_clf'] == False].index
    drop_indices = np.random.choice(false_indices, len(true_indices)*factor, replace = False)
    df_data_sampled = pd.concat([df_data_sampled, df_data.drop(drop_indices)], ignore_index = True) 
    
    # Separate between data X and target Y    
    Y = df_data_sampled.target_clf
    df_data_sampled.drop(['target_reg', 'target_clf'], axis = 1, inplace = True)
    
    return df_data_sampled, Y

"""-------------------------------------------------------------
Function: fit_and_predict(X, Y, X_test, Y_test, method)

Fit training dataset X, Y using model specified by method, and then
evaluating the fitted model using X_test and Y_test
Three different models have been tested:
method = "GBC" : Standard Gradient Boosting Classifier with 
                 Learning rate = 0.01 and 500 estimators
method = "RF"  : Random Forest Classifier with 1000 estimators
method = "knn" : K nearest neighbour classifier with k = 50

After fitting, the test data X_test and Y_test is used predicted
and then the model evaluated with the confusion matrix and the auc
score. 
Return: clf - handle for fitted model
        test_proba - class probability of the test samples
        y_pred - class prediction of the test samples
        cm - confusion matrix of the model for the test samples
        auc - auc score of the model for the test samples
-----------------------------------------------------------------"""  
def fit_and_predict(X, Y, X_test, Y_test, method):
    if method == "GBC":
        clf = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 500)
    elif method == "RF":
        clf = RandomForestClassifier(n_estimators = 1000)
    elif method == "knn":
        clf = KNeighborsClassifier(n_neighbors = 50)
    
    clf.fit(X, Y)
    test_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    auc = roc_auc_score(Y_test, test_proba[:,1])
    
    return clf, test_proba, y_pred, cm, auc

"""-------------------------------------------------------------
Function: compare_comment_count(target, predictions, pred_prob, entries)

sum of the comments (specify by "target" column) obtained in selecting 
"entries" number of posts, using four different ways:
    1. Fully Random Selection
    2. Randomly select from posts that were predicted as positive as
       specified in "predictions"
    3. Rank the posts according to the positive class probability specified 
       in "pred_prob" and selecting the top posts
    4. Selecting the posts with top comments (pretending we know what these
       are before hand)
Return: total_count - comments obtained from randomly selected positive classes 
        max_count - comments obtained from selecting known top post
        random_count - comments obtained from randomly selection 
        rank_count - commented obtained from probability ranking
 -----------------------------------------------------------------"""  
def compare_comment_count(target, predictions, pred_prob, entries = 50):
    df = pd.DataFrame(columns = ['comments', 'important_pred', 'pred_prob'])
    df.comments = target
    df.important_pred = predictions
    df.pred_prob = pred_prob
    df_predicted = df[df['important_pred'] == True]
    pred_indices = df_predicted.index
    
    chosen_posts = np.random.choice(pred_indices, entries, replace = False)
    total_count = df.loc[chosen_posts, 'comments'].sum()
    max_count = df.sort_values('comments', ascending = False).iloc[0:entries].comments.sum()
    rank_count = df_predicted.sort_values('pred_prob', ascending = False).iloc[0:entries].comments.sum()
    random_count = df.loc[np.random.choice(df.index, entries, replace = False), 'comments'].sum()
    
    
    return total_count, max_count, random_count, rank_count


"""-------------------------------------------------------------
Function:
write_test_results(file_name, model_name, results_pack, overwrite = False)

Write results contained in "results_pack" from the model of "model_name" into
file with name "file_name". The results should contain the confusion matrix, 
the auc score, and the various comment count metrics. 

If Overwrite is true, overwrites the file, otherwise append at the end of the 
file.

Return: None
 -----------------------------------------------------------------"""  
def write_test_results(file_name, model_name, results_pack, overwrite = False):
    
	#unpack results
    auc, cm, pred_count, max_count, random_count, rank_count = results_pack
    
	#extract true positive, true negative, false positive and false negative
	tn, fp, fn, tp = cm.ravel() 
    if overwrite == True:
        f = open(file_name, 'w') 
    else:
        f = open(file_name, 'a') 
    f.write("-----------------------------------------------\n")
    f.write("Blogpost Prediction Test Results - Model " + model_name + ':\n')
    f.write("AUC score is " + str(auc) + '\n')
    f.write("True Positives: " + str(tp) + '\n')
    f.write("False Negative: " + str(fn) + '\n')
    f.write("False Positive: " + str(fp) + '\n')
    f.write("True Negative: " + str(tn) + '\n')
    f.write("Comments Captured by randomly selecting 50 from positive predictions: " + str(pred_count) + '\n')
    f.write("Comments Captured by ranking top 50 positive probabilities: " + str(rank_count) + '\n')
    f.write("Comments Captured by randomly selecting 50 posts: " + str(random_count) + '\n')
    f.write("Maximum possible comments captured if top 50 post is known prior:" + str(max_count) + '\n')
    f.write("-----------------------------------------------\n\n")
    f.close()

"""-------------------------------------------------------------
Function:
score_model(df_validation, model, model_name, overwrite = False)

Score models using validation data as one dataframe. 
Score similar way as test data, i.e. taking top 50 posts
three types of model_names:
    model_name = 'GBC Basic' - basic GBC model. 
    model_name = 'GBC + RF + kNN' - combined three models
    model_name = 'GBC Resampled' - multiple GBC models with resampling

the models are passed as a list into "model"
Write result into file. If Overwrite is true, overwrites the file, 
otherwise append at the end of the file.

Return: None
 -----------------------------------------------------------------""" 
def score_model(df_validation, model, model_name, overwrite = False):
    
	#seperate out the true values from the parameters
    Y_validation = df_validation.target_clf
    X_validation = df_validation.drop(['target_reg', 'target_clf'], axis = 1)    
    
	#Predict with model using the parameters, then calculate various scores by comparing prediction and true value
    if model_name == 'GBC Basic':
        y_pred = model.predict(X_validation)
        test_proba = model.predict_proba(X_validation)[:,1]
    
    elif model_name == 'GBC + RF + kNN':    
        test_proba_gbc = model[0].predict_proba(X_validation)
        test_proba_rf = model[1].predict_proba(X_validation)
        test_proba_knn = model[2].predict_proba(X_validation
		#Bagging the three models using an optimised weighting to get the final predicted label 
        y_pred = ((test_proba_gbc[:,1] + test_proba_rf[:,1] + test_proba_knn[:,1]) > 1.8) & ((test_proba_gbc[:,0] + test_proba_rf[:,0] + test_proba_knn[:,0]) < 1.2) 
        #For class probability, a simple average is used
		test_proba = (test_proba_gbc[:,1] + test_proba_rf[:,1] + test_proba_knn[:,1])/3
    
    elif model_name == 'GBC Resampled':    
	    #Calculate class probability for the five models from the different resampling
        test_proba_1 = model[0].predict_proba(X_validation)
        test_proba_2 = model[1].predict_proba(X_validation)
        test_proba_3 = model[2].predict_proba(X_validation)
        test_proba_4 = model[3].predict_proba(X_validation)
        test_proba_5 = model[4].predict_proba(X_validation)
        
		#Bagging of multiple model results 
        test_proba_p = test_proba_1[:,1] + test_proba_2[:,1] + test_proba_3[:,1] + test_proba_4[:,1] + test_proba_5[:,1] 
        test_proba_n = test_proba_1[:,0] + test_proba_2[:,0] + test_proba_3[:,0] + test_proba_4[:,0] + test_proba_5[:,0] 
        y_pred = ((test_proba_p > 2.6) & (test_proba_n < 2.2)) 
        test_proba = test_proba_p/5

    #Evaluate result by comparing with random selection 
    model_count, max_count, random_count, rank_model_count = compare_comment_count(df_validation.target_reg, y_pred, test_proba, entries = 100)
    #Standard prediction metrics for a classification problem: confusion matrix and AUC score
	cm = confusion_matrix(Y_validation, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(Y_validation, test_proba)
    
	#Output results into a text file
    result_pack = (auc, cm,  model_count, max_count, random_count, rank_model_count)
    write_test_results('Evaluation_Results.txt', model_name, result_pack, overwrite)

"""-------------------------------------------------------------
Function:
evaluate_model(df_val, model_list, model_type, precision)

Evalute model of model_type using validation data contained in df_val.
There are three model_types:
    model_type = 1: Standard Gradient boosting classifier (GBC)
    model_type = 2: Ensemble of GBC, Random Forest and kNN model
    model_type = 3: Ensemble of GBC models with different resampling

For each model, the thershold for positive and negative prediction was
tuned according to the test data.

For each model, the models used in input through model_list. 

The model is then evaluated with the four commenting strategy, i.e
    1. Selecting the top post
    2. Selecting the top 5 posts. Note that if the number of positive
       prediction is less than 5, then only the positive posts will be
       used
    3. Selecting all positive posts
    4. Selecting the top portion of the post, specified by the precision

Return: Four tuples encapsulating the results 1, 2, 3 and 4 above. The first three
contains the results as specified in the compare_comment_count()
function above. The fourth tuple, for precision selection, return the
number of posts rather than comments
 -----------------------------------------------------------------"""  
def evaluate_model(df_val, model_list, model_type, precision):
    X_val = df_val.drop(['target_reg', 'target_clf'], axis = 1)     
    # Seperate Model types    
    if model_type == 1:
        # Model type one just a simple model
        y_pred = model_list[0].predict(X_val)
        test_proba = model_list[0].predict_proba(X_val)[:,1]
    else:
        # Model type two and three have adjusted thershold for class prediction
        if (model_type == 2):
            p_thershold = 0.6
            n_thershold = 0.4
        elif (model_type == 3):
            p_thershold = 0.52
            n_thershold = 0.44
        
        # Model type two and three also combine multiple models specified in model_list
        test_proba_p = []
        test_proba_n = []
        for clf in model_list:
            test_proba_p.append(clf.predict_proba(X_val)[:,1])
            test_proba_n.append(clf.predict_proba(X_val)[:,0])
        # Create prediction based on modified thersholds    
        y_pred = (sum(test_proba_p)/len(test_proba_p) > p_thershold) & (sum(test_proba_n)/len(test_proba_n) < n_thershold)
        # Take avereage class probability of the models as final class probability        
        test_proba = sum(test_proba_p)/len(test_proba_p)
    
    # If number of positive class is less than five, then only take the positive classes
    p_count = np.count_nonzero(y_pred)     
    if p_count < 5:
        entries = p_count
    else: 
        entries = 5
    
    # Use the prediction and class probability calculated to evalute comment captured for each case
    top_one_tuple = compare_comment_count(df_val.target_reg, y_pred, test_proba, entries = 1)
    top_five_tuple = compare_comment_count(df_val.target_reg, y_pred, test_proba, entries = entries)
    all_positive_tuple = compare_comment_count(df_val.target_reg, y_pred, test_proba, entries = p_count)
    
    # Slightly complicated case for precision selection
    df_val['test_proba'] = test_proba
    df_val['y_pred'] = y_pred
    pred_tp = df_val.sort_values('test_proba', ascending = False).iloc[0:int(p_count*precision)].target_clf.sum()
    actual_tp = p_count*precision
    random_select = df_val.loc[np.random.choice(df_val.index, int(p_count*precision), replace = False), 'target_clf'].sum()
    random_select_p = df_val.loc[np.random.choice(df_val[df_val.y_pred == True].index, int(p_count*precision), replace = False), 'target_clf'].sum()
    precision_tuple = (random_select_p, actual_tp, random_select, pred_tp)
    
    return np.array(top_one_tuple), np.array(top_five_tuple), np.array(all_positive_tuple), np.array(precision_tuple) 


"""-------------------------------------------------------------
Function:
evaluate_model_all_files(validation_filelist, cm, name_list_final, clf_list, model_type)

Evalute model of model_type by compilating the 30 day period specified
by the validation data. Data from each day is contained in each file 
specified in validation_filelist, with name_list_final used for columns
names. 

There are three model_types:
    model_type = 1: Standard Gradient boosting classifier (GBC)
    model_type = 2: Ensemble of GBC, Random Forest and kNN model
    model_type = 3: Ensemble of GBC models with different resampling

For each model, the models used in input through model_list. Both model_type
and model_list are inputs to evaluate_model() as described above

The confusion matrix is passed in cm and the precision is calculated 
to serve as the input to evaluate_model() as described above.  

Return: top_one: Result tuple for selecting top comment
        top_five: Result tuple for selecting top 5 comments
        all_pos: Result tuple for selecting all positive predictions
        prec_pos: Result tuple for selecting the precision portion

Note that the first three returned are number of comments captured, 
specified by compare_comment_count()

The last one returned is the total number of posts
------------------------------------------------------------------"""  
def evaluate_model_all_files(validation_filelist, cm, name_list_final, clf_list, model_type):

    #calculate precision to be used for evaluating the model
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp+fp)
    
    top_one = np.zeros(4)
    top_five = np.zeros(4)
    all_pos = np.zeros(4)
    prec_pos = np.zeros(4)
   
    #Evalute model for all files, and accumulate results 
    for filename in validation_filelist:
        df_val = load_daily_validation(filename, name_list_final)
        df_val.Day_P = lb.transform(df_val.Day_P)
        df_val.Day_B = lb.transform(df_val.Day_B)
    
        array_one, array_five, array_all, array_prec = evaluate_model(df_val, clf_list, model_type, precision)
       
        
        top_one = top_one + array_one
        top_five = top_five + array_five
        all_pos = all_pos + array_all
        prec_pos = prec_pos + array_prec   
    
    return top_one, top_five, all_pos, prec_pos

"""-------------------------------------------------------------
Function:
write_eval_results(filename, clf_result_list, reg_result_list)

Write evaluation results into file specified by filename
the classification only model results are specified in 
clf_result_list
Result from the regression model (which has a different format)
is specified in reg_result_list 

Return: None
------------------------------------------------------------------"""  
def write_eval_results(filename, clf_result_list, reg_result_list):
    counter = 1   
    random_one = 0
    random_five = 0
    random_all = 0
    random_prec = 0
    f = open(filename, 'w')     
    for result in clf_result_list:
        top_one, top_five, all_pos, prec_pos = result
        f.write('--------------------------------------\n')
        f.write('Evaulation Results - Model ' + str(counter) + ' \n')
        f.write('Top Single Post (Random): ' + str(top_one[2]) + ' \n')
        random_one = random_one + top_one[2]
        f.write('Top Single Post (Random from Positive): ' + str(top_one[0]) + ' \n')
        f.write('Top Single Post (Prob Rank from Positive): ' + str(top_one[3]) + ' \n')
        f.write('Top Single Post (A Priori result): ' + str(top_one[1]) + ' \n')
        f.write('Top Five Post (Random): ' + str(top_five[2]) + ' \n')
        random_five = random_five + top_five[2]
        f.write('Top Five Post (Random from Positive): ' + str(top_five[0]) + ' \n')
        f.write('Top Five Post (Prob Rank from Positive): ' + str(top_five[3]) + ' \n')
        f.write('Top Five Post (A Priori result): ' + str(top_five[1]) + ' \n')
        f.write('All Positive Post (Random): ' + str(all_pos[2]) + ' \n')
        random_all = random_all + all_pos[2]
        f.write('All Positive Post (Random from Positive): ' + str(all_pos[0]) + ' \n')
        f.write('All Positive Post (Prob Rank from Positive): ' + str(all_pos[3]) + ' \n')
        f.write('All Positive Post (A Priori result): ' + str(all_pos[1]) + ' \n')
        f.write('Precision Post (Random): ' + str(prec_pos[2]) + ' \n')
        random_prec = random_prec + prec_pos[2]
        f.write('Precision Post (Random from Positive): ' + str(prec_pos[0]) + ' \n')
        f.write('Precision Post (Prob Rank from Positive): ' + str(prec_pos[3]) + ' \n')
        f.write('Precision Post (A Priori result): ' + str(prec_pos[1]) + ' \n')
        f.write('--------------------------------------\n')
        counter = counter + 1
    
    counter_reg = 1
    for result in reg_result_list:
        f.write('--------------------------------------\n')
        f.write('Evaulation Results - (Regression) Model ' + str(counter_reg) + ' \n')
        f.write('Top Single Post (Rank from Positive): ' + str(result[0]) + ' \n')
        f.write('Top Single Post (A Priori result): ' + str(result[4]) + ' \n')
        f.write('Top Five Post (Rank from Positive): ' + str(result[1]) + ' \n')
        f.write('Top Five Post (A Priori result): ' + str(result[5]) + ' \n')
        f.write('All Positive Post (Rank from Positive): ' + str(result[2]) + ' \n')
        f.write('All Positive Post (A Priori result): ' + str(result[6]) + ' \n')
        f.write('Precision Post (Rank from Positive): ' + str(result[3]) + ' \n')
        f.write('Precision Post (A Priori result): ' + str(result[7]) + ' \n')
        f.write('--------------------------------------\n')
        counter_reg = counter_reg + 1
    
    # Take average of the random selections - this will be quoted as the result
    # for random selection    
    f.write('Average Random Posts: \n')
    f.write('Top One: ' + str(random_one/(counter - 1)) + '\n')
    f.write('Top five: ' + str(random_five/(counter - 1)) + '\n')
    f.write('Top all: ' + str(random_all/(counter - 1)) + '\n')
    f.write('Top prec: ' + str(random_prec/(counter - 1)) + '\n')
    
    f.close()


"""------------------------------------------------------------------------
START OF SCRIPT 
-------------------------------------------------------------------------"""

# Load training data
df_stats, df_total_comment, df_prev_comments, df_prev_trackback, df_time, df_length, df_bow, \
df_weekday_B, df_weekday_P, df_parents, df_target, name_list_final =  read_train_data()

# Create Dataframe for storing the the total number of comment and comments in 24hrs
df_target_2 = df_target.merge(df_total_comment, left_index = True, right_index = True)

"""
Session 1: 
Processing day of week data to find relationship between when the blog is posted and 
the number of comment it attracts. 
Two metrics are used. The total number of comments before basetime and 24hr after base 
time is used to gauge the popularity of the post after it was posted. The number
of comments in the next 24 hours is used to gauge whether a particular day will have more
replies

"""

# Convert the 7 weekday columns into 1 column with 7 categories
df_weekday_P['Day'] = 'Monday'
df_weekday_P.loc[df_weekday_P.Tuesday_P == 1, 'Day'] = 'Tuesday'
df_weekday_P.loc[df_weekday_P.Wednesday_P == 1, 'Day'] = 'Wednesday'
df_weekday_P.loc[df_weekday_P.Thursday_P == 1, 'Day'] = 'Thursday'
df_weekday_P.loc[df_weekday_P.Friday_P == 1, 'Day'] = 'Friday'
df_weekday_P.loc[df_weekday_P.Saturaday_P == 1, 'Day'] = 'Saturday'
df_weekday_P.loc[df_weekday_P.Sunday_P == 1, 'Day'] = 'Sunday'

# Put num of comments into dataframe and classify entries according to the total number
# of comments received before and after basetime
df_weekday_P = df_weekday_P.merge(df_target_2, left_index = True, right_index = True)
df_weekday_P['high_interest'] = df_weekday_P['target_reg'] + df_weekday_P['total_c'] > 100

# Generate statistics per weekday to observe any trends 
df3 = df_weekday_P.groupby('Day').agg({'total_c': ['mean'], 'target_reg':['mean'], 'high_interest':['sum'], 'Day':['count']})
#Normalise number of high interst blogs to the total number of blogs
df3['prob'] = df3.high_interest['sum']*100/df3.Day['count']
# Plot trend using a bar plot 
ax = df3['prob'].plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 16)
ax.set(xlabel="Day on which blogpost was published", ylabel="% of blogposts exceeding 100 comments")
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
fig = ax.get_figure()
fig.savefig("histogram_day_published.png", bbox_inches='tight')
plt.show()
# Calculate the probability of getting a popular post - baseline
original_probability = df3.high_interest['sum'].sum()/df3.Day['count'].sum()
# Calculate the probability of getting a popular post if only publish on weekdays
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
new_probability_d = df3.loc[weekdays].high_interest['sum'].sum()/df3.loc[weekdays].Day['count'].sum()


# Repeat for the weekday of basetime, looking at the number of predicted comments only
# to gauge if a specific day people are more likely to comment
df_weekday_B['Day'] = 'Monday'
df_weekday_B.loc[df_weekday_B.Tuesday_B == 1, 'Day'] = 'Tuesday'
df_weekday_B.loc[df_weekday_B.Wednesday_B == 1, 'Day'] = 'Wednesday'
df_weekday_B.loc[df_weekday_B.Thursday_B == 1, 'Day'] = 'Thursday'
df_weekday_B.loc[df_weekday_B.Friday_B == 1, 'Day'] = 'Friday'
df_weekday_B.loc[df_weekday_B.Saturaday_B == 1, 'Day'] = 'Saturday'
df_weekday_B.loc[df_weekday_B.Sunday_B == 1, 'Day'] = 'Sunday'
df_weekday_B = df_weekday_B.merge(df_target_2, left_index = True, right_index = True)
df_weekday_B['high_comment'] = df_weekday_B['target_reg'] > 100
# Generate statistics per weekday to observe any trends 
df3_B = df_weekday_B.groupby('Day').agg({'total_c': ['mean'], 'target_reg':['mean'], 'high_comment':['sum']})
# Plot trend using a bar plot 
ax = df3_B['total_c'].plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 16, color = 'red')
ax.set(xlabel="Day on which blogpost was published", ylabel="% of blogposts exceeding 100 comments")
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
fig = ax.get_figure()
fig.savefig("histogram_day_published.png", bbox_inches='tight')
plt.show()




"""
Session 2: 
Processing length of blog to find relationship between the size of the blog and 
the number of comment it attracts. 
The total number of comments before basetime and 24hr after base 
time is used to gauge the popularity of the post after it was posted. 

"""

# classify entries according to the total number of comments received before and after basetime
df_length['high_interest'] = (df_target_2['total_c'] + df_target_2['target_reg']) > 100

# bin number blogpost length to allow histogram to be plotted. Numbers chosen based on looking
# at the descriptive statistics of blog length using describe()
bins =  range(0,60000,500)
ind = np.digitize(df_length['length'],bins)
df_length_2 = df_length.groupby(ind).agg({'high_interest':['sum', 'count']})
df_length_2['prob'] = (df_length_2.iloc[:,0]/df_length_2.iloc[:,1])*100

# Rename bins to indicate the actual number of words, and trim to show only posts below 15000 as most
# information is already capture within this range
df_length_2.index = df_length_2.index * 500
df_length_2 = df_length_2[df_length_2.index < 15000]

ax2 = df_length_2['prob'].plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 16, color = 'yellow')
ax2.set(xlabel="Maximum Length of Blogpost", ylabel="% of Blogpost exceeding 100 comments")
ax2.xaxis.label.set_fontsize(16)
ax2.yaxis.label.set_fontsize(16)
fig2 = ax2.get_figure()
fig2.savefig("histogram_length_of_blog.png", bbox_inches='tight')
plt.show()
# Calculate the probability of getting a popular post if only publish on weekdays
# Original probability is computed again just as a check
df_length['optimal'] = ((df_length.length > 4000) & (df_length.length < 6500))
original_probability = df_length.high_interest.sum()/df_length.shape[0]
new_probability_l = df_length[df_length.optimal == True].high_interest.sum()/df_length.optimal.sum()


"""
Session 3: 
Processing the temporal evolution of the comments and links to see if it affects whether 
the blogpost would become "viral"

"""

# classify entries according to the total number of comments received after basetime
df_prev_comments['high_comment'] = df_target_2.target_reg > 100
df_prev_trackback['high_comment'] = df_target_2.target_reg > 100
df_target_2['high_comment'] = df_target_2.target_reg > 100


# First looking at the difference in comment in the last 24 hrs. Again, each is binned
# according to the descriptive statistics to allow histogram to be plotted

df_prev_comments.diff_c.describe()
df_prev_trackback.diff_l.describe()

# Bin and plot: comment differences between yesterday and today
bins = np.arange(-1000, 1000, 50)
ind = np.digitize(df_prev_comments['diff_c'], bins)
df4 = df_prev_comments.groupby(ind).agg({'high_comment':['sum']})
df4.index = -1050 + df4.index*50
ax = df4.high_comment.plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 12)
ax.set(xlabel="Difference in number of comments yesterday and the day before", ylabel="Blogpost exceeding 100 comments in next 24 hrs")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
fig = ax.get_figure()
fig.savefig("histogram_comment_diff.png", bbox_inches='tight')
plt.show()
# Bin and plot: link differences between yesterday and today
bins = np.arange(-20, 20, 1)
ind = np.digitize(df_prev_trackback['diff_l'], bins)
df4 = df_prev_trackback.groupby(ind).agg({'high_comment':['sum']})
df4.index = -21 + df4.index*1
ax = df4.high_comment.plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 12, color = 'magenta')
ax.set(xlabel="Difference in number of links yesterday and the day before", ylabel="Blogpost exceeding 100 comments in next 24 hrs")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
fig = ax.get_figure()
fig.savefig("histogram_link_diff.png", bbox_inches='tight')
plt.show()

# Next look at the effect of comments already made - total and that made within the first
# 24 hours of the publication of the post

# Bin and plot: total comments in last 72 hours
bins = np.arange(0, 2000, 100)
ind = np.digitize(df_target_2['total_c'], bins)
df4 = df_target_2.groupby(ind).agg({'high_comment':['sum']})
df4.index = (df4.index-1)*100
ax = df4.high_comment.plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 12, color = 'green')
ax.set(xlabel="Total number of comment prior to today", ylabel="Blogpost exceeding 100 comments in next 24 hrs")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
fig = ax.get_figure()
fig.savefig("histogram_comment_history.png", bbox_inches='tight')
plt.show()
# Bin and plot: total links in last 72 hours
bins = np.arange(0, 50, 1)
ind = np.digitize(df_prev_trackback['total_l'], bins)
df4 = df_prev_trackback.groupby(ind).agg({'high_comment':['sum']})
df4.index = (df4.index - 1) * 1
ax = df4.high_comment.plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 12, color = 'red')
ax.set(xlabel="Total number of links prior to today", ylabel="Blogpost exceeding 100 comments in next 24 hrs")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
fig = ax.get_figure()
fig.savefig("histogram_link_history.png", bbox_inches='tight')
plt.show()
# Bin and plot: total comments in first 24 hours after posting
bins = np.arange(0, 1000, 50)
ind = np.digitize(df_prev_comments['first_24hr'], bins)
df4 = df_prev_comments.groupby(ind).agg({'high_comment':['sum']})
df4.index = (df4.index-1)*50
ax = df4.high_comment.plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 12, color = 'yellow')
ax.set(xlabel="Total number of comments in the first 24hr", ylabel="Blogpost exceeding 100 comments in next 24 hrs")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
fig = ax.get_figure()
fig.savefig("histogram_comment_first_24h.png", bbox_inches='tight')
plt.show()
# Bin and plot: total links in first 24 hours after posting
bins = np.arange(0, 50, 1)
ind = np.digitize(df_prev_trackback['first_24hr_l'], bins)
df4 = df_prev_trackback.groupby(ind).agg({'high_comment':['sum']})
df4.index = (df4.index-1)
ax = df4.high_comment.plot(kind = 'bar', legend = False, figsize = (8, 5), fontsize = 12)
ax.set(xlabel="Total number of links in the first 24hr", ylabel="Blogpost exceeding 100 comments in next 24 hrs")
ax.xaxis.label.set_fontsize(12)
ax.yaxis.label.set_fontsize(12)
fig = ax.get_figure()
fig.savefig("histogram_link_first_24h.png", bbox_inches='tight')
plt.show()

"""
Session 4: 
Using only the Bag of words features, generate a model which gives a feature importance plot
This can be use to see if there are particular keywords that affects whether a blogpost
becomes popular. This is compared with just looking at the probability of positive post
vs negative post for each keyword
"""
# Create quick model and plot feature importance
df_target_2['high_interest'] = df_target_2['target_reg'] + df_target_2['total_c'] > 100
Y = df_target_2.high_interest
X = df_bow
clf = RandomForestClassifier(n_estimators = 500)
clf.fit(X, Y)
plt.figure(figsize=(8,5))
plt.bar(range(200), clf.feature_importances_)
plt.xlabel('Keywords', size = 14)
plt.ylabel('Importance (arb. units)', size = 14)
plt.show()
# Compare the positive and negative probability for each keyword and plot
df_bow['high_interest'] = df_target_2['high_interest']
df_word_prob = df_bow.groupby('high_interest').agg(['sum'])
df_word_prob = df_word_prob.transpose()
df_word_prob['prob'] = df_word_prob.iloc[:,1]/(df_word_prob.iloc[:,0]+df_word_prob.iloc[:,1])
df_word_prob['word'] = list(range(0,200,1))
plt.figure(figsize=(8,5))
plt.bar(df_word_prob['word'], df_word_prob['prob'])
plt.xlabel('Keywords', size = 14)
plt.ylabel('Probability of high interest posts', size = 14)
plt.savefig("histogram_keywords.png", bbox_inches='tight')
plt.show()
# It turns out that the second method is better, due to the fact that feature
# importance does not specify whether it has positive and negative impact
# So second method is used, the the probability of success when at least
# one of the top twenty keyword is present is calculated
original_probability = df_bow.high_interest.sum()/df_bow.shape[0]
topwords = df_word_prob.sort_values('prob', ascending = False).iloc[0:20].word
df_bow['twenty_top'] = df_bow.iloc[:,topwords].sum(axis = 1) > 0
df_prob2 = df_bow.groupby('twenty_top').agg({'high_interest':'sum'})
new_probability_w = df_prob2.iloc[1].sum()/df_bow['twenty_top'].sum()


# Calculate the probability of post exceeding 100 post if all three filters
# 1. Publish on weekdays, 2. Optimal Blog length and 3. Have top 20 keywords
# is applied.  
df_comb = df_weekday_P[['Day', 'high_interest']]
df_comb['optimal_l'] = df_length.loc[:,'optimal']
df_comb['twenty_top'] = df_bow.loc[:,'twenty_top']
df_comb['All'] = df_comb.loc[:,'optimal_l'] & df_comb.loc[:,'twenty_top'] & df_comb.Day.isin(weekdays)
hits = df_comb[df_comb.All == True].high_interest.sum()
all_posts = df_comb[df_comb.All == True].high_interest.count()
final_probability = hits/all_posts


# Record probabilities in a text file
f = open('Content_improvement_probabilities.txt', 'w')
f.write('Baseline probability of popular post:'+ str(original_probability) +'\n')
f.write('Probability of popular post published on weekdays only:'+ str(new_probability_d)+'\n')
f.write('Probability of popular post with 4500 - 6500 words only:'+ str(new_probability_l)+'\n')
f.write('Probability of popular post with 20 top keywords:'+ str(new_probability_w)+'\n')
f.write('Number of posts with all of the above:'+ str(all_posts)+'\n')
f.write('Number of posts with all of the above that is popular:'+ str(hits)+'\n')
f.write('Probability of popular post if all of above satisfied:'+ str(final_probability)+'\n')
f.close()

# Clean up dataframes in preparation for next sessions
df_bow.drop('twenty_top', axis = 1, inplace = True)
df_length.drop('optimal', axis = 1, inplace = True)
df_bow.drop('high_interest', axis = 1, inplace = True)




"""
Session 5: 
Combine data and start predictive model
Looking at the parent data, it is decided that it is left out in the first go
The df_stats is also left out, as it gives the statistical value of the whole site,
so does not seem to be useful to predict individual posts

"""

#Clean up data
df_prev_comments.drop('high_comment', axis = 1, inplace = True)
df_prev_trackback.drop('high_comment', axis = 1, inplace = True)
df_length.drop('high_interest',axis = 1, inplace = True)

#Putting the training data together
df_data = df_prev_comments.merge(df_prev_trackback, left_index = True, right_index = True)
df_data['total_c'] = df_total_comment.total_c
df_data = df_data.merge(df_length, left_index = True, right_index = True)
df_data['time_diff'] = df_time.time_diff
df_data['Day_P'] = df_weekday_P.Day
df_data['Day_B'] = df_weekday_B.Day
df_data = df_data.merge(df_bow, left_index = True, right_index = True)
df_data['target_reg'] = df_target_2['target_reg']
df_data['target_clf'] = df_data.target_reg > 50

# Load test data. Note the test data used is the first 15 days of Feburary. The rest
# of the data will be used to generate the final statistics for the final models
df_stats_test, df_test, df_parents_test = load_test_data(name_list_final)

# Reorder so that the test data and train data has same column order
cols = df_test.columns.tolist()
df_data = df_data[cols]

# Encode weekdays into categorical labels
lb = LabelEncoder()
df_data.Day_P = lb.fit_transform(df_data.Day_P)
df_data.Day_B = lb.transform(df_data.Day_B)
df_test.Day_P = lb.transform(df_test.Day_P)
df_test.Day_B = lb.transform(df_test.Day_B)

# Seperate X and Y for test data
Y_test = df_test.target_clf
X_test = df_test.drop(['target_reg', 'target_clf'], axis = 1)

# Resample data to combat class imbalance  
X, Y = resample_data(df_data, 15)

# Fit model using a Gradient Boosted Decision Tree Classifier
clf_v, test_proba, y_pred, cm_v, auc = fit_and_predict(X, Y, X_test, Y_test, "GBC")
tn, fp, fn, tp = cm_v.ravel()
# Use predictions to select blogposts and compare the comments captured, and write result
pred_count, max_count, random_count, rank_count = compare_comment_count(df_test.target_reg, y_pred, test_proba[:,1], entries = 50)
result_pack = (auc, cm_v, pred_count, max_count, random_count, rank_count)
write_test_results('Test_Results.txt', 'Basic GBC', result_pack, overwrite = True)

"""
Session 6: 
Improving prediction
Try to improve prediction by more complex models
Strategy #1: Add features
Strategy #2: Voting from different models
Strategy #3: Multiple resampling to get more information in negative class to reduce FP
Strategy #4: Seperate problem into a classification and a regression model

"""
# Strategy 1: Feature engineering

# Add in mean and max of total number of comments and last 24 hrs, which gives indication of popularity of site
# Note that none of these improves the precision or recall
df_data['site_mean'] = df_stats.mean_50
df_data['site_max'] = df_stats.max_50
df_data['site_24hr_mean'] = df_stats.mean_52
df_data['site_24hr_max'] = df_stats.max_52
df_test['site_mean'] = df_stats_test.mean_50
df_test['site_max'] = df_stats_test.max_50
df_test['site_24hr_mean'] = df_stats_test.mean_52
df_test['site_24hr_max'] = df_stats_test.max_52
Y_test = df_test.target_clf
X_test = df_test.drop(['target_reg', 'target_clf'], axis = 1)

# Resample data to combat class imbalance  
X, Y = resample_data(df_data, 15)

# Fit model using a Gradient Boosted Decision Tree Classifier
clf, test_proba, y_pred, cm, auc = fit_and_predict(X, Y, X_test, Y_test, "GBC")
tn, fp, fn, tp = cm.ravel()
# Use predictions to select blogposts and compare the comments captured
pred_count, max_count, random_count, rank_count = compare_comment_count(df_test.target_reg, y_pred, test_proba[:,1], entries = 50)
result_pack = (auc, cm, pred_count, max_count, random_count, rank_count)
write_test_results('Test_Results.txt', 'Basic GBC with added Site Stats', result_pack, overwrite = False)


#Count number of words in bag of words that appeared
df_data['word_count'] = df_data[df_bow.columns].sum(axis = 1)
df_test['word_count'] = df_test[df_bow.columns].sum(axis = 1)

# Reorder columns in df_data to match df_test
cols = df_test.columns.tolist()
df_data = df_data[cols]
# Seperate X and Y of test data
Y_test = df_test.target_clf
X_test = df_test.drop(['target_reg', 'target_clf'], axis = 1)

# Resample data to combat class imbalance  
X, Y = resample_data(df_data, 15)

# Fit model using a Gradient Boosted Decision Tree Classifier
clf, test_proba, y_pred, cm, auc = fit_and_predict(X, Y, X_test, Y_test, "GBC")
tn, fp, fn, tp = cm.ravel()
# Use predictions to select blogposts and compare the comments captured
pred_count, max_count, random_count, rank_count = compare_comment_count(df_test.target_reg, y_pred, test_proba[:,1], entries = 50)
result_pack = (auc, cm, pred_count, max_count, random_count, rank_count)
write_test_results('Test_Results.txt', 'Basic GBC with added Word count', result_pack, overwrite = 'False')


# Strategy 2: Combining Models by voting and changing prediction thershold
# A Random Forest and kNN model will be created in addition to the gradient boosted model
# The results are combined in two ways - by combining the prediction probablity and then changing 
# thresholding to get the class prediction, or by combining the prediction probability and directly
# use that to rank and select the top posts

# First remove extra features as they did not do anything
df_data.drop(['word_count', 'site_mean', 'site_max', 'site_24hr_mean', 'site_24hr_max'], axis = 1, inplace = True)
df_test.drop(['word_count', 'site_mean', 'site_max', 'site_24hr_mean', 'site_24hr_max'], axis = 1, inplace = True)
cols = df_test.columns.tolist()
df_data = df_data[cols]
Y_test = df_test.target_clf
X_test = df_test.drop(['target_reg', 'target_clf'], axis = 1)

# Combining different models and tweaking thershold for prediction
X_train, Y_train = resample_data(df_data, 15)
clf_gbc, test_proba_gbc, y_pred_gbc, cm_gbc, auc_gbc = fit_and_predict(X_train, Y_train, X_test, Y_test, "GBC")
clf_rf, test_proba_rf, y_pred_rf, cm_rf, auc_rf = fit_and_predict(X_train, Y_train, X_test, Y_test, "RF")
clf_knn, test_proba_knn, y_pred_knn, cm_knn, auc_knn = fit_and_predict(X_train, Y_train, X_test, Y_test, "knn")

# Thershold tweaking was performed using test data, optimising the precision and recall tradeoff
y_pred = ((test_proba_gbc[:,1] + test_proba_rf[:,1] + test_proba_knn[:,1]) > 1.8) & ((test_proba_gbc[:,0] + test_proba_rf[:,0] + test_proba_knn[:,0]) < 1.2) 
cm_combined = confusion_matrix(Y_test, y_pred)
y_pred_proba = (test_proba_gbc[:,1] + test_proba_rf[:,1] + test_proba_knn[:,1])/3

# Evaluate and record final model on test data
total_count_combined, max_count, random_count, rank_count_combined = compare_comment_count(df_test.target_reg, y_pred, y_pred_proba)
tn, fp, fn, tp = cm_combined.ravel()
auc_combined = roc_auc_score(Y_test, y_pred_proba)
result_pack = (auc_combined, cm_combined, total_count_combined, max_count, random_count, rank_count_combined)
write_test_results('Test_Results.txt', 'Combined GBC+ RF + kNN', result_pack, overwrite = False)


# Strategy 3: Combining multiple models generated from resampling
# The rationale for this is that in order to combat imbalance classes, we have removed large 
# number of majority class samples. This means that everytime we resample, we will have 
# different information on the majority class, hence giving slightly different predictions
# The idea is that by combining these, we can hopefully predict the majority class even better
# and reduces the number of false positives

# Resample and train multiple GBC models
X_train, Y_train = resample_data(df_data, 10)
clf_1, test_proba_1, y_pred_1, cm_1, auc_1 = fit_and_predict(X_train, Y_train, X_test, Y_test, "GBC")
X_train, Y_train = resample_data(df_data, 15)
clf_2, test_proba_2, y_pred_2, cm_2, auc_2 = fit_and_predict(X_train, Y_train, X_test, Y_test, "GBC")
X_train, Y_train = resample_data(df_data, 20)
clf_3, test_proba_3, y_pred_3, cm_3, auc_3 = fit_and_predict(X_train, Y_train, X_test, Y_test, "GBC")
X_train, Y_train = resample_data(df_data, 18)
clf_4, test_proba_4, y_pred_4, cm_4, auc_4 = fit_and_predict(X_train, Y_train, X_test, Y_test, "GBC")
X_train, Y_train = resample_data(df_data, 12)
clf_5, test_proba_5, y_pred_5, cm_5, auc_5 = fit_and_predict(X_train, Y_train, X_test, Y_test, "GBC")

# Thershold tweaking was performed using test data, optimising the precision and recall tradeoff
test_proba_p = test_proba_1[:,1] + test_proba_2[:,1] + test_proba_3[:,1] + test_proba_4[:,1] + test_proba_5[:,1] 
test_proba_n = test_proba_1[:,0] + test_proba_2[:,0] + test_proba_3[:,0] + test_proba_4[:,0] + test_proba_5[:,0] 
y_pred = ((test_proba_p > 2.6) & (test_proba_n < 2.2)) 
cm_resampled = confusion_matrix(Y_test, y_pred)
y_pred_proba = test_proba_p/5

# Evaluate and record final model on test data
total_count_resample, max_count, random_count, rank_count_resample = compare_comment_count(df_test.target_reg, y_pred, test_proba_3[:,1])
tn, fp, fn, tp = cm_resampled.ravel()
auc_sampled = roc_auc_score(Y_test, y_pred_proba)
result_pack = (auc_sampled, cm_resampled, total_count_resample, max_count, random_count, rank_count_resample)
write_test_results('Test_Results.txt', 'Resampled GBC', result_pack, overwrite = False)

# Strategy 4: Seperating the problem into classification and regression model
# The rationale for this is that in order to combat imbalance classes, we have removed large 
# number of majority class samples. This means that everytime we resample, we will have 
# different information on the majority class, hence giving slightly different predictions
# The idea is that by combining these, we can hopefully predict the majority class even better
# and reduces the number of false positives

# Create class label seperating between commented and uncommented posts
df_data['commented'] = df_data.target_reg > 0
df_test['commented'] = df_test.target_reg > 0

# Create seperate dataframe only with posts that have comments
df_commented = df_data[df_data.target_reg > 0]

# First model: Seperate data between with and without comments

cols = df_test.columns.tolist()
df_data = df_data[cols]
Y_test = df_test.commented
X_test = df_test.drop(['target_reg', 'target_clf', 'commented'], axis = 1)

# Resample data. Since of different class definition, the function above cant be used
true_indices = df_data[df_data['commented'] == True].index
replicate_indices = np.random.choice(true_indices, len(true_indices), replace = True)
df_data_sampled = df_data.loc[replicate_indices]
false_indices = df_data[df_data['commented'] == False].index
drop_indices = np.random.choice(false_indices, len(true_indices), replace = False)
df_data_sampled = pd.concat([df_data_sampled, df_data.drop(drop_indices)], ignore_index = True) 
Y = df_data_sampled.commented
df_data_sampled.drop(['target_reg', 'target_clf', 'commented'], axis = 1, inplace = True)

# Fit classification model with a GBC
clf_z, test_proba_z, y_pred_z, cm_z, auc_z = fit_and_predict(df_data_sampled, Y, X_test, Y_test, "GBC")

#Second model: model data with comments with regression

# Prepare training data. Note the total number of comment is used as the target
X_train = df_commented.drop(['target_reg', 'target_clf', 'commented'], axis = 1)
Y_train = df_commented.target_reg
# Prepare Test data - first incorporating the predicition of classification
# Then seperate out data that is classified as having comments
df_test_commented = df_test.copy()
# Add on classification prediction 
df_test_commented['pred'] = y_pred_z
# Record down actual posts that were positive
actual_positive = df_test.target_clf.sum()
total_posts = df_test.shape[0]
# Take only data that was classifed as having comment to create test data set
df_test_commented = df_test_commented[df_test_commented.pred == True]
X_test = df_test_commented.drop(['target_reg', 'target_clf', 'commented', 'pred'], axis = 1)
# Create and fit regression data using a gradient boosted regressor
reg = GradientBoostingRegressor(learning_rate = 0.01, n_estimators = 500)
reg.fit(X_train, Y_train)
# Predict test data and score. Note that without class probability, there is no AUC score 
y_pred_reg = reg.predict(X_test)
df_test_commented['reg_pred'] = y_pred_reg
rank_count_reg = df_test_commented.sort_values('reg_pred', ascending = False).iloc[0:50].target_reg.sum()
max_count_reg = df_test_commented.sort_values('target_reg', ascending = False).iloc[0:50].target_reg.sum()

# Calculate confusion matrix
total_positive = df_test_commented[df_test_commented.reg_pred > 50].shape[0]
tp = df_test_commented[df_test_commented.reg_pred > 50].target_clf.sum()
fp = total_positive - tp
total_negative = total_posts - total_positive
fn = actual_positive - tp
tn = total_negative - fn
cm_r = np.array([[tn, fp],[fn, tp]])
precision_model4 = tp/total_positive

# Write results - there is no random selections for this model
result_pack = ('N/A', cm_r, 'N/A', max_count_reg , 'N/A', rank_count_reg)
write_test_results('Test_Results.txt', 'Classification + Regression', result_pack, overwrite = False)

"""
Session 6: 
Final Prediction
Choosen three models above, and predict validation samples

"""

# First score models by using all of the validation data at once

#Load data
df_validation = load_validation_all(name_list_final)
df_validation.Day_P = lb.transform(df_validation.Day_P)
df_validation.Day_B = lb.transform(df_validation.Day_B)

#Score models
score_model(df_validation, clf_v, 'GBC Basic', overwrite = True)
clf_list = [clf_gbc, clf_rf, clf_knn]
score_model(df_validation, clf_list, 'GBC + RF + kNN', overwrite = False)
clf_list = [clf_1, clf_2, clf_3, clf_4, clf_5]
score_model(df_validation, clf_list, 'GBC Resampled', overwrite = False)


# Then evaluate by summing up daily comment capture
validation_filelist = []
for filename in os.listdir('.'):
    if 'test' in filename:
        if (filename.split('.')[1] == '03'):
            validation_filelist.append(filename)
        elif ((filename.split('.')[1] == '02') and (int(filename.split('.')[2]) >= 16)):
            validation_filelist.append(filename)   


#model_1: Vanilla Model
model1_results = evaluate_model_all_files(validation_filelist, cm_v, name_list_final, [clf_v], 1)

#model_2: Combined Models
clf_list = [clf_gbc, clf_rf, clf_knn]
model2_results = evaluate_model_all_files(validation_filelist, cm_combined, name_list_final, clf_list, 2)

#model_3: Resampled Models
clf_list = [clf_1, clf_2, clf_3, clf_4, clf_5]
model3_results = evaluate_model_all_files(validation_filelist, cm_resampled, name_list_final, clf_list, 3)

#model 4: Classification + Regression
top_one = 0
max_one = 0
top_five = 0
max_five = 0
all_pos = 0
max_all = 0
prec_pos = 0
max_pos = 0


for filename in validation_filelist:
    df_val = load_daily_validation(filename, name_list_final)
    df_val.Day_P = lb.transform(df_val.Day_P)
    df_val.Day_B = lb.transform(df_val.Day_B)
    df_val['commented'] = df_val.target_reg > 0
    
    Y_val = df_val.commented
    X_val = df_val.drop(['target_reg', 'target_clf', 'commented'], axis = 1)
    
    # Two stage predicition
    y_pred = clf_z.predict(X_val)
    df_val_commented = df_val.copy()
    df_val_commented['pred'] = y_pred
    df_val_commented = df_val_commented[df_val_commented.pred == True] 
    X_val = df_val_commented.drop(['target_reg', 'target_clf', 'commented', 'pred'], axis = 1)
    y_pred_reg = reg.predict(X_val)
    df_val_commented['reg_pred'] = y_pred_reg
    
    # Calculate top selected posts for the four scenarios, and add up over all data 
    positive_predictions = df_val_commented[df_val_commented.reg_pred > 50].shape[0]     
    high_comment_posts = df_val.target_clf.sum()  
    top_one = top_one + df_val_commented.sort_values('reg_pred', ascending = False).iloc[0:1].target_reg.sum()
    value = 5 if positive_predictions > 5 else positive_predictions
    top_five = top_five + df_val_commented.sort_values('reg_pred', ascending = False).iloc[0:value].target_reg.sum()
    all_pos = all_pos + df_val_commented.sort_values('reg_pred', ascending = False).iloc[0:positive_predictions].target_reg.sum()
    prec_pos = prec_pos + df_val_commented.sort_values('reg_pred', ascending = False).iloc[0:int(positive_predictions * precision_model4)].target_clf.sum()
    max_one = max_one + df_val.sort_values('target_reg', ascending = False).iloc[0:1].target_reg.sum()  
    max_five = max_five + df_val.sort_values('target_reg', ascending = False).iloc[0:value].target_reg.sum() 
    max_all = max_all + df_val.sort_values('target_reg', ascending = False).iloc[0:positive_predictions].target_reg.sum() 
    max_pos = max_pos + positive_predictions * precision_model4

# Pack results into list
model4_results = [top_one, top_five, all_pos, prec_pos, max_one, max_five, max_all, max_pos]

# Write all results
write_eval_results('comments_evaluation.txt', [model1_results, model2_results, model3_results], [model4_results])

"""END OF SCRIPT"""


