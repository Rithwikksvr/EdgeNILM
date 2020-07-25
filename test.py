import numpy as np
import pandas as pd
import time
import    math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
from sklearn.metrics import    mean_absolute_error
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import time
import sys

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from functions import *
from networks import Seq2Point

def test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr):

    dir_name = "fold_%s_models"%(fold_number)
    dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
    dir_name = os.path.join(dir_name, model_name)

    models_dir = dir_name

    test_file_name = 'test_%s.h5'%(fold_number)
    all_appliances_mains_lst, all_appliances_truth = load_h5_file(test_file_name, appliances)
    
    # Taking the first 10% or 20%

    for i in range(len(all_appliances_mains_lst)):
        n = len(all_appliances_mains_lst[i])
        for j in range(n):
            df = all_appliances_mains_lst[i][j]
            all_appliances_mains_lst[i][j] = df.iloc[:int(fraction_to_test * len(df))]

            df = all_appliances_truth[i][j]
            all_appliances_truth[i][j] = df.iloc[:int(fraction_to_test * len(df))]


    parameters_path = os.path.join(dir_name, 'parameters.json')
    f = open(parameters_path)
    parameters = json.load(f)

    all_appliances_predictions = []
    total_time = 0
    model_size = 0
    for appliance_index, appliance_name in enumerate(appliances):
        mains_mean = parameters[appliance_name]['mains_mean']
        mains_std = parameters[appliance_name]['mains_std']
        app_mean = parameters[appliance_name]['app_mean']
        app_std = parameters[appliance_name]['app_std']
        appliance_mains_dfs = all_appliances_mains_lst[appliance_index]

        no_of_homes = len(appliance_mains_dfs)
        if 'mtl' not in model_name:
                model_path = os.path.join(dir_name, "%s.pth"%(appliance_name))
        else:
                model_path = os.path.join(dir_name, "weights.pth")
        model_size+= int(os.stat(model_path).st_size)/(1024*1024)
        if not cuda:
            model = torch.load(model_path,map_location=torch.device('cpu'))    
        else:
            model = torch.load(model_path)    
        model.eval()    
        

        appliance_prediction = []

        for home_id in range(no_of_homes):
            home_mains = appliance_mains_dfs[home_id]
            l = len(home_mains)
                        
            processed_mains = mains_preprocessing([home_mains], sequence_length)
            processed_mains = (processed_mains - mains_mean)/mains_std
            a =time.time()
            
            if 'mtl' in model_name:
                prediction = predict_mtl(model, processed_mains, appliance_index, cuda, batch_size)
            else:
                prediction = predict(model, processed_mains, cuda, batch_size)
            
            b=time.time()
            total_time+=b-a
            prediction = prediction * app_std + app_mean
            prediction = prediction.flatten()
            prediction = np.where(prediction>0, prediction,0)
            df = pd.DataFrame({appliance_name: prediction})
            df.index = home_mains.index
            appliance_prediction.append(df)

            # print (home_mains.shape)
            # print ()
        all_appliances_predictions.append(appliance_prediction)


        
        # print ("Finished predicting for appliance %s"%(appliance_name))

    results = []
    results.append(model_name)
    results.append(sequence_length)
    results.append(fold_number)
    results.append(batch_size)
    results.append(model_size)
    total_error = 0
    for app_index, app_name in enumerate(appliances):
        truth_ = pd.concat(all_appliances_truth[app_index],axis=0).values
        pred_ = pd.concat(all_appliances_predictions[app_index],axis=0).values
        error = mean_absolute_error(truth_, pred_)
        results.append(error)
        total_error+=error
        if plot:
            plt.figure(figsize=(300,8))
            plt.plot(truth_,'r',label="Truth")
            plt.plot(pred_,'b',label="Pred")
            plt.legend()
            plt.savefig("images/%s_%s_%s_fold_%s.png"%(model_name, app_name,sequence_length,fold_number))
        print ("%s Error: %s"%(app_name, error))
    print ("Total Error: %s"%(total_error))
    print ("Time taken: ", total_time)
    results.append(total_error)
    results.append(total_time)
    results_arr.append(results)

    return all_appliances_truth, all_appliances_predictions




appliances = ["fridge",'dish washer','washing machine']
appliances.sort()

batch_size=4096
fold_numbers=[1, 2, 3]
sequence_lengths = [499, 99]
fraction_to_test = 1
cuda=True
plot=False
"""Unpruned Model"""

method=sys.argv[1]

create_dir_if_not_exists('results')

for method in ['fully_shared_mtl_pruning','unpruned_model','tensor_decomposition','normal_pruning','iterative_pruning', 'fully_shared_mtl']:

    results_arr = []
    for fold_number in fold_numbers:

        print ("Batch size:", batch_size)
        for sequence_length in sequence_lengths:

            if method=='unpruned_model':
                print ("-"*50)
                print ("Results unpruned model; sequence length: %s "%(sequence_length))
                truth, all_predictions = test_fold('unpruned_model', appliances, fold_number, sequence_length, batch_size, results_arr)
                
                print ("-"*50)
                print ("\n\n\n")
            
            elif method=='normal_pruning':
                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for %s percent Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "pruned_model_%s_percent" %(pruned_percentage)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)
                    
                    print ("-"*50)
                    print ("\n\n\n")


            elif method=='iterative_pruning':        

                for iterative_pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for %s percent Iterative Pruning; sequence length: %s "%(iterative_pruned_percentage, sequence_length))
                    model_name = "iterative_model_%s_percent" %(iterative_pruned_percentage)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)
                    
                    print ("-"*50)
                    print ("\n\n\n")

            elif method=='tensor_decomposition':

                for rank in [1,2, 4,8 ]:
                    print ("-"*50)
                    print ("Results for rank %s tensor decomposition; sequence length: %s "%(rank, sequence_length))
                    model_name = 'tensor_decomposition_rank_%s'%(rank)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                    print ("-"*50)
                    print ("\n\n\n")
            elif method == 'fully_shared_mtl':
                print ("-"*50)
                print ("Results for Fully shared MTL Model; sequence length: %s "%(sequence_length))
                model_name = method
                truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                print ("-"*50)
                print ("\n\n\n")
        
            elif method == 'normal_mtl':
                print ("-"*50)
                print ("Results for Normal  MTL Model; sequence length: %s "%(sequence_length))
                model_name = method
                truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                print ("-"*50)
                print ("\n\n\n")
            
            elif method=='fully_shared_mtl_pruning':

                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for Fully shared MTL %s percent Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "fully_shared_mtl_pruning_%s_percent" %(pruned_percentage)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                    
                    print ("-"*50)
                    print ("\n\n\n")

            
    columns  = ['Model Name',"Sequence Length","Fold Number","Batch Size","Model size"]
    for app_name in appliances:
        columns.append(app_name+" Error")
    columns.append("Total Error")
    columns.append("Total Time taken")

    results_arr= np.array(results_arr)
    df = pd.DataFrame(data=results_arr, columns=columns, index = range(len(results_arr)))
    df.to_csv(os.path.join('results','%s.csv'%(method)),index=False)












# import time

# def compute_statistics(model_path, test_x, test_y, app_mean, app_std, batch_size, cuda):
#     model = torch.load(model_path,map_location=torch.device('cpu'))
#     a = time.time()
#     test_pred = predict(model,test_x,cuda,batch_size) 
#     b = time.time()

#     time_taken = b-a

#     test_pred = app_mean + test_pred * app_std
#     test_truth = app_mean + test_y * app_std


#     error = mean_absolute_error(test_truth, test_pred)
#     model_size = os.stat(model_path).st_size/(1024*1024)

#     return time_taken, error, model_size, test_pred

# compute_statistics('unpruned_weights.pth',test_x, test_y, app_mean, app_std, batch_size, cuda)

# time_taken, error, model_size, test_pred = compute_statistics('pruned_weights_10.pth',test_x, test_y, app_mean, app_std, batch_size, cuda)
# print (time_taken, error, model_size)

# time_taken, error, model_size, test_pred = compute_statistics('tensor_decomposition_weights_rank_1',test_x, test_y, app_mean, app_std, batch_size, cuda)
# print (time_taken, error, model_size)

# time_taken, error, model_size, test_pred = compute_statistics('tensor_decomposition_weights_rank_2',test_x, test_y, app_mean, app_std, batch_size, cuda)
# print (time_taken, error, model_size)

# time_taken, error, model_size, test_pred = compute_statistics('tensor_decomposition_weights_rank_3',test_x, test_y, app_mean, app_std, batch_size, cuda)
# print (time_taken, error, model_size)

# time_taken, error, model_size, test_pred = compute_statistics('tensor_decomposition_weights_rank_4',test_x, test_y, app_mean, app_std, batch_size, cuda)
# print (time_taken, error, model_size)

# time_taken, error, model_size, test_pred = compute_statistics('Tensor decomposition weights/tensor_decomposition_weights_rank1.pth',test_x, test_y, app_mean, app_std, batch_size, cuda)
# print (time_taken, error, model_size)

# os.listdir()

# time_taken, error, model_size, test_pred = compute_statistics('global_pruning_weights.pth',test_x, test_y, app_mean, app_std, batch_size, cuda)
# print (time_taken, error, model_size)









