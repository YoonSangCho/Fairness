import argparse 
import pandas as pd
from collections import defaultdict
from collections import Counter  
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import isotonic

#defining constants
AA = 0
White = 1   
test_models = [AA, White]   
old_model = 0
new_model = 1
train_models = [old_model, new_model]
prop1 = 0
prop4 = 1
prop5 = 2
prop6 = 3
prop9 = 4
new_prop_sum = 5     
old_props=[prop1, prop4, prop5, prop6, prop9]
all_props = old_props+[p+new_prop_sum for p in old_props]

def dynamics(model_type, final_values_matrix, sat_values_matrix, alpha, num_users, added_users_amt, added_users_prop, timesteps):
    #get smoothed predictor  
    AA_model_vals = [final_values_matrix[AA][p][model_type] for p in old_props] 
    AA_model_sat = [sat_values_matrix[AA][p][model_type] for p in old_props] 
    White_model_vals = [final_values_matrix[White][p][model_type] for p in old_props] 
    White_model_sat = [sat_values_matrix[White][p][model_type] for p in old_props]  
    x = [.1, .4, .5, .6, .9] 
    xinv = [1-xi for xi in x]  
    AA_ir = isotonic.IsotonicRegression(out_of_bounds='clip')
    AA_ir.fit(x, AA_model_vals)  
    White_ir = isotonic.IsotonicRegression(out_of_bounds='clip')
    White_ir.fit(xinv, White_model_vals)
    sat_AA_ir = isotonic.IsotonicRegression(out_of_bounds='clip')
    sat_AA_ir.fit(x, AA_model_sat)  
    sat_White_ir = isotonic.IsotonicRegression(out_of_bounds='clip')
    sat_White_ir.fit(xinv, White_model_sat)
    
    #run simulations
    curr_prop = alpha
    curr_AA_users = alpha*num_users 
    curr_White_users = num_users - curr_AA_users
    AA_retention_list = []
    num_AA_users_list = [] 
    AA_sat_list = [] 
    White_retention_list = []
    num_White_users_list = [] 
    White_sat_list = []  
    AA_added_per_time = ((added_users_amt*added_users_prop)*1.0)
    White_added_per_time = ((added_users_amt-(added_users_amt*added_users_prop))*1.0)
    AA_sat_list.append(sat_AA_ir.predict([curr_prop])[0])
    White_sat_list.append(sat_White_ir.predict([curr_prop])[0])
    for t in range(0,timesteps):
        AA_retention = AA_ir.predict([curr_prop])[0] 
        White_retention = White_ir.predict([1-curr_prop])[0]
        #print("AA Retention with "+str(curr_AA_users+curr_White_users)+" total users and "+str(curr_prop)+" AA proportion is "+str(AA_retention)+" of users at time "+str(t)+" .")
        #print("White Retention with "+str(curr_AA_users+curr_White_users)+" total users and "+str(1-curr_prop)+" White proportion is "+str(White_retention)+" of users at time "+str(t)+" .")
        AA_retention_list.append(AA_retention) 
        White_retention_list.append(White_retention) 
        #update
        curr_AA_users = (curr_AA_users * AA_retention) + AA_added_per_time
        curr_White_users = (curr_White_users * White_retention) + White_added_per_time
        num_AA_users_list.append(curr_AA_users)
        num_White_users_list.append(curr_White_users)  
        curr_prop = (1.0 * curr_AA_users) / (curr_AA_users+ curr_White_users) 
        AA_sat_list.append(sat_AA_ir.predict([curr_prop])[0])
        White_sat_list.append(sat_White_ir.predict([1-curr_prop])[0])
    return AA_retention_list, White_retention_list, num_AA_users_list, num_White_users_list, AA_sat_list, White_sat_list 


def newerr(x,y,yerr,shift, linestyle = '-', color='black',label='none'):
    ind_steps = np.arange(shift, len(x), step=4)
    plt.plot(x, y, color=color, linewidth=2.0, label=label, linestyle=linestyle)
    plt.errorbar(x[ind_steps], y[ind_steps], (yerr[0][ind_steps], yerr[1][ind_steps]), color=color, fmt='none', alpha=0.5)
    #plt.plot(x,y,color=color,label=label)
    #plt.plot(x,y-yerr[0],color=color,linestyle=':')
    #plt.plot(x,y+yerr[1],color=color,linestyle=':')
    #plt.fill_between(x, y-yerr[0], y+yerr[1],color=color, alpha=0.1)


def plot_sat_over_time(old_AA_sat, old_White_sat, new_AA_sat, new_White_sat, time_sat_fn, ylabel="User Satisfaction"):
    x = np.array([i+1 for i in range(0, len(old_AA_sat[1]))])
    errmap = lambda x: (x[1]-x[0], x[2]-x[1])
    plt.figure(figsize=(4,3))
    newerr(x, old_AA_sat[1], errmap(old_AA_sat), 0, color='green', label="ERM (AAE)")     
    newerr(x, new_AA_sat[1], errmap(new_AA_sat), 1, color='blue', label="DRO (AAE)") 
    newerr(x, old_White_sat[1], errmap(old_White_sat), 2, color='green', label="ERM (SAE)", linestyle=':') 
    newerr(x, new_White_sat[1], errmap(new_White_sat), 3, color='blue', label="DRO (SAE)", linestyle=':')    
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    if ylabel == "User Satisfaction":
        plt.legend(loc='best')#handles=[oa, na, ow, nw], loc=4)
    plt.tight_layout()
    plt.savefig(time_sat_fn)
    plt.close() 

def plot_users_over_time(old_AA_users, old_White_users, new_AA_users, new_White_users, time_retention_fn):
    x = np.array([i+1 for i in range(0, len(old_AA_users[1]))])
    errmap = lambda x: (x[1]-x[0], x[2]-x[1])
    plt.figure(figsize=(4,3))
    newerr(x, old_AA_users[1], errmap(old_AA_users),0, color='green', label="ERM (AAE)")     
    newerr(x, new_AA_users[1], errmap(new_AA_users), 1, color='blue', label="DRO (AAE)") 
    newerr(x, old_White_users[1], errmap(old_White_users), 2, color='green', label="ERM (SAE)", linestyle=':') 
    newerr(x, new_White_users[1], errmap(new_White_users), 3, color='blue', label="DRO (SAE)", linestyle=':')    
    plt.xlabel("Time")
    plt.ylabel("Number of Users") 
    #plt.legend(loc='best')#handles=[oa, na, ow, nw], loc=4)
    plt.tight_layout()
    plt.savefig(time_retention_fn)
    plt.close()

        
def plot_retention_with_smoothing(final_values_matrix, ylabel, fn):
    AA_old_y = [final_values_matrix[AA][p][old_model] for p in old_props]
    AA_new_y = [final_values_matrix[AA][p][new_model] for p in old_props]
    White_old_y = [final_values_matrix[White][p][old_model] for p in old_props]
    White_new_y = [final_values_matrix[White][p][new_model] for p in old_props]
    x = [.1, .4, .5, .6, .9]
    xinv = [1-xi for xi in x] 
    #create regresiion 
    AA_old_ir = isotonic.IsotonicRegression(out_of_bounds='clip')
    AA_new_ir = isotonic.IsotonicRegression(out_of_bounds='clip')
    White_old_ir = isotonic.IsotonicRegression(out_of_bounds='clip')
    White_new_ir = isotonic.IsotonicRegression(out_of_bounds='clip') 
    
    #fit
    AA_old_ir.fit(x, AA_old_y)
    AA_new_ir.fit(x, AA_new_y)
    White_old_ir.fit(xinv, White_old_y)
    White_new_ir.fit(xinv, White_new_y) 
    
    #predict
    AA_old_pred = AA_old_ir.predict(x)
    AA_new_pred = AA_new_ir.predict(x)
    White_old_pred = White_old_ir.predict(xinv)
    White_new_pred = White_new_ir.predict(xinv)
    
    #plot
    wn, = plt.plot(xinv, White_new_pred, color=(0, .99, .99), label="White Users with Model Correction")
    wo, = plt.plot(xinv, White_old_pred, color=(0, .4, .4), label="White Users w/o Model Correction")
    an, = plt.plot(x, AA_new_pred, color=(0.99, 0, 0), label="AA Users with Model Correction")
    ao, = plt.plot(x, AA_old_pred, color=(0.4, 0, 0), label="AA Users w/o Model Correction")
    plt.xlabel("Proportion of African-American Classfied Tweets in Model")
    plt.ylabel(ylabel)
    plt.legend(handles=[wn, wo, an, ao], loc=4)
    plt.axis([0.1, .9, 0.5, 1])
    plt.savefig(fn)
    plt.close() 
    
    
def plot_retention(final_values_matrix, ylabel, fn): 
    AA_old_y = [final_values_matrix[AA][p][old_model] for p in old_props]
    AA_new_y = [final_values_matrix[AA][p][new_model] for p in old_props]
    White_old_y = [final_values_matrix[White][p][old_model] for p in old_props]
    White_new_y = [final_values_matrix[White][p][new_model] for p in old_props]
    x = [.1, .4, .5, .6, .9]     
    #Add to plot
    wn, = plt.plot(x, White_new_y, color=(0, .99, .99), label="White Users with Model Correction")
    wo, = plt.plot(x, White_old_y, color=(0, .4, .4), label="White Users w/o Model Correction")
    an, = plt.plot(x, AA_new_y, color=(0.99, 0, 0), label="AA Users with Model Correction")
    ao, = plt.plot(x, AA_old_y, color=(0.4, 0, 0), label="AA Users w/o Model Correction")
    plt.xlabel("Proportion of African-American Classfied Tweets in Model")
    plt.ylabel(ylabel)
    plt.legend(handles=[wn, wo, an, ao], loc=4)
    plt.axis([0.1, 0.9, 0.5, 1])
    plt.savefig(fn) 
    plt.close()
    
def satisfaction(satisfaction_values_matrix, threshold=True): 
    for test_model in test_models:
        for prop in old_props:
            for train_model in train_models:
                new_prop = prop
                if(train_model == new_model):
                    new_prop = prop+new_prop_sum
                satisfaction_vals = sumsq6[test_model][new_prop]["sum"]
                satisfaction_rate = 0
                for v in satisfaction_vals:
                    if threshold and v >= 2:
                        satisfaction_rate += 1  
                    elif not threshold:
                        satisfaction_rate += v
                satisfaction_rate  =  (satisfaction_rate*1.0)/len(satisfaction_vals)
                satisfaction_values_matrix[test_model][prop][train_model] = satisfaction_rate
        
def retention(final_values_matrix): 
    for test_model in test_models:
        for prop in old_props:
            for train_model in train_models:
                new_prop = prop
                if(train_model == new_model):
                    new_prop = prop+new_prop_sum
                retention_counter = Counter(yes[test_model][new_prop]["sum"])
                retention_rate = retention_counter['Yes']*1.0/(retention_counter['No']+retention_counter['Yes'])
                final_values_matrix[test_model][prop][train_model] = retention_rate

def aggregate(list_aggregates, filtered_data):
    for key, row in filtered_data.iterrows():
        if(row["Input.tweetsource"] == AA and not pd.isnull(row["Input.hit"])):
            for ag_indx, ag_val in enumerate(list_aggregates):
                if(row[aggregate_strings[ag_indx]] >= 1.0):
                    list_aggregates[ag_indx][AA][row["Input.hit"]]["sum"].append(row[aggregate_strings[ag_indx]])
        elif(row["Input.tweetsource"] == White and not pd.isnull(row["Input.hit"])):
            for ag_indx, ag_val in enumerate(list_aggregates):
                if(row[aggregate_strings[ag_indx]] >= 1.0):
                    list_aggregates[ag_indx][White][row["Input.hit"]]["sum"].append(row[aggregate_strings[ag_indx]])
    
    for ag_indx, ag_val in enumerate(list_aggregates[:-1]):
        for prop in all_props:
            for model in test_models:
                list_aggregates[ag_indx][model][prop]["average"] = np.mean(list_aggregates[ag_indx][model][prop]["sum"])
                list_aggregates[ag_indx][model][prop]["std"] = np.std(list_aggregates[ag_indx][model][prop]["sum"]) 
    
def read_data(fn):
    data = pd.read_csv(fn) 
    filtered_data = data.loc[:,['Input.tweetsource', 'Input.seed', 'Input.hit', 'Answer.Q2Answer', 'Answer.Q4Answer', 'Answer.Q5Answer', 'Answer.Q6Answer']]
    return filtered_data


def simulation(final_values_matrix, satisfaction_values_matrix, t, n, p, p_new, a):
    old_AA_retention_list, old_White_retention_list, old_AA_users_list, old_White_users_list, old_AA_sat, old_White_sat = dynamics(old_model, final_values_matrix=final_values_matrix, sat_values_matrix=satisfaction_values_matrix, alpha = p, num_users=n, added_users_amt=a, added_users_prop=p_new, timesteps=t)
    new_AA_retention_list, new_White_retention_list, new_AA_users_list, new_White_users_list,  new_AA_sat, new_White_sat = dynamics(new_model, final_values_matrix=final_values_matrix, sat_values_matrix=satisfaction_values_matrix, alpha = p, num_users=n, added_users_amt=a, added_users_prop=p_new, timesteps=t)
    return old_AA_retention_list, new_AA_retention_list, old_White_retention_list, new_White_retention_list, old_AA_users_list, new_AA_users_list, old_White_users_list, new_White_users_list, old_AA_sat, old_White_sat, new_AA_sat, new_White_sat
                         
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()                                               
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args() 
    print("Reading csv file"+str(args.file))  
    filtered_data = read_data(args.file)
    
    #questions in csv files we are aggregating
    sumsq2 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sumsq4 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sumsq6 = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
    yes = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
    list_aggregates = [sumsq2, sumsq4, sumsq6, yes]
    aggregate_strings = ["Answer.Q2Answer", "Answer.Q4Answer", "Answer.Q6Answer", "Answer.Q5Answer"] 
     
    print("Aggregating across rows...") 
    aggregate(list_aggregates, filtered_data) 
    
    #Gather Retention
    print("Calculating retention rates and creating matrix...")
    final_values_matrix = np.zeros((len(test_models), len(old_props), len(train_models)))  
    retention(final_values_matrix)  
    
    #Gather Satisfaction
    print("Calculating satisfaction values and creating matrix...") 
    satisfaction_values_matrix = np.zeros((len(test_models), len(old_props), len(train_models)))
    satisfaction(satisfaction_values_matrix, True)      
    
    #Bootstrapped dyanmics
    np.random.seed(0)
    val=100.0
    prop=0.5
    p_new = 0.5
    tmax=50
    ret_all_sim_outputs = []
    ret_all_sim_outputs_2 = []  
    sat_all_sim_outputs = []
    sat_vs_time_all_outputs = [] 
    for replicate in range(100):
        choices = np.random.choice(filtered_data.shape[0], filtered_data.shape[0], replace=True)
        resampled_data = filtered_data.iloc[choices]
        sumsq2 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        sumsq4 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        sumsq6 = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
        yes = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
        list_aggregates = [sumsq2, sumsq4, sumsq6, yes]
        aggregate(list_aggregates, resampled_data)
        final_values_matrix = np.zeros((len(test_models), len(old_props), len(train_models)))  
        retention(final_values_matrix)    
        satisfaction_values_matrix = np.zeros((len(test_models), len(old_props), len(train_models)))
        satisfaction(satisfaction_values_matrix, False)
        ret_sim_output = simulation(final_values_matrix, satisfaction_values_matrix, tmax, val, prop, p_new, val)
        ret_sim_output_2 = simulation(final_values_matrix, final_values_matrix, tmax, val, prop, p_new, val)
        ret_all_sim_outputs.append(ret_sim_output)
        ret_all_sim_outputs_2.append(ret_sim_output_2)    

    user_idxs = [4,5,6,7] 
    sat_idxs = [8, 9, 10, 11]
    ret_replicate_matrix = np.array([[sim[typeid] for sim in ret_all_sim_outputs] for typeid in user_idxs])
    ret_replicate_percentiles = np.percentile(ret_replicate_matrix,[25,50,75], axis=1)
    #Treating satisfaction as indicator of retention of users - higher % of users that are satisfied , % percentage to stay in system
    sat_replicate_matrix = np.array([[sim[typeid] for sim in ret_all_sim_outputs] for typeid in sat_idxs])
    sat_replicate_percentiles = np.percentile(sat_replicate_matrix,[25,50,75], axis=1)
    prefix = ''
    plot_users_over_time(ret_replicate_percentiles[:,0,:], ret_replicate_percentiles[:,2,:], ret_replicate_percentiles[:,1,:], ret_replicate_percentiles[:,3,:], prefix+str(10*prop)+"retention_autocomplete.png") 
    plot_sat_over_time(sat_replicate_percentiles[:,0,:], sat_replicate_percentiles[:,1,:], sat_replicate_percentiles[:,2,:], sat_replicate_percentiles[:,3,:], prefix+str(10*prop)+"satisfaction_time_autocomplete_nothreshold.png")
    sat_replicate_matrix = np.array([[sim[typeid] for sim in ret_all_sim_outputs_2] for typeid in sat_idxs])
    sat_replicate_percentiles = np.percentile(sat_replicate_matrix,[25,50,75], axis=1)
    plot_sat_over_time(sat_replicate_percentiles[:,0,:], sat_replicate_percentiles[:,1,:], sat_replicate_percentiles[:,2,:], sat_replicate_percentiles[:,3,:], prefix+str(10*prop)+"retention_time_autocomplete_nothreshold.png",
                       ylabel = 'Retention rate')