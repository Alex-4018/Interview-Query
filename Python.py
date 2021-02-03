#1. Sum to N -- Given a list of integers, find all combinations that equal the value N.
def sumton(integers,target):
    dp = [[] for i in range(target+1)]
    for c in integers:
            for i in range(target+1):
                if i<c: continue
                if i==c:
                    dp[i].append([c])
                else:
                    for blist in dp[i-c]:
                        dp[i].append(blist+[c])
    return dp[target]
    
    
#2. Nightly Job
Every night between 7pm and midnight, two computing jobs from two different sources are randomly started with each one lasting an hour. Unfortunately, when the jobs simultaneously run, they cause a failure in some of the other company’s nightly jobs, resulting in downtime for the company that costs $1000. The CEO, who only has enough time today to hear no more than one word, needs a single number representing the annual (365 days) cost of this problem. Write a function to simulate this problem and output an estimated cost. How would you solve this using probability?

import numpy as np
process_1 = np.random.randint(0, 5*60*60, size=10**7)
process_2 = np.random.randint(0, 5*60*60, size=10**7)
overlap_percentage = np.mean(np.abs(process_1 - process_2) <= 3600)
annual_cost = overlap_percentage * 365 * 1000
annual_cost
#########################    
from numpy import random
import matplotlib.pyplot as plt
import numpy as np

n=100000 
sttim1 = random.uniform(7,12,n) 
sttim2 = random.uniform(7,12,n)
print('n = ',n)
kk = 0
for i in np.arange(n):
    if np.abs(sttim2[i] - sttim1[i]) < 1.0 :
            kk += 1
print('Percentage time failure occurs = ',100*kk/n)
print('Loss = ',kk/n*365*1000)    
################################    
def cost(days):
    cost = 0
    for day in range(days):
        i = np.random.uniform(7,12)
        j = np.random.uniform(7,12)
        if -1 < i-j < 1:
            cost +=1000
    return cost 

simulation = []
for i in range(1000):
    simulation.append(cost(365))
    
np.array(simulation).mean()    
    
#3. Multimodal Sample   
import random
def sample_multimodal(keys, weights, n):
    return random.choices(keys, weights=weights,  k=n)
###################################
def select_n(Keys, weights, n):
    prob_int = np.cumsum(np.array(weights)/np.sum(weights))
    color_list = []
    for x in range(n):
        color_list.append(Keys[np.sum(random.random() > prob_int)])

    print(color_list)    
    
#4. Acquisition Threshold
def bonus_threshold(client_spends_list, pct = 0.10):
    client_spends_list.sort(reverse = True)
    bin = int(len(client_spends_list) * pct)
    return client_spends_list[bin]

#5. Weekly Aggregation --Given a list of timestamps in sequential order, return a list of lists grouped by week (7 days) using the first timestamp as the starting point.    
from datetime import datetime
from collections import defaultdict
def read_date(date):
    return datetime.strptime(date, '%Y-%m-%d')

def weeks_from_date(starting_date, date):
    delta = read_date(date) - read_date(starting_date)
    return delta.days // 7

def group_by_weeks(ts):
    starting_date = ts[0]
    grouped = defaultdict(list)
    for date in ts:
        grouped[weeks_from_date(starting_date, date)].append(date)
    return list(grouped.values())    
###################################    
import pandas as pd
dataframe = pd.DataFrame({'time_stamp':ts}) 
dataframe['time_stamp'] = pd.to_datetime(dataframe['time_stamp']) 
dataframe['Week_Number'] = dataframe['time_stamp'].dt.week 
---dataframe['Week_Number'] = ((dataframe['time_stamp']-dataframe['time_stamp'][0])//7).dt.days---
dataframe['time_stamp'] = dataframe['time_stamp'].astype(str)

output = dataframe.groupby('Week_Number')['time_stamp'].apply(list).tolist()   
################################
import pandas as pd
def get_grouped_week(ts):
    dates = pd.DataFrame(ts)
    dates[0] = pd.to_datetime(dates[0])
    dates['week'] = dates[0].apply(lambda x: x.week)
    
    grouped_list = []
    for val in set(dates['week'].values):
        week_val = dates[dates['week'] == val]
        date_vals = list(week_val[0].values)
        grouped_list.append(date_vals)
    return grouped_list    
    
    
    
    
    
    
    
    
    
    
    
