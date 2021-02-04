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
    
#6. Last Page Number --Write a function to return the last page number in the string. If the string of integers is not in correct page order, return the last number in order.
def lastpage(input):
    i=int(input[0])
    start=0
    while  i >= int(input[0]):
        a=len(str(i+1))
        if start+a+1 > len(input):
            return (i)
        if start+a+1 <= len(input):
            if i+1 != int(input[start+1:start+1+a]):
                            return(i)
        start +=a
        i +=1    
#############################
def last_page_number(string):
    pos=0
    page=0
    while pos < len(string):
        page_str=str(page+1)
        if string[pos:pos+len(page_str)]==page_str:
            pos+=len(page_str)
            page+=1
        else:
            break
    return page

#7. Friendship Timeline -- lists the pairs of friends with their corresponding timestamps of the friendship beginning and then the timestamp of the friendship ending.

def friendships(added,friends_removed):
    friendships=[]
    dic={}
    for i in added:
        for j in friends_removed:
            if sorted(i['user_ids'])==sorted(j['user_ids']):
                dic['user_ids']=sorted(i['user_ids'])
                dic['start_date']=i['created_at']
                dic['end_date']=j['created_at']
                friendships.append(dic)
                dic={}
                friends_removed.remove(j)
                break
    print(sorted(friendships, key=lambda x: x['user_ids']))


#8. New Resumes    
def new_resumes(existing_ids,names,urls):
    dic=[]
    output=[]
    id=[i.split('/')[-1] for i in urls] 
    for key in id: 
        for value in names: 
            dic.append((value,key))
            names.remove(value) 
            break  
    for key in dic:
        if int(key[1]) not in existing_ids:
            output.append(key)
    print(output)

#9. Get Top N words    
def topnwords(n,posting):
    post=posting.lower().replace('\n','').split(' ')
    table=[]
    for i in set(post):
        table.append((i,post.count(i)))
    print (sorted(table, key=lambda x: (x[1], x[0]), reverse=True)[0:n])    

#10. Find Bigrams
def findbigrams(sentence):
    bigrams=[]
    sen=sentence.lower().replace('\n','').split(" ")
    for i in range(len(sen)-1):
        bigrams.append((sen[i],sen[i+1]))
    print(bigrams)
   
#11. Compute Deviation
def deviation(input):
    result={}
    for i in input:
        sum=0
        var=0
        for j in i['values']:
            sum+=j
            mean=sum/len(i['values'])
        for k in i['values']:
            var+=(k-mean)**2
        std=((var/len(i['values']))**0.5)
        result[i['key']]=round(std,2)       
    print (result)
#################################
import math 
def compute_dev(dictionary_list):
    result={}
    for dictionary in dictionary_list:
        v=dictionary['values']
        mean=sum(v)/len(v)
        deviation= math.sqrt(sum( (x-mean)**2 for x in v )/len(v))
        result[dictionary['key']]=deviation
    return result
    

    
    
    
    
    
    
