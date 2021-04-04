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
 ###############################
 def backtrack(rem, curr, first):
    if rem==0:
        output.append(curr[:])
    elif rem<0:
        return 

    for i in range(first, len(integers)):
        curr.append(integers[i])
        backtrack(rem-integers[i], curr, i)
        curr.pop(-1)
output=[]
backtrack(target, [], 0)
print(output)
    
#2. Nightly Job
Every night between 7pm and midnight, two computing jobs from two different sources are randomly started with each one lasting an hour. 
Unfortunately, when the jobs simultaneously run, they cause a failure in some of the other company’s nightly jobs, resulting in downtime for the company that costs $1000. 
The CEO, who only has enough time today to hear no more than one word, needs a single number representing the annual (365 days) cost of this problem. 
Write a function to simulate this problem and output an estimated cost. How would you solve this using probability?

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
#np.mean(simulation) 
    
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


ts = [
    '2019-01-01', 
    '2019-01-02',
    '2019-01-08', 
    '2019-02-01', 
    '2019-02-02',
    '2019-02-05',
]

output = [
    ['2019-01-01', '2019-01-02'], 
    ['2019-01-08'], 
    ['2019-02-01', '2019-02-02'],
    ['2019-02-05'],
]
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
    dates[0] = pd.to_datetime(dates[0]) # 0 is the name of the column
    dates['week'] = dates[0].apply(lambda x: x.week)
    
    grouped_list = []
    for val in set(dates['week'].values):
        week_val = dates[dates['week'] == val]
        date_vals = list(week_val[0].values) # date_vals = list(week_val[0])
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

input = '12345'
output = 5

input = '12345678910111213'
output = 13

input = '1235678'
output = 3


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

friends_added = [{'user_ids': [1, 2], 'created_at': '2020-01-01'},
                 {'user_ids': [3, 2], 'created_at': '2020-01-02'},
                 {'user_ids': [2, 1], 'created_at': '2020-02-02'},
                 {'user_ids': [4, 1], 'created_at': '2020-02-02'}]

friends_removed = [{'user_ids': [2, 1], 'created_at': '2020-01-03'},
                   {'user_ids': [2, 3], 'created_at': '2020-01-05'},
                   {'user_ids': [1, 2], 'created_at': '2020-02-05'}]


friendships = [{
    'user_ids': [1, 2],
    'start_date': '2020-01-01',
    'end_date': '2020-01-03'
  },
  {
    'user_ids': [1, 2],
    'start_date': '2020-02-02',
    'end_date': '2020-02-05'
  },
  {
    'user_ids': [2, 3],
    'start_date': '2020-01-02',
    'end_date': '2020-01-05'
  },
]

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

existing_ids = [15234, 20485, 34536, 95342, 94857]
names = ['Calvin', 'Jason', 'Cindy', 'Kevin']
urls = [
    'domain.com/resume/15234', 
    'domain.com/resume/23645', 
    'domain.com/resume/64337', 
    'domain.com/resume/34536',
]    
    
    
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

input = [
    {
        'key': 'list1',
        'values': [4,5,2,3,4,5,2,3],
    },
    {
        'key': 'list2',
        'values': [1,1,34,12,40,3,9,7],
    }
]

output -> {'list1': 1.12, 'list2': 14.19}

#12. Stop Words Filter
def stripped_paragraph(stopwords,paragraph):
    par=paragraph.split()
    print(par)
    for i in par:
        if i in stopwords:
            par.remove(i)
    return " ".join(par)    

#13. Buy or Sell
def  get_profit_dates(stock_prices, dts):
    min_price=min(stock_prices)
    min_index=stock_prices.index(min_price)
    max_price=max(stock_prices)
    max_index=stock_prices.index(max_price)
    print((max_price-min_price,dts[min_index],dts[max_index]))    
######################
def get_max_profit(stock_prices, dts):
    max_profit = 0
    buy=None
    sell=None
    for i in range(len(stock_prices)-1):
        for j in range(i+1, len(stock_prices)):
            diff = stock_prices[j] -  stock_prices[i]
            if max_profit < diff:
                max_profit = diff
                buy=i
                sell=j
    return max_profit, dts[buy], dts[sell]
    
#14. Recurring character
def recurringvchar(input):
    letter=[]
    for i in input:
        if i not in letter:
            letter.append(i)
        else:
            if i in letter:
                return i
    return 'None'
        
#15. Biggest Tip   
def biggesttip(user_ids,tips):
    biggest_tips=0
    for i in tips:
        if i>biggest_tips:
            biggest_tips=i
            index=tips.index(i)
    return biggest_tips, user_ids[index]

#16. Generate Normal Distribution
import numpy as np 
import matplotlib.pyplot as plt

def generate_and_plot_samples(N): 
    mean = 0 
    std = 1 
    data = np.random.normal(mean, std, N) 
    plt.hist(data) 
    plt.show()

#16. Normalize Grades
def normalize_grades(tuples):
    low_value=min(x[1] for x in tuples)
    high_value=max(x[1] for x in tuples)
    return [(x[0], round(((x[1]-low_value)/(high_value-low_value)),2)) for x in tuples]
        
#17. RMS Error
def calculate_rmse(list1,list2):
    error_sum=0
    for i in range(len(list1)):
        error=(list1[i]-list2[i])**2
        error_sum+=error
    return (error_sum/len(list1))**0.5

#18. Counting file lines
def count_lines(filenm):
    i=0
    with open("log.txt") as f:
    count = 0
    for l in f:
        count  += i

#19. Weighted Keys
def random_key(weights):
    result=[]
    total=sum(weights.values())
    for key, value in weights.items():
           print((key,"{:.1%}".format(value/total)))
 
#20. Replace words with stems
def replaceword(roots, sentence):
    sen=sentence.lower().split(" ")
    for i in range(len(sen)):
        for j in roots:
            length=len(j)
            if sen[i][0:length]==j:
                sen[i]=j
    result=' '.join(sen)
    print('"',result,'"')
    
    
#21. String Subsequence
def isSubSequence(string1,string2):
    str1=string1
    str2=string2
    for i in str2:
        if i not in str1:
            str2=str2.replace(i,'')
    if str2==str1:
        return True
    else:
        return False

#22. Flatten JSON
def solution(input):
    output = {}
    for kys in input.keys():
        for ky in input[kys].keys():
            output[str(kys+'_'+ky)] = input[kys][ky]
    return output

Input: { 'a':{'b':'c',
              'd':'e'} }
Output: {'a_b':'c', 'a_d':'e'}

#23. Fizzbuzz
def fizzbuzz(n):
    result=[]
    for i in range(1,n+1):
        if i%3 == 0 and i%5 == 0:
            result.append('Fizzbuzz')
        elif i%3==0:
            result.append('Fizz')
        elif i%5==0:
            result.append('Buzz')
        else:
            result.append(i)
    print(result)
    
#24. Gaussian Generation
from scipy import stats
def generate_distribution(N,M):
    return stats.norm.rvs(loc = M, scale = 1, size = N).astype(int).tolist()
    
generate_distribution(N = 9 , M =3)

#25. N-gram Dictionary
def solution(string,n):
    output={}
    for i in range(len(string)-n+1):
        temp=string[i:i+n]
        if temp in output:
            output[temp]+=1
        else:
            output[temp]=1
    return output     

string = 'banana' 
n=3 
output = {'ban':1, 'ana':2, 'nan':1} 


#26. Bucket Test Scores
import pandas as pd
import numpy as np
n = 2000
df = pd.DataFrame(zip(list(range(n)), np.random.randint(9,12+1,n), np.random.randint(0,100+1,n)), columns=['user_id', 'grade', 'test_score']).set_index('user_id')
#print(df)
bins = [0, 50, 75, 90, 100]
group_size = df.groupby(['grade', pd.cut(df['test_score'], bins, labels=['<50', '<75', '<90', '<100'], right=False)]).size()
print(group_size)
percentage = (group_size.groupby('grade').cumsum() / df.groupby('grade').size() * 100).rename('percentage').round(0).astype(str) + '%'
print(percentage.reset_index().to_markdown())

#27. Moving Window
def moving_window(input1,val):
    result=[]
    for i in range(len(input1)):
        if i< val-1:
            result.append(sum(input1[0:i+1])/(i+1))
        else:
            result.append((sum(input1[i-val+1:i+1]))/val)
    return result

#28. Minimum Change
def find_change(cents):
    count=0
    cents_set=[1,10,25]
    cents_set.reverse()
    for i in cents_set:
        count+=(cents//i)
        cents-=(cents//i)*i
    return count
    
#29. Truncated Distribution
import scipy.stats as st
def truncateddist(n, percentilethreshold): 
    counter = 1 
    result = [ ] 
    while counter <=n:
        randomsample = round(st.norm(2,1).rvs(1)[0],1) 
        if randomsample <=st.norm(2,1).ppf(percentilethreshold): 
            result.append(randomsample) 
            counter+=1 
    return result

#30. Prime to N
def primeton(n):
    result=[]
    for i in range(2,n):
        if all(i %j!=0 for j in range(2,i)):
            result.append(i)
    return result

#31. Business Days
import pandas as pd
date1 = '2021-01-31'
date2 = '2021-02-18'

#print(pd.bdate_range(date1, date2))
print(pd.bdate_range(date1, date2).shape[0])
########################
from datetime import date,timedelta
def businessday(date1,date2):
    fromdate = date(2010,1,1)
    todate = date(2010,3,31)
    daygenerator = (fromdate + timedelta(x + 1) for x in range((todate - fromdate).days))
    return sum(1 for day in daygenerator if day.weekday() < 5)

#32. Variance
def variance(test_list):
    avg=sum(test_list)/len(test_list)
    variance=0
    for i in test_list:
        variance+= (i-avg)**2
    return variance/len(test_list)

#33. TF-IDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
p1 = 'I saw a cat'
p2 = 'I saw a dog'
p3 = 'I saw a horse'
p4 = 'I have a dog'

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([p1,p2,p3,p4])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df.head()
