#1. Equivalent Index
class Solution:
    def findIndex(self, nums): 
        for l in range(len(nums)):
            a=sum(l[:i])
            b=sum(l[i+1:])
            if a==b:
                return i
            else return -1
#####################################
def findIndex(nums):
    l = len(nums)
    if l <= 2:
        return -1
    for n in range(l):
        if n > 0 and n < l-1:
            left = sum(nums[:n])
            right = sum(nums[n+1:])
            if left == right:
                return n
    return -1
    
#2. String mapping
def string_map(string1,string2):
    if len(string1) != len(string2):
        return False
    map_char = dict()
    for char1, char2 in zip(string2, string1):
        if char1 not in map_char:
            map_char[char1] = char2
        elif map_char[char1] != char2:
            return False
        else:
            return True
    return True  
    
#3. Merge sorted list
def mergesortedlist(test_list1,test_list2):
    size_1 = len(test_list1) 
    size_2 = len(test_list2) 
    res = [] 
    i, j = 0, 0
    while i < size_1 and j < size_2: 
        if test_list1[i] < test_list2[j]: 
            res.append(test_list1[i]) 
            i += 1
        else: 
            res.append(test_list2[j]) 
            j += 1
    res = res + test_list1[i:] + test_list2[j:]
    return res  
#############################    
def mergesortedlist(test_list1,test_list2):    
    res = sorted(test_list1 + test_list2) 
    return res
    
#4. Mosr Repetition (Frequency)
def most_frequent(List): 
    counter = 0
    num = List[0] 
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num    
    
#5. Binary Search

def binary_search(data, elem):
    low = 0
    high = len(data) - 1
    while low <= high: 
        middle = (low + high)//2
       
        if data[middle] == elem:
            return middle
        elif data[middle] > elem:
            high = middle - 1
        else:
            low = middle + 1
    return -1
    
data = [1,3,4,6,7,8,10,13,14,18,19,21,24,37,40,45,71]
elem = 7

binary_search(data, elem)    
    
#6. Max Width - Hard 
def match_length(words):
    if len(words) == 1:
        return words[0] + ' ' * (16-len(words[0]))
    total_space = 16 - len(''.join(words))
    space_add1 = total_space//(len(words) -1)
    space_add2 = total_space%(len(words) -1)
    for i in range(len(words)-1):
        words[i] += ' ' * space_add1
    for j in range(space_add2):
        words[j] += ' '
    return ''.join(words)

def length_16(words):
    result = []
    words_sub = []
    i = 0
    while i < len(words):
        words_sub.append(words[i])
        if len(' '.join(words_sub)) < 16:
            i+=1
        if len(' '.join(words_sub)) >= 16:
            words_sub = words_sub[:-1]
            result.append(match_length(words_sub))
            words_sub =[]
    result.append(match_length(words_sub))
    return  result     

#7. Permutation Palindrome
def perm_palindrome(myString):
    chars = []
    count = 0
    for item in myString:
        if item in chars:
            count -= 1
        else:
            count += 1
            chars.append(item)
    if count != 1:
        print('False')
    else:
        print('True')
###########################
def palindrome(input): 
    dict = {} 
    for i in input: 
        if i in dict: 
            dict[i]+= 1 
        else: dict[i] = 1
    flag = 0
    for i in dict.values():
        if i%2 != 0:
            flag += 1
    if flag > 1:
        return False
    return True        
    
 #8. One element missing
def return_missing_integer(list_x,list_y):
    return sum(list_x) - sum(list_y)   
    
    
#9. String Shift 
def rotateString(A: str, B: str) -> bool: 
        if(len(A) == len(B)): 
            buffer_string = A + A 
            return B in buffer_string 
        else: return False    
    
#10. String with same letters
def sameletters(a, b):
    a=''.join(sorted(a))
    b=''.join(sorted(b))
    return a==b    
   
#11. Find the missing number
def missingNumber(nums):
    for index,i in enumerate(nums):
        if i-index!=0:
            print(index)
            break
#####################
def missingNumber(nums):
    list = []
    for num in range(1000):
        if num in nums:
            list.append(num)
        else: 
            return num

#12. Greatest Common Denominator
def gcd(list1):
    a=min(list1)
    for i in range(a,0,-1):
        if all(num%i==0 for num in list1):
            return(i)

#13. Interquartile Distance
def interq(array):
    Q1 = np.percentile(array, 25, interpolation = 'midpoint') 
    Q3 = np.percentile(array, 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1 
    print(IQR)
#########################
import math
def interq(array):
    l=len(array)+1
    a=l*1/4-1
    b=l*3/4-1
    if a is not int:
        Q1 = (np.sort(array)[math.ceil(a)]+np.sort(array)[math.floor(a)])/2
        Q3 = (np.sort(array)[math.ceil(b)]+np.sort(array)[math.floor(b)])/2
    else:
        Q1 = np.sort(array)[a]
        Q3 = np.sort(array)[b]
    
    IQR = Q3 - Q1 
    print(IQR)
    
#14. Equivalent Index  
def findIndex(nums):
    l= len(nums)
    for i in range(0,l):
        sum_l=sum(nums[0:i])
        sum_r=sum(nums[i+1:])
        if sum_l==sum_r:
            return i
    return -1    
    
#15. Bernoulli Sample  
from scipy.stats import bernoulli
def sample_from_normal(p, n=1000):
    total = 0
    for _ in range(n):
        total += bernoulli.rvs(p)
    return total
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

















    
    
    
    
    
    
