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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
