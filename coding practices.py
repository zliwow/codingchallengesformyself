# LC 347 M Top K Frequent Elements 

from calendar import c
from inspect import stack
import re


def topFrequent(nums, k):
    dict = {}
    for i in nums:
        if i not in dict:
            dict[i] = 1
        else:
            dict[i] += 1

    res = sorted(dict, key=dict.get)
    print (res[-k:])
    return res[-k:]
    
# topFrequent([1,1,1,2,2,3],2) Hash

# LC 118 E Pascal's Triangle

def generate(numRows):
    res = [[1]]

    for i in range(numRows -1):
        temp = [0] + res[-1] + [0]
        row = []
        for j in range(len(res[-1]) + 1):
            row.append(temp[j]+temp[j+1])
        res.append(row)
    print(res)
    return res



# generate(5) 
# 0 1 0 top of the tree

# LC 119 E Pascal's Triangle II
def getRow(rowIndex):
        res = [[1]]

        for i in range(rowIndex):
            temp = [0] + res[-1] + [0]
            row = []
            for j in range(len(res[-1]) + 1):
                row.append(temp[j]+temp[j+1])
            res.append(row)
        print(res[-1])
        return res[-1]

# getRow(5)
# omit row -1 to become it's index     couting original res as first line

# LC 150 Evaluate Reverse Polish Notation
# First Attempt
# import math
# def lc(tokens):
#     sb = ["+","-","*","/"]
    
#     while len(tokens) > 1:
#         for i,j in enumerate(tokens):
#             if j in sb:
#                 temp = ""
#                 temp = f"{tokens[i-2]}  {tokens[i]}  {tokens[i-1]}"
#                 tokens.remove(tokens[i-2])
#                 tokens.remove(tokens[i-2])
#                 tokens.remove(tokens[i-2])
#                 fomular = eval(temp)
#                 if fomular > 0:
#                     fomular = math.floor(fomular)
#                 else:
#                     fomular = math.ceil(fomular)
#                 tokens.insert(i - 2,fomular)
#                 break
#     print(tokens[0])
#     return tokens[0]
            
# Failed from Math problem
    
# lc(["10","6","9","3","+","-11","*","/","*","17","+","5","+"])

def evalPRN(tokens):
    stack = []
    for char in tokens:
        if char == "+":
            stack.append(stack.pop() + stack.pop())
        elif char == "-":
            a, b = stack.pop(), stack.pop()
            stack.append(b - a)
        elif char == "*":
            stack.append(stack.pop() * stack.pop())
        elif char == "/":
            a, b = stack.pop(), stack.pop()
            stack.append(int(b / a))
        else:
            stack.append(int(char))
    return stack[0]
    
evalPRN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"])

# 112. Path Sum 
def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    def dfs(node, curSum):
        if not node:
            return False
        
        curSum += node.val
        if not node.left and not node.right:
            return curSum == targetSum
        
        return (dfs(node.left, curSum) or 
                dfs(node.right, curSum))
    return dfs(root, 0)