# things to learn  bucketsort , heap


# LC 347 M Top K Frequent Elements 

from turtle import left


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
def hasPathSum(root, targetSum):
    def dfs(node, curSum):
        if not node:
            return False
        
        curSum += node.val
        if not node.left and not node.right:
            return curSum == targetSum
        
        return (dfs(node.left, curSum) or 
                dfs(node.right, curSum))
    return dfs(root, 0)

# 27. Remove Element  needs to return both the number and edit the list IN-PLACE
def removeElement(nums, val):
    k = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1
    return k
removeElement([3,2,2,3], 2)

# 48. Rotate Image   IN-PLACE  MATRIX   
def rotate(matrix):
    l = 0
    r = len(matrix) - 1
    while l < r:
        for i in range(r - l):
            top, bottom = l , r

            # Save top left
            topLeft = matrix[top][l + i]
            # bottom left to top left
            matrix[top][l + i] = matrix[bottom - i][l]

            # bottom right to bottom left
            matrix[bottom - i][l] = matrix[bottom][r - i]

            # top right to bottom right
            matrix[bottom][r - i] = matrix[top + i][r]

            # topLeft to top right
            matrix[top][r] = topLeft

        r -= 1
        l += 1
    return matrix



rotate([[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]])

# 1886 Determine Whether Matrix Can Be Obtained By Rotation    MATRIX VALIDATION 
def findRotation(matrix, target):
    l , r = 0, len(matrix) - 1
    for i in range(4): # To ensure the matrix spin 4 times. 90 degree each time.
        while l < r:
            for i in range(r - l):
                top, bottom = l , r
                topLeft = matrix[top][l + i]
                matrix[top][l + i] = matrix[bottom - i][l]
                matrix[bottom - i][l] = matrix[bottom][r - i]
                matrix[bottom][r - i] = matrix[top + i][r]
                matrix[top + i][r] = topLeft
                if matrix == target:
                    return True
            r -= 1
            l += 1
        l , r = 0, len(matrix) - 1 # To reset the spin algo, like rewinding
    return False
findRotation([[0,0,0],[0,1,0],[1,1,1]],[[1,1,1],[0,1,0],[0,0,0]])



# 219. Contains Duplicate II

def containsNearbyDuplicate(nums, k):
    dict = {}
    for i, n in enumerate(nums):
        if n not in dict:
            dict[n] = i
        else:
            if i - dict[n] <= k:
                return True
            dict[n] = i
    return False



containsNearbyDuplicate([1,2,3,1,2,3], 2)

# 88. Merge Sorted Array 

def merge(nums1, m, nums2, n):
    # last index
    last = m + n -1
    # merge in reverse order
    while m > 0 and n > 0:
        if nums1[m -1] > nums2[n - 1]:
            nums1[last] = nums1[m - 1]
            m -= 1
        else:
            nums1[last] = nums2[n - 1]
            n -= 1
        last -= 1
    # fill nums1 with leftover nums2 elements
    while n > 0:
        nums1[last] = nums2[n - 1]
        n , last = n - 1, last -1

merge([1,2,3,0,0,0],3,[2,5,6],3)

# 242. Valid Anagram
def isAnagram(s, t):
    # return sorted(s) == sorted(t) May be asked to write custom sorting function

    # return Counter(s) == Counter(t) One liner

    if len(s) != len(t):
        return False
    countS, countT = {}, {}

    for i in range(len(s)):
        countS[s[i]] = 1 + countS.get(s[i], 0)
        countT[t[i]] = 1 + countT.get(t[i], 0)

    for c in countS:
        if countS[c] != countT.get(c, 0):
            return False
    return True

isAnagram(s = "anagram", t = "nagaram")

# 438. Find All Anagrams in a String
def findAnagram(s, p):
    # for edge case
    if len(p) > len(s):
        return []
    # set up the base for the two hashes
    pCount, sCount = {}, {}
    for i in range(len(p)):
        pCount[p[i]] = 1 + pCount.get(p[i], 0)
        sCount[s[i]] = 1 + sCount.get(s[i], 0)

    res = [0] if sCount == pCount else []
    # sliding window
    l = 0
    for r in range(len(p), len(s)):
        sCount[s[r]] = 1 + sCount.get(s[r], 0)
        sCount[s[l]] -= 1

        if sCount[s[l]] == 0:
            sCount.pop(s[l])
        l += 1
        if sCount == pCount:
            res.append(l)
    return res

findAnagram( "cbaebabacd","abc")

# 1. Two Sum
def twoSum(nums, target):
    dict = {}
    for i, j in enumerate(nums):
        diff = target - j
        if diff in dict:
            return [dict[diff], i]
        dict[j] = i
twoSum([2,7,11,15], 9)

# 167. Two Sum II - Input Array Is Sorted
def twoSumSecond(numbers, target):
    l = 0
    r = len(numbers) - 1
    while l < r:
        curSum = numbers[l] + numbers[r]
        if curSum > target:
            r -= 1
        elif curSum < target:
            l += 1
        else:
            # print(l + 1, r + 1)
            return l + 1, r + 1
twoSumSecond([2,7,11,15], 9)

# 15. 3Sum
def threeSum(nums):
    res = []
    nums.sort()

    for i, j in enumerate(nums):
        if i > 0 and j == nums[i - 1]:
            continue
        l = i + 1
        r = len(nums) - 1
        while l < r:
            curSum = j + nums[l] + nums[r]
            if curSum > 0:
                r -= 1
            elif curSum < 0:
                l += 1
            else:
                res.append([j, nums[l], nums[r]])
                l += 1
                while nums[l] == nums[l - 1] and l < r:
                    l += 1
    # print(res)
    return res

threeSum([-1,0,1,2,-1,-4])

# 49. Group Anagrams
def groupAnagrams(strs):
    from collections import defaultdict
    res = defaultdict(list) # mapping charCount to list of Anagrams

    for i in strs:
        count = [0] * 26 # a ... z

        for c in i:
            count[ord(c) - ord('a')] += 1
        res[tuple(count)].append(i)
    return res.values() # [["eat","tea","ate"],["tan","nat"],["bat"]]

groupAnagrams(["eat","tea","tan","ate","nat","bat"])

# solution without import

def groupAnagrams(strs):
    res = {}
    for i in strs:
        sortedWord = tuple(sorted(i)) # tuple the sorted word because dict does not allow list
        if sortedWord not in res: # if sortedWord not in res, add it
            res[sortedWord] = [i] # !listing i 
        else:
            res[sortedWord].append(i) # else append the word to sublist
    return res.values()

groupAnagrams(["eat","tea","tan","ate","nat","bat"])


# 347. Top K Frequent Elements
# my solution first
def topKFrequent(nums,k):
    dict = {}
    for i in nums:
        dict[i] = 1 + dict.get(i, 0)
    res = sorted(dict, key=dict.get)
    return res[-k:]

topKFrequent([1,1,1,2,2,3], 2)
# neetcode bucketsort
def topKFrequent(nums,k):
    count = {}
    freq = [[] for i in range(len(nums) + 1)]
    for n in nums:
        count[n] = 1 + count.get(n, 0)
    for n, c in count.items():
        freq[c].append(n)
    
    res = []
    for i in range(len(freq) - 1, 0 , -1):
        for n in freq[i]:
            res.append(n)
            if len(res) == k:
                return res

topKFrequent([1,1,1,2,2,3], 2)

# 451. Sort Characters By Frequency   populate dict, sorted the dict and reverse it
# then res the key(str) * value. Join res into str
def frequencySort(s):
    dict = {}
    for i in s:
        dict[i] = 1 + dict.get(i, 0)
    res = []
    for c in sorted(dict, key=lambda x:dict[x], reverse=True):
        res.append(c * dict[c])
    return ''.join(res)
frequencySort("tree")

# 125. Valid Palindrome
def isPalindrome(s):
    res = ''
    for i in s:
        if i.isalnum():
            res += i.lower()
    if res == res[::-1]:
        return True
    return False

isPalindrome("race a car")

# 680. Valid Palindrome II
def validatePalindrome(s):
    l = 0
    r = len(s) - 1
    while l < r:
        if s[l] != s[r]:
            no_L = s[l + 1: r + 1]
            no_R = s[l:r]
            return ( no_L == no_L[::-1] or no_R == no_R[::-1])
        l += 1
        r -= 1
    return True # for that if the string start as a palindrome
validatePalindrome("abca")

# 11. Container With Most Water
def maxArea(height): # 
    res = 0
    l = 0
    r = len(height) - 1
    while l < r:
        if height[l] > height[r]:
            area = height[r] * (r - l)
            res = max(res, area)
            r -= 1
        else:
            area = height[l] * (r - l)
            res = max(res, area)
            l += 1
    # print(res)
    return res

maxArea([1,8,6,2,5,4,8,3,7])

# 42. Trapping Rain Water find max left and max right which determines the amount of water in between
def trap(height):
    if not height: return 0
    l = 0
    r = len(height) - 1
    maxL = height[l]
    maxR = height[r]
    res = 0
    while l < r:
        if height[l] < height[r]:
            l += 1
            maxL = max(maxL, height[l])
            res += maxL - height[l]
        else:
            r -= 1
            maxR = max(maxR, height[r])
            res += maxR - height[r]

    # print(res)
    return res

trap([0,1,0,2,1,0,1,3,2,1,2,1])

# 121. Best Time to Buy and Sell Stock
def maxProfit(prices):
    l = 0
    r = 1
    res = 0
    while r < len(prices):
        if prices[l] < prices[r]:
            profit = prices[r] - prices[l]
            res = max(res, profit)
        else:
            l = r # when price[l] > prices[r], l becomes r
        r += 1
    # print(res)

maxProfit([7,1,5,3,6,4])

## !!!!!! # 309. Best Time to Buy and Sell Stock with Cooldown  dynamic programming, yet to understand
def maxProfit(prices):
    # if buy i + 1
    # if sell i + 2 because the cooldown
    
    dp= {} # key = (i, buying) val = max_profit
    
    def dfs(i, buying):
        if i >= len(prices):
            return 0
        if (i, buying) in dp:
            return dp[(i, buying)]
        
        if buying:
            buy = dfs(i + 1, not buying) - prices[i]
            cooldown = dfs(i + 1, buying)
            dp[(i, buying)] = max(buy, cooldown)
        else:
            sell = dfs(i + 2, not buying) + prices[i]
            cooldown = dfs(i + 1, buying)
            dp[(i, buying)] = max(sell, cooldown)
            
        return dp[(i, buying)]
    return dfs(0, True)


# 3. Longest Substring Without Repeating Characters # sliding window
# adding from right, deleting from left
def lengthOfLongestSubstring(s):
    charSet = set()
    l = 0
    res = 0
    for r in range(len(s)):
        while s[r] in charSet:
            charSet.remove(s[l])
            l += 1
        charSet.add(s[r])
        res = max(res, r - l + 1)
    # print(res)

lengthOfLongestSubstring("abcabcbb")


# 424. Longest Repeating Character Replacement
# two pointer sliding window
def characterReplace(s, k):
    dict = {}
    l = 0
    res = 0
    for r in range(len(s)):
        dict[s[r]] = 1 + dict.get(s[r], 0) # hash every letter in the string
        if (r - l + 1) - max(dict.values()) > k:  # validate the window
            dict[s[l]] -= 1
            l += 1
        res = max(res, r - l + 1) 
    # print(res)

characterReplace("AABABBA", 1)

# 567. Permutation in String

def checkInclusion(s1, s2):
    if len(s1) > len(s2): # for edge case
        return False
    s1Count, s2Count = [0] * 26, [0] * 26 # hash using list
    for i in range(len(s1)):  # populate the two list base on length of s1
        s1Count[ord(s1[i]) - ord("a")] += 1
        s2Count[ord(s2[i]) - ord("a")] += 1
    matches = 0
    for i in range(26): # static 26 because only lower case english letter will appear
        matches += 1 if s1Count[i] == s2Count[i] else 0
    
    l = 0
    for r in range(len(s1), len(s2)): # set up the two pointer, starting from where it left off from the population step
        if matches == 26: # check True condition
            return True
        index = ord(s2[r]) - ord(["a"]) # get the r value in s2
        s2Count[index] += 1
        if s1Count[index] == s2Count[index]: # if the index value matches, matches go closer to 26
            matches += 1
        elif s1Count[index] + 1 == s2Count[index]: # if it doesnt match
            matches -= 1

        index = ord(s2[l]) - ord(["a"]) # get l value in s2
        s2Count[index] -= 1
        if s1Count[index] == s2Count[index]:
            matches += 1
        elif s1Count[index] + 1 == s2Count[index]: 
            matches -= 1
        l +=1

# checkInclusion("ab", "eidbaooo")

# 20. Valid Parentheses
def isValid(s):
    stack = []
    dict = {
        ")" : "(",
        "]" : "[",
        "}" : "{" 
    }
    for i in s:
        if i in dict:
            if stack and stack[-1] == dict[i]:
                stack.pop()
            else:
                return False
        else:
            stack.append(i)
    return True if not stack else False


isValid("()[]{}")

# 22. Generate Parentheses
def generateParenthesis(n):
    # only add open parenthesis if open < n
    # only add a closing parenthesis if closed < open
    # valid If open == closed == n
        
    stack = []
    res = []
    def backtrack(openN, closedN):
        if openN == closedN == n:
            res.append("".join(stack))
            return
        if openN < n:
            stack.append("(")
            backtrack(openN+1 , closedN)
            stack.pop()
            
        if closedN < openN:
            stack.append(")")
            backtrack(openN, closedN + 1)
            stack.pop()
    backtrack(0 , 0)
    print(res)
generateParenthesis(3)

# 155. Min Stack 
# Design a stack that supports push, pop, top, and retrieving the minimum element in <!important> constant time.
class MinStack:
    def __init__(self): # initiate the two stacks(lists in python)
        self.stack = []
        self.minStack = []
    def push(self, val): # push in to original stack as well as populating the minStack
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)
    def pop(self): # popping elements from the two lists
        self.stack.pop()
        self.minStack.pop()
    def top(self): # return the top element from orginal list
        return self.stack[-1]
    def getMin(self): # return the top element from minStack
        return self.minStack[-1]


# 739. Daily Temperatures
# Monotonic Stacking -- always in decreasing order  *equals are siblings level 74 73 73 72
def dailyTemperatures(temperatures):
    res = [0] * len(temperatures) # establish a list with zeros
    stack = [] 
    for i , t in enumerate(temperatures): # get both the index and temp 
        while stack and t > stack[-1][0]: # stack[-1][0] is the previous temp
            stackT, stackInd = stack.pop()
            res[stackInd] = (i - stackInd)
        stack.append([t,i]) # append in reversed order
    print(res)

dailyTemperatures([73,74,75,71,69,72,76,73])

# 84. Largest Rectangle in Histogram
# Stack O(n) <!important>
def largestRectangleArea(heights):
    maxArea = 0
    stack = []# pair value index, height
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            index, height = stack.pop()
            maxArea = max(maxArea, height * (i - index))
            start = index
        stack.append((start, h))

    for i , h in stack:
        maxArea = max(maxArea, h *(len(heights) - i))

    # print(maxArea)

largestRectangleArea([2,1,5,6,2,3])

# 704. Binary Search
def search(nums, target):
    l , r = 0, len(nums) - 1
    while l < r:
        m = l + ((r - l) //2)
        if nums[m] > target:
            r = m - 1
        elif nums[m] < target:
            l = m + 1
        else:
            return m
    return -1

search([-1,0,3,5,9,12], 9)

# 74. Search a 2D Matrix
def searchMatrix(matrix, target):
    rows, cols = len(matrix), len(matrix[0])
    top , bot = 0, rows - 1
    while top <= bot:
        mid = (top + bot) // 2
        if target < matrix[mid][0]:
            bot = mid - 1
        elif target > matrix[mid][-1]:
            top = mid + 1
        else:
            break
    row = (top + bot) // 2
    l, r = 0, cols - 1
    while l <= r:
        m = (l + r) // 2 
        if target < matrix[row][m]:
            r = mid - 1
        elif target > matrix[row][m]:
            l = mid + 1
        else:
            return
    return False

searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3)

# 875. Koko Eating Bananas
def minEatingSpeed(piles, h):
    l, r = 0, max(piles)
    res = r
    while l <= r:
        mid = (l + r) // 2 # create mid point
        hours = 0 # initialize and reset hour calculation
        for i in piles: # find out how many hours its needed to finish the piles
            hours += ((i - 1)//mid) + 1
        if hours <= h: 
            res = min(res, mid) # if less than target hours, get min, try to find the lowest value
            r = mid - 1
        else:
            l = mid + 1
    print(res)
        


minEatingSpeed([3,6,7,11], 8)

# 33. Search in Rotated Sorted Array

# neetcode 
# l , r = 0, len(nums) -1
# while l <= r:
#     mid = (l + r)//2
#     if target == nums[mid]:
#         return mid
#     if nums[l] <= nums[mid]:
#         if target > nums[mid] or target < nums[l]:
#             l = mid + 1
#         else:
#             r = mid - 1
            
#     else:
#         if target < nums[mid] or target > nums[r]:
#             r = mid - 1
#         else:
#             l = mid + 1
# return -1

# mine with dictionary and binary search the key
def search(nums, target) :
    dict = {}
    for i, v in enumerate(nums):
        dict[v] = i
    new = sorted(dict)
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r)//2
        if target == new[mid]:
            return dict[target]
        elif target < new[mid]:
            r = mid - 1
        elif target > new[mid]:
            l = mid + 1

    return -1
search([4,5,6,7,0,1,2], 0)

# 153. Find Minimum in Rotated Sorted Array
# Binary search
def findMin(nums):
    res = nums[0] # set original res as an abtrary number
    l, r = 0, len(nums) - 1
    while l <= r:
        if nums[l] < nums[r]: # if left pointer is smaller than the right pointer, 
                              # potentially the left pointer could be the solution
            res = min(res,nums[l])
            break # to ensure the second if statement does not run once the first is achieved
        mid = (l + r) // 2
        res = min(res, nums[mid]) 
        if nums[l] >= nums[mid]: # if left pointer is larger than mid, search the right portion
            l = mid + 1
        else: # else search the left portion
            r = mid - 1
    print(res)
findMin([2,1])

# 206. Reverse Linked List
def reverseList(head):
    curr = head
    prev = None
    while curr: # while curr != None
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

# 21. Merge Two Sorted Lists Linked
# def mergeTwoLists(l1,l2):
#     dum = ListNode()
#     tail = dum.next
#     while l1 and l2:
#         if l1.val < l2.val:
#             tail.next = l1
#             l1 = l1.next
#         else:
#             tail.next= l2
#             l2 = l2.next
#     if l1:
#         tail.next = l1
#     elif l2:
#         tail.next = l2
#     return dum.next

# 143. Reorder List
def reorder(head):
    slow , fast = head, head.next # slow, fast pointers. Used to determine the middle point.
    while fast and fast.next: # making sure head is valide and head.next is not None
        slow = slow.next
        fast = fast.next.next
    second = slow.next # start of the second ll
    slow.next = None # middle pointer points at None to finish the ll
    prev = None # set to reverse the second ll
    while second: # reversing
        tmp = second.next
        second.next = prev
        prev = second
        second = tmp
    first, second = head, prev   # merge two ll, prev is now head of second ll
    while second: # merging
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first = tmp1
        second = tmp2

# 226. Invert Binary Tree
def invertTree(self, root):
    if not root:
        return False
    temp = root.left
    root.left = root.right
    root.right = temp

    self.invertTree(root.left)
    self.invertTree(root.right)
    return root

# 234. Palindrome Linked List
def isPalindromLL(head):
    res = []
    while head:
        res.append(head.val)
        head = head.next
    l = 0
    r = len(res) - 1
    while l < r:
        if res[l] != res[r]:
            return False
        l+=1
        r-=1
    return True

# space O(1) in place
def isPalindromLL(head):
    slow = head
    fast = head
    while fast and fast.next: # ensure fast.next.next dont run out
        fast = fast.next.next 
        slow = slow.next # middle point
        
    # reversing
    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt
        
    # check if palindrome
    l = head
    r = prev
    while r:
        if l.val != r.val:
            return False
        l = l.next
        r = r.next
    return True


# 2 Add Two Numbers LL
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
def addTwoNumbers(l1, l2):
    ls1 = []
    ls2 = []
    while l1:
        ls1.append(l1.val)
        l1 = l1.next
    while l2:
        ls2.append(l2.val)
        l2 = l2.next
    
    l1n = int("".join(map(str, ls1[::-1])))
    l2n = int("".join(map(str, ls2[::-1])))
    f = l1n + l2n
    res = list(map(int, str(f)))
    
    # cur = dummy = ListNode(0)
    # for i in res[::-1]:
    #     cur.next = ListNode(i)
    #     cur = cur.next
    # return dummy.next

# better solution without using list

# def addTwoNumbers(l1, l2):
#     dummy = ListNode()
#     cur = dummy
#     carry = 0
#     while l1 or l2 or carry:
#         v1 = l1.val if l1 else 0
#         v2 = l2.val if l2 else 0
#     # new digits
#     val = v1 + v2 + carry
#     carry = val // 10
#     val = val % 10
#     cur.next = ListNode(val)

#     # update pointer
#     cur = cur.next
#     l1 = l1.next if l1 else None
#     l2 = l2.next if l2 else None

#     return dummy.next

# 104. Maximum Depth of Binary Tree
# Recursion
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left),maxDepth(root.right))

# 78. Subsets
def subset(nums):
    res = []
    subset = []
    def dfs(i):
        if i >= len(nums):
            res.append(subset.copy())
            return
        
        subset.append(nums[i])
        dfs(i + 1)

        subset.pop()
        dfs(i + 1)
    dfs(0)
    return res
    
# 28. Implement strStr()
# first solution
def strStr(haystack,needle):
    if needle == "":
        return 0
    l = 0
    r = len(needle) - 1
    while r < len(haystack):
        if haystack[l : r + 1] == needle:
            return l
        else:
            l += 1
            r += 1
    return -1

# 283. Move Zeroes
def moveZeros(nums):
    l = 0 
    for r in range(len(nums)):
        if nums[r]:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
moveZeros([0,1,0,3,12])
