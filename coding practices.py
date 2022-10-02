# things to learn  bucketsort , heap


# LC 347 M Top K Frequent Elements 

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

# 349. Intersection of Two Arrays
def intersection(nums1, nums2):
    nums1 = set(nums1)
    nums2 = set(nums2)
    res= []
    for i in nums1:
        if i in nums2:
            res.append(i)
    return res

# 2215. Find the Difference of Two Arrays
def findDifference(nums1, nums2):
    n = set(nums1)
    m = set(nums2)
    return [list(n - m), list(m - n)]

# 485. Max Consecutive Ones

def findMaxConsecutiveOnes(nums):
    res = 0
    count = 0 
    for i in nums:
        if i != 1:
            count = 0
        else:
            count += 1
            res = max(count, res)
    return res

# 1446. Consecutive Characters
def maxPower(s):
    dict = {}
    res = 0
    for i in s:
        if i not in dict:
            dict = {}
        dict[i] = 1 + dict.get(i, 0)
        res = max(res, dict[i])
    return res

# 1869. Longer Contiguous Segments of Ones than Zeros

def checkZeroOnes(s):
    one = 0
    zero = 0
    mOne = 0
    mZero = 0
    for i in s:
        if i == "1":
            one += 1
            mOne = max(one, mOne)
            zero = 0
        else:
            zero += 1
            mZero = max(zero, mZero)
            one = 0
    if mOne <= mZero:
        return False
    else:
        return True

# 496. Next Greater Element I
def nextGreaterElement(nums1, nums2):
    res = []
    for i in nums1:
        idx = nums2.index(i)
        v = 0
        for j in nums2[idx:]:
            if j > i:
                v = 1
                res.append(j)
                break
        if v == 0:
            res.append(-1)
    return res

# 205. Isomorphic Strings
def isIsomorphic(s, t):
    dict = {}
    for i in range(len(s)):
        if s[i] not in dict:
            dict[s[i]] = t[i]
        elif s[i] in dict:
            if dict[s[i]] == t[i]:
                pass
            else:
                return False
    dict2 = {}
    for i in range(len(s)):
        if t[i] not in dict2:
            dict2[t[i]] = s[i]
        elif t[i] in dict2:
            if dict2[t[i]] == s[i]:
                pass
            else:
                return False
    return True

# shorter version, same runtime
def isIsomorphic(s, t):
    mapST, mapTS = {}, {}
    
    for c1, c2 in zip(s, t):
        if ((c1 in mapST and mapST[c1] != c2) or
            c2 in mapTS and mapTS[c2] != c1):
            return False
        mapST[c1] = c2
        mapTS[c2] = c1
    return True
            
# 290. Word Pattern
def wordPattern(pattern, s):
    s = s.split()
    if len(pattern) != len(s):
        return False
    
    charToWord, wordToChar = {}, {}
    
    for c, w in zip(pattern, s):
        if c in charToWord and charToWord[c] != w:
            return False
        if w in wordToChar and wordToChar[w] != c:
            return False
        charToWord[c] = w
        wordToChar[w] = c
    return True

# 345. Reverse Vowels of a String
def reverseVowels(s):
    s = list(s)   # !important, because a string can't be switched easily
    vowels = "aeiouAEIOU"
    l = 0
    r = len(s) - 1
    while l < r:
        if l not in vowels:
            l += 1
        elif r not in vowels:
            r -= 1
        else:
            s[l] , s[r] = s[r], s[l]
            l += 1
            r -= 1
    return "".join(s)

# 383. Ransom Note
def canConstruct(ransomNote, magazine):
    for i in ransomNote:
        if i in magazine:
            magazine = magazine.replace(i,"",1)
        else: return False
    return True

# 389. Find the Difference
def findTheDifference(s, t):
    for i in set(t):
        if s.count(i) != t.count(i): return i

# 1268. Search Suggestions System
def suggestedProducts(products, searchWord):
    products.sort()
    res = []
    l ,r = 0, len(products) - 1
    for i in range(len(searchWord)):
        c = searchWord[i]

        while l <= r and (len(products[l]) <= i or products[l][i] != c):
            l += 1 # check fail conditions
        while l <= r and (len(products[r]) <= i or products[r][i] != c):
            r -= 1
        res.append([])
        remain = r - l + 1
        for j in range(min(3, remain)):
            res[-1].append(products[l + j])
    return res

# 520. Detect Capital
def detectCapitalUse(word):
    if word == word.upper():
        return True
    elif word == word.title():
        return True
    elif word == word.lower():
        return True
    else:
        return False

# 2129. Capitalize the Title
def capitalizeTitle(title):
    res = ""
    for i in title.split():
        if len(i) <= 2:
            res+= i.lower()
        else:
            res += i.title()
        res += " "
    return res[:-1]

# 709. To Lower Case
def toLowerCase(s):
    s = list(s)
    for i in range(len(s)):
        if s[i].isupper():
            s[i] = s[i].lower()
    return "".join(s)

# 551. Student Attendance Record I
def checkRecord(s):
    return False if "LLL" in s or s.count('A') >= 2 else True
                
# 83. Remove Duplicates from Sorted List    LL
def deleteDuplicates(head):
    cur = head
    while cur:
        while cur.next and cur.next.val == cur.val:
            cur.next = cur.next.next
        cur = cur.next
    return head

# 82. Remove Duplicates from Sorted List II

def deleteDuplicates(head):
    dict = {}
    while head:
        dict[head.val] = 1 + dict.get(head.val, 0)
        head = head.next
    temp = []
    for i,j in dict.items():
        if j ==1:
            temp.append(i)
    
    # dummy = cur = ListNode()
    
    # for i in temp:
    #     cur.next = ListNode(i)
    #     cur = cur.next
    # return dummy.next

# 46. Permutations !important
    def permute(self, nums):
        res = []
        
        # base case
        if (len(nums) == 1):
            return [nums[:]]
        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)
            
            for perm in perms:
                perm.append(n)
            res.extend(perms)
            nums.append(n)
            
        return res

# 989. Add to Array-Form of Integer
def addToArrayForm(num, k):
    n1 = int(''.join(str(i) for i in num))
    hold = n1 + k
    res = []
    for i in str(hold):
        res.append(i)
    return res
# 66. Plus One
def plusOne(digits):
    n = ''
    for i in digits:
        n += str(i)
    n = int(n) + 1
    res = []
    for i in str(n):
        res.append(int(i))
    return res

# 442. Find All Duplicates in an Array  !memory o(1)
def findDuplicates(nums):
    res = []
    for n in nums:
        m = abs(n)
        if nums[m -1] < 0 :
            res.append(m)
        else:
            nums[m -1] *= -1
    return res

# 41. First Missing Positive
def firstMissingPositive(nums):
    # dict = {}
    # for i in nums:
    #     dict[i] = 1 + dict.get(i, 0)
    # n = 1
    # for j in range(len(nums)):
    #     if n in dict:
    #         n += 1
    #     else:
    #         return n
    # return len(nums) + 1
    # o(n)
    
    for i in range(len(nums)):
        if nums[i] < 0:
            nums[i] = 0
    for i in range(len(nums)):
        val = abs(nums[i])
        if 1 <= val <= len(nums):
            if nums[val -1] > 0:
                nums[val - 1] *= -1
            elif nums[val -1] == 0:
                nums[val - 1] = -1 * (len(nums) + 1)
    for i in range(1, len(nums) + 1):
        if nums[i - 1] >= 0:
            return i
    return len(nums) + 1

# 204. Count Primes Use Sieve of Eratosthenes.
def countPrimes(n):
    if n <= 1:
        return 0
    
    nums = [None] * n
    nums[0] = nums[1] = False
    for i in range(n):
        if nums[i] == None:
            nums[i] = True
            
            for j in range(i * i, n, i):
                nums[j] = False
    return sum(nums)


# 503. Next Greater Element II
def nextGreaterElements(nums):
    n = len(nums)
    ans = [-1] * n
    stack = []
    
    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            ans[stack.pop()] = nums[i]
        stack.append(i)
        
    # handle the next greater element on the left side
    
    for i in range(n):
        if i == stack[-1]:
            break
        while stack and nums[i] > nums[stack[-1]]:
            ans[stack.pop()] = nums[i]
    return ans
        
# 141. Linked List Cycle
# store id(head) in dictionary
def hasCycle(head):
        dict = {}
        while head:
            if id(head) in dict:
                return True
            else:
                dict[id(head)] = 1
            head = head.next
        return False
# Floyd tortoise and hare
def hasCycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
        
    return False

#142. Linked List Cycle II  tortoise and hare
def detectCycle(head):
    if not head: return None # nocycle
    
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        if fast == slow: break
    if not fast.next or not fast.next.next: return None # nocycle
    
    slow2 = head
    
    while slow.next:
        if slow == slow2: return slow
        slow = slow.next
        slow2 = slow2.next

# 5. Longest Palindromic Substring
def longestPalindrome(s):
    res = ''
    resLen = 0
    
    for i in range(len(s)):
        # odd length
        l, r = i,i
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > resLen:
                res = s[l:r+1]
                resLen = r-l+1
            l -= 1
            r += 1
            
        # even length
        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > resLen:
                res = s[l:r+1]
                resLen = r-l+1
            l -= 1
            r += 1
            
    return res

# 2367. Number of Arithmetic Triplets
def arithmeticTriplets(nums,diff):
    res = 0
    for i in nums:
        if i + diff in nums and i + diff *2 in nums:
            res += 1

    return res

# 2130. Maximum Twin Sum of a Linked List
def pairSum(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt
        
    l = head
    r = prev
    res = 0
    while r:
        res = max(res,(l.val + r.val))
        l = l.next
        r = r.next
    return res
    

# 328. Odd Even Linked List

def oddEvenList(head):
    if not head or not head.next or not head.next.next:
        return head
    first = cur =  head
    second = secondhead = head.next
    i = 1
    while cur:
        if i > 2 and i % 2 != 0:
            first.next = cur
            first = first.next
        elif i > 2 and i % 2 == 0:
            second.next = cur
            second = second.next
        cur = cur.next
        i+= 1
        
    second.next = None # last even node needs to point at nothing
    first.next = secondhead
    return head
        
# 1502. Can Make Arithmetic Progression From Sequence
def canMakeArithmeticProgression(arr):
    
    arr = sorted(arr)
    vertify = arr[1] - arr[0]
    l = 0
    r = 1
    while r < len(arr):
        if arr[l] + vertify != arr[r]:
            return False
        l += 1
        r += 1
    return True

# 2095. Delete the Middle Node of a Linked List
def deleteMiddle(head):
    if not head.next:
        return head.next
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        if not fast or not fast.next:
            slow.next = slow.next.next
        else:
            slow = slow.next
    return head

# 203. Remove Linked List Elements
def removeElements(head, val):
    prev , cur = None, head
    while cur:
        if cur.val == val:
            if prev:
                prev.next = cur.next
            else:
                head = cur.next
            cur = cur.next
        else:
            prev = cur
            cur = cur.next
    return head


# 237. Delete Node in a Linked List # need help understand
def deleteNode(self, node):
    del_node = node.next
    node.val = del_node.val
    node.next = del_node.next
    del del_node
    return

# 19. Remove Nth Node From End of List
def removeNthFromEnd(head,n,ListNode): # for yellow marker only, listnode doesnt not exist in the original parameter
    dummy = ListNode(0, head)
    left = dummy
    right = head
    
    while n > 0 and right:
        right = right.next
        n -= 1
        
    while right:
        left = left.next
        right = right.next
    # delete
    left.next = left.next.next
    return dummy.next

# 1721. Swapping Nodes in a Linked List

def swapNodes(head,k):
    slow, fast = head, head
    
    # mark first and move fast together, leave slow behind for targeting the second node
    for _ in range(k -1):
        fast = fast.next
    first = fast
    
    # move slow and mark it as the second target node, move fast to the end
    while fast.next:
        fast = fast.next
        slow = slow.next
        
    # swap val
    slow.val, first.val = first.val, slow.val
    
    return head

# 24. Swap Nodes in Pairs
def swapPairs(head):
    if not head or not head.next:
        return head
    
    
    l , r = head, head.next
    while r:
        l.val, r.val = r.val, l.val
        if r.next:
            l = r.next
            r = l.next
        else:
            break
    return head
            
# 287. Find the Duplicate Number
# think of it as linked list cycle
def findDuplicate(nums):
    # floyd's tortoise and hare
    # phrase 1, finding intersection
    slow , fast = 0 , 0 # start at 0(head) because 0 is never part of the cycle
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # phrase 2, finding slow2(from start) intersect with slow
    slow2 = 0
    while True:
        slow = nums[slow]
        slow2 = nums[slow2]
        if slow == slow2:
            return slow

# 645. Set Mismatch
def findErrorNums(nums):
    n = len(nums) # number of elements in nums
    a = sum(range(1,n +1))# the sum of the correct sequence 1+2+3+4
    b = sum(nums) # the sum of sequence with duplicate 1+2+2+4
    c = sum(set(nums)) # the sum of sequence without duplicate 1+2+4
    missing = a - c # find what is missing # 3
    duplicate = b - c # find what is duplicate # 2
    return [duplicate, missing]

# 189. Rotate Array
def rotate(nums, k):
    """
    Do not return anything, modify nums in-place instead.
    """
    k = k % len(nums) # making sure k < len(nums)
    l , r = 0, len(nums) - 1 # two pointer start and end
    while l < r: # swap values of entire list
        nums[l] , nums[r] = nums[r], nums[l]
        l+= 1
        r-= 1
    l, r = 0, k - 1 # reset the pointers to start and k
    while l < r: # swap values of the first section
        nums[l] , nums[r] = nums[r], nums[l]
        l+= 1
        r-= 1
    l, r = k, len(nums) - 1 # reset the pointers to k and end
    while l < r: # swap values of the second half
        nums[l] , nums[r] = nums[r], nums[l]
        l+= 1
        r-= 1

# 1295. Find Numbers with Even Number of Digits
def findNumbers(nums):
    res = 0
    for i in nums:
        if len(str(i)) % 2 == 0:
            res += 1
    return res

# 977. Squares of a Sorted Array
# two pointer, put int from the largest number
def sortedSquares(nums):
    l = 0
    r = len(nums) - 1
    res = []
    while l <= r:
        if nums[l] * nums[l] > nums[r] * nums[r]:
            res.append(nums[l] * nums[l])
            l += 1
        else:
            res.append(nums[r] * nums[r])
            r -= 1

    return res[::-1]

# 1089. Duplicate Zeros
# in-place
def duplicateZeros(arr):
    """
    Do not return anything, modify arr in-place instead.
    """
    
    c = 0
    while c < len(arr):
        if arr[c] == 0:
            arr.insert(c, 0)
            arr.pop()
            c += 2
        else:
            c += 1

# 543. Diameter of Binary Tree
# O(n2)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def depth(root:TreeNode):
            if root == None:
                return 0
            left = depth(root.left)
            right = depth(root.right)
            return 1 + max(left, right)
        def FindDiameter(root:TreeNode):
            if root == None:
                return 0
            height_left = depth(root.left)
            height_right = depth(root.right)
            case1 = height_left + height_right # diameter may pass through the root
            case2 = FindDiameter(root.left)# in case of left
            case3 = FindDiameter(root.right)# in case of right
            return max(case1, case2, case3)
        return FindDiameter(root)
# O(1) regular dfs
def diameterOfBinaryTree(self, root: TreeNode) -> int:
    res = [0]
    
    def dfs(root):
        if not root:
            return -1
        left = dfs(root.left)
        right = dfs(root.right)
        
        res[0] = max(res[0], 2 + left + right)
        
        return 1 + max(left, right)
                        
    dfs(root)
    return res[0]

# 110. Balanced Binary Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        
        def dfs(root): # return bool and height
            if not root:
                return [True, 0]
            
            left, right = dfs(root.left), dfs(root.right)
            balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1
            # only true if left and right are both true
            
            return [balanced, 1 + max(left[1], right[1])]
        
        return dfs(root)[0]

# 100. Same Tree
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    if not p and not q: # empty trees are equal
        return True
    if not p or not q or p.val != q.val: # if one of them is none or value not the same
        return False
    
    return (self.isSameTree(p.left, q.left) and
        self.isSameTree(p.right, q.right))
    
# 450. Delete Node in a BST
# O(h) height of tree
# traverse the tree finding matching key. Then in case 4, get the minimal val of last left key and replace the matching key with it. Delete the last left node afterward.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def deleteNode(self, root: TreeNode, key: int) :
        if not root:
            return
        if root.val == key:
            # 4 cases 
            if not root.left and not root.right: return None
            if not root.left and root.right: return root.right
            if root.left and not root.right: return root.left
            # if both 
            pnt = root.right
            while pnt.left: pnt = pnt.left
            root.val = pnt.val
            root.right = self.deleteNode(root.right, root.val)
            
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
            
        return root
                
# 108. Convert Sorted Array to Binary Search Tree
# make a empty Treenode, insert value using recursive method
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sortedArrayToBST(self, nums):
        def helper(l, r):
            if l > r:
                return None
            m = (l + r) // 2
            
            root = TreeNode(nums[m])
            root.left = helper(l, m- 1)
            root.right = helper(m + 1, r)
            return root
        return helper(0, len(nums) -1)

# 26. Remove Duplicates from Sorted Array
# keep a left pointer and use right pointer to find unique values then swap
def removeDuplicates(nums):
    l = 1
    for r in range(1, len(nums)):
        if nums[r] != nums[r - 1]:
            nums[l] = nums[r]
            l += 1
    return l

# 1346. Check If N and Its Double Exist
# acompany the conditions of the question
def checkIfExist(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i == j:
                pass
            elif arr[i] == arr[j] * 2:
                return True
    return False

# 941. Valid Mountain Array
# check from left then check from right 
def validMountainArray(arr):
    if len(arr) <= 2:
        return False
    l = 0
    r = len(arr) - 1
    while l + 1 < len(arr) - 1 and arr[l] < arr[l + 1]:
        l += 1
    while r - 1 > 0 and arr[r] < arr[r - 1]:
        r -= 1
    return l == r

# 1299. Replace Elements with Greatest Element on Right Side
# in-place practice
def replaceElements(arr):
    r_max = -1
    for i in range(len(arr) -1, -1, -1):
        tmp = arr[i]
        arr[i] = r_max # first loop goes straight to -1, the rest will update 
        if tmp > r_max: # if fund a new max
            r_max = tmp
    return arr

# 905. Sort Array By Parity
# in place
def sortArrayByParity(nums):
    l = 0
    r = len(nums) - 1
    while l < r:
        if nums[l] % 2 != 0:
            nums[l], nums[r] = nums[r], nums[l]
            r -= 1
        else:
            l += 1   
    return nums

# 2164. Sort Even and Odd Indices Independently
def sortEvenOdd(nums):
    l = 0
    r = 1
    even = nums[l::2]
    odd = nums[r::2]
    even.sort()
    nums[l::2]= even
    odd.sort()
    nums[r::2] = odd[::-1]
    return nums

# 922. Sort Array By Parity II
def sortArrayByParityII(nums):
    even = 0
    odd = 1
    while even < len(nums) and odd < len(nums):
        if nums[even] % 2 == 0:  # untill even index has a odd value
            even += 2
        else:
            if nums[odd] % 2 != 0: # untill odd index has a even value
                odd += 2
            else:
                nums[even], nums[odd] = nums[odd], nums[even]
                even += 2
                odd += 2
    return nums

# 1051. Height Checker
def heightChecker(heights):
    expect = sorted(heights)
    res = 0
    for c1, c2 in zip(expect, heights):
        if c1 != c2:
            res += 1
    return res

# 414. Third Maximum Number
def thirdMax(nums):
    if nums:
        nums = set(nums)
        nums = sorted(nums)
        if len(nums) >= 3:
            return nums[-3]
        else:
            return nums[-1]

# 724. Find Pivot Index
# pivot index is a index where the sum of before and after are equal ex. 1 1 2 1 1  pvt is at index 2
def pivotIndex(nums):
    total = sum(nums)
    leftSum = 0
    for i in range(len(nums)):
        rightSum = total - nums[i] - leftSum
        if leftSum == rightSum:
            return i
        leftSum += nums[i]
    return -1

# 747. Largest Number At Least Twice of Others
def dominantIndex(nums):
    og = nums
    nums = sorted(nums)
    res = 0
    if nums[-2] + nums[-2] <= nums[-1]:
        res = og.index(nums[-1])
        return res
    return -1