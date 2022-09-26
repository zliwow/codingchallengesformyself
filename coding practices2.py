def count_bits(n):
    return sum([int(d) for d in format(n,"b")])
    # list comprehension and use of format to get binary
count_bits(10)

def is_square(n):   
    import math
    if n < 0:
        return False
    res = math.ceil(n ** 0.5) # ** 0.5 act as square root
    if res * res != n:
        return False
    else:
        return True

def alphabet_position(text):
    res = ''
    for i in text:
        temp = 0
        if i.isalpha(): 
            temp = ord(i.lower()) + 1 - ord('a')
            res = res + str(temp) + " "
    return res[:-1]

# list filtering    7
# in place
def filter_list(l):
    le = 0
    while le <= len(l) - 1:
        if isinstance(l[le], int):
            le += 1
            pass
        else:
            l.remove(l[le])
    return l

# Isograms   Reverse anagram 7
def is_isogram(string):
    dict = {}
    for i in string:
        i = i.lower()
        dict[i] = 1 + dict.get(i, 0)
    for i in dict.values():
        if i > 1:
            return False
    return True

# Duplicate Encoder 6kyu
# given a string, if appear more than once in the string, return ) else (, case insensitive
def duplicate_encode(word):
    dict = {}
    for i in word:
        i = i.lower()
        dict[i] = 1 + dict.get(i, 0)
    res = ''
    for i in word:
        i = i.lower()
        if dict[i] > 1:
            res = res + ")"
        else:
            res = res + "("
    return res
    
# Shortest Word 7kyu
def find_short(s):
    s = s.split()
    res = len(s[0])
    for i in s:
        i = len(i)
        res = min(res,i)
    return res

# Complementary Dna 7 kyu
def DNA_strand(dna):
    res = ""
    for i in dna:
        if i == "A":
            res = res + "T"
        elif i == "T":
            res = res + "A"
        elif i == "C":
            res = res + "G"
        elif i == "G":
            res = res + "C"
        else:
            res = res + i
    return res

# Pete, the baker 5 kyu
def cakes(recipe, available):
    minimal = []
    for i in recipe:
        if i not in available:
            return 0
        else:
            temp = available[i] // recipe[i]
            minimal.append(temp)
    return min(minimal)
    
# The Hashtag Generator
def generate_hashtag(s):
    if s:
        res = "#"
        for i in s.split():
            res += i.title()
        return res if len(res) <= 140 else False
    else:
        return False

# Scramblies
def scramble(s1, s2):
    dict1 = {}
    dict2 = {}
    for i in s1:
        dict1[i] = 1 + dict1.get(i, 0)
    for j in s2:
        dict2[j] = 1 + dict2.get(j, 0)
    for k in dict2:
        if k in dict1 and dict2[k] <= dict1[k]:
            pass
        else:
            return False
    return True