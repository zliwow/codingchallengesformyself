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
    