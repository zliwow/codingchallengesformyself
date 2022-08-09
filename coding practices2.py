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