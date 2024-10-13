#Say "Hello, World!" With Python#

if __name__ == '__main__':
    print("Hello, World!")

#Python If-Else#

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())

    if n % 2 != 0:
        print("Weird")
    else:
        if n >= 2 and n <= 5:
            print("Not Weird")
        elif n >= 6 and n <= 20:
            print("Weird")
        else:
            print("Not Weird")

#Arithmetic Operators#

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    c=a+b
    d=a-b
    e=a*b
    print("%d\n%d\n%d"%(c,d,e))

#Python: Division#

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    c=a//b
    d=a/b
    print("%d\n%f"%(c,d))

#Loops#

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)

#Write a function#

def is_leap(year):
    leap = False
    
    # Write your logic here
    if year%4==0:
        if year%100==0:
            if year%400==0:
                leap=True
            else:
                leap=False
        else:
             leap=True
    else:
        leap=False

    return leap


#Print Function#

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i+1,end="")


#List Comprehensions#

x = int(input())
y = int(input())
z = int(input())
n = int(input())
output=[]
for i in range(x+1):
for j in range(y+1):
            for k in range(z+1):
                if i+j+k==n:
                    continue
                else:
                    output.append([i,j,k])
    
    print(output)


#Find the Runner-Up Score!#

n = int(input())
arr = map(int, input().split())
arr=sorted(arr,reverse=True)
for i in range(len(arr)):
    if arr[i]==arr[0]:
        continue
    else:
        print(arr[i])  
        break


#Nested Lists#

records = []
for _ in range(int(input())):
        name = input()
        score = float(input())
        records.append([name, score])
    
scores = sorted({score for name, score in records})

second_lowest_score = scores[1]

second_lowest_students = sorted([name for name, score in records if score == second_lowest_score])

for student in second_lowest_students:
        print(student)



#Finding the percentage#

n = int(input())
student_marks = {}
    
for _ in range(n):
        line = input().split()
        name = line[0]
        scores = list(map(float, line[1:]))
        student_marks[name] = scores
    
query_name = input()
    
marks = sum(student_marks[query_name]) 
avg = marks / len(student_marks[query_name])
    
print(f"{avg:.2f}")


#Lists#

n = int(input())
lst = []
for _ in range(n):
        cmd = input().split()
        if cmd[0] == 'insert':
            lst.insert(int(cmd[1]), int(cmd[2]))
        elif cmd[0] == 'print':
            print(lst)
        elif cmd[0] == 'remove':
            lst.remove(int(cmd[1]))
        elif cmd[0] == 'append':
            lst.append(int(cmd[1]))
        elif cmd[0] == 'sort':
            lst.sort()
        elif cmd[0] == 'pop':
            lst.pop()
        elif cmd[0] == 'reverse':
            lst.reverse()
                       
#Tuples#

n = int(input())
integer_list = map(int, input().split())
integer_list=tuple((integer_list))
    
print(hash(integer_list))

#sWAP cASE#

def swap_case(s):
    return s.swapcase()


#String Split and Join#

def split_and_join(line):
    return "-".join(line.split(" "))

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


#What's Your Name#

def print_full_name(first, last):
    generic = "Hello firstname lastname! You just delved into python."
    output = generic.replace("firstname", first).replace("lastname", last)
    print(output)

#Mutations#

def mutate_string(string, position, character):
    return string[:position] + character + string[(position+1):]


#Find a string#

def count_substring(string, sub_string):
    count = 0
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count


#String Validators#

s = input()
print(any(c.isalnum() for c in s))
print(any(c.isalpha() for c in s))
print(any(c.isdigit() for c in s))
print(any(c.islower() for c in s))
print(any(c.isupper() for c in s))


#Text Alignment#

thickness = int(input())
c = 'H'

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


#Text Wrap#

import textwrap

def wrap(string, max_width):
    return "\n".join(textwrap.wrap(string, max_width))

#Designer Door Mat#

N, M = map(int, input().split())

for i in range(1, N, 2): 
    print((i * ".|.").center(M, "-"))
    
print("WELCOME".center(M, "-"))

for i in range(N-2, -1, -2): 
    print((i * ".|.").center(M, "-"))


#String Formatting#

def print_formatted(number):
    width = len(bin(number)) - 2
    for i in range(1, number + 1):
        print(f"{i:{width}d} {i:{width}o} {i:{width}X} {i:{width}b}")

#Alphabet Rangoli#

def print_rangoli(size):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    lines = []
    for i in range(size):
        s = '-'.join(alphabet[size-1:i:-1] + alphabet[i:size])
        lines.append(s.center(4*size-3, '-'))
    print('\n'.join(lines[::-1] + lines[1:]))

#Capitalize!#

def solve(s):
    l = s.split(" ")

    for i in range(len(l)):
        l[i] = l[i].capitalize()

    return ' '.join(l)

#The Minion Game#

def minion_game(string):
    vowels = 'AEIOU'
    kev_score = 0
    stu_score = 0
    length = len(string)
    for i in range(length):
        if string[i] in vowels:
            kev_score += length - i
        else:
            stu_score += length - i
    if kev_score > stu_score:
        print("Kevin", kev_score)
    elif stu_score > kev_score:
        print("Stuart", stu_score)
    else:
        print("Draw")


#Merge the Tools!#

def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        print(''.join(dict.fromkeys(string[i:i+k])))


#Introduction to Sets#

def average(array):
    return round(sum(set(array)) / len(set(array)), 3)


#Symmetric Difference#

m = int(input())
a = set(map(int, input().split()))
n = int(input())
b = set(map(int, input().split()))

sym_diff = a.symmetric_difference(b)

for num in sorted(sym_diff):
    print(num)

#No Idea!#

n, m = map(int, input().split())
arr = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))

happiness = sum((i in A) - (i in B) for i in arr)
print(happiness)

#Set .add()#

num_stamps = int(input())
stamps = set()

for _ in range(num_stamps):
    stamps.add(input())

print(len(stamps))

#Set .discard(), .remove() & .pop()#

n = int(input())
s = set(map(int, input().split()))

m = int(input())

for _ in range(m):
    command = input().split()
    
    if command[0] == "pop":
        s.pop()
    elif command[0] == "remove":
        s.remove(int(command[1]))
    elif command[0] == "discard":
        s.discard(int(command[1]))

print(sum(s))

#Set .union() Operation#

n = int(input())
english = set(map(int, input().split()))

m = int(input())
french = set(map(int, input().split()))

print(len(english.union(french)))

#Set .intersection() Operation#

n = int(input())
english_subscribers = set(map(int, input().split()))
m = int(input())
french_subscribers = set(map(int, input().split()))

both_subscribed = english_subscribers.intersection(french_subscribers)
print(len(both_subscribed))

#Set .difference() Operation#

n_english = int(input())
english_set = set(map(int, input().split()))
n_french = int(input())
french_set = set(map(int, input().split()))

only_english = english_set - french_set

print(len(only_english))

#Set .symmetric_difference() Operation#

n_eng = int(input())
eng_subs = set(map(int, input().split()))

n_french = int(input())
french_subs = set(map(int, input().split()))

print(len(eng_subs.symmetric_difference(french_subs)))

print(len(s_eng.symmetric_difference(s_fr)))

#Set Mutations#

n = int(input())
A = set(map(int, input().split()))

num_operations = int(input())
for _ in range(num_operations):
    operation, _ = input().split()
    other_set = set(map(int, input().split()))
    getattr(A, operation)(other_set)

print(sum(A))

#The Captain's Room#

k = int(input())
room_list = list(map(int, input().split()))

captains_room = (sum(set(room_list)) * k - sum(room_list)) // (k - 1)
print(captains_room)

#Check Subset#

t = int(input())
for _ in range(t):
    a_len = int(input())
    a = set(map(int, input().split()))
    b_len = int(input())
    b = set(map(int, input().split()))
    print(a.issubset(b))

#Check Strict Superset#

A = set(input().split())
n = int(input())
result = all(A > set(input().split()) for _ in range(n))
print(result)

#Collections.Counter()#

from collections import Counter

num_shoes = int(input())
shoe_sizes = Counter(map(int, input().split()))
num_customers = int(input())

total_money = 0

for _ in range(num_customers):
    size, price = map(int, input().split())
    if shoe_sizes[size] > 0:
        total_money += price
        shoe_sizes[size] -= 1

print(total_money)

#DefaultDict Tutorial#

from collections import defaultdict

n, m = map(int, input().split())

group_A = defaultdict(list)

for i in range(1, n + 1):
    word = input().strip()
    group_A[word].append(i)

for _ in range(m):
    word = input().strip()
    if word in group_A:
        print(' '.join(map(str, group_A[word])))
    else:
        print(-1)

#Collections.namedtuple()#

from collections import namedtuple

n = int(input())
columns = input().split()
Students = namedtuple('Students', columns)
marks = [int(Students(*input().split()).MARKS) for _ in range(n)]
print(f"{sum(marks) / n:.2f}")

#Collections.OrderedDict()#

from collections import OrderedDict

item_price = OrderedDict()

num_items = int(input())

for _ in range(num_items):
    *item_name, price = input().split()
    item_name = " ".join(item_name)
    price = int(price)
    
    if item_name in item_price:
        item_price[item_name] += price
    else:
        item_price[item_name] = price

for item_name, net_price in item_price.items():
    print(item_name, net_price)

#Word Order#

from collections import OrderedDict

n = int(input())
wordcount = OrderedDict()

for _ in range(n):
    word = input().strip()
    if word in wordcount:
        wordcount[word] += 1
    else:
        wordcount[word] = 1

print(len(wordcount))
print(" ".join(map(str, wordcount.values())))

#Collections.deque()#

from collections import deque

d = deque()
n = int(input())

for _ in range(n):
    command = input().split()
    if command[0] == 'append':
        d.append(command[1])
    elif command[0] == 'appendleft':
        d.appendleft(command[1])
    elif command[0] == 'pop':
        d.pop()
    elif command[0] == 'popleft':
        d.popleft()

print(" ".join(d))

#Company Logo#

from collections import Counter

company_name = input()

char_count = Counter(company_name)

sorted_chars = sorted(char_count.items(), key=lambda x: (-x[1], x[0]))

for char, count in sorted_chars[:3]:
    print(char, count)

#Piling Up!#

from collections import deque

test_cases = int(input())

for _ in range(test_cases):
    num_cubes = int(input())
    side_lengths = deque(map(int, input().split()))

    possible = True
    last_picked = max(side_lengths[0], side_lengths[-1])

    while side_lengths:
        if side_lengths[0] > side_lengths[-1]:
            current = side_lengths.popleft()
        else:
            current = side_lengths.pop()

        if current > last_picked:
            possible = False
            break

        last_picked = current

    print("Yes" if possible else "No")

#Calendar Module#

import calendar

month, day, year = map(int, input().split())

day_of_week = calendar.weekday(year, month, day)

days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
print(days[day_of_week])

#Time Delta#

import math
import os
import random
import re
import sys

from datetime import datetime

def time_delta(t1, t2):
    time_format = "%a %d %b %Y %H:%M:%S %z"
    
    time1 = datetime.strptime(t1, time_format)
    time2 = datetime.strptime(t2, time_format)
    
    delta = abs(int((time1 - time2).total_seconds()))
    
    return str(delta)

if __name__ == '__main__':
    t = int(input())
    
    for t_itr in range(t):
        t1 = input().strip()
        t2 = input().strip()
        
        result = time_delta(t1, t2)
        print(result)

#Exceptions#

t = int(input())

for _ in range(t):
    try:
        a, b = input().split()
        print(int(a) // int(b))
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)

#Zipped!#

students, subjects = input().split()

students = int(students)
subjects = int(subjects)

scores = [] 

for _ in range(subjects):
    scores.append(list(map(float, input().split())))

scores_tot = zip(*scores)

for student_scores in scores_tot:
    print(f"{sum(student_scores)/subjects:.1f}")

#Athlete Sort#

import math
import os
import random
import re
import sys

nm = input().split()
n = int(nm[0])
m = int(nm[1])
arr = []

for _ in range(n):
    arr.append(list(map(int, input().rstrip().split())))

k = int(input())

arr.sort(key=lambda x: x[k])

for i in arr:
    print(*i)

#ginortS#

def custom_sort(s):
    lower = sorted([c for c in s if c.islower()])
    upper = sorted([c for c in s if c.isupper()])
    odddigits = sorted([c for c in s if c.isdigit() and int(c) % 2 != 0])
    evendigits = sorted([c for c in s if c.isdigit() and int(c) % 2 == 0])
    return ''.join(lower + upper + odddigits + evendigits)

s = input()
print(custom_sort(s))

#Map and Lambda Function#

cube = lambda x: x ** 3

def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]

#Detect Floating Point Number#

import re

def is_floating_point(num):
    pattern = re.compile(r"^[+-]?\d*\.\d+$")

    return bool(re.match(pattern, num))

n = int(input())
for _ in range(n):
    test_str = input().strip()
    print(is_floating_point(test_str))

#Re.split()#

regex_pattern = r"\,|\."# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

#Group(), Groups() & Groupdict()#

import re

s = input()
m = re.search(r'([a-zA-Z0-9])\1', s)

if m:
    print(m.group(1))
else:
    print(-1)

#Re.findall() & Re.finditer()#

import re

s = input()

pattern = r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])([aeiouAEIOU]{2,})(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])'

matches = re.findall(pattern, s)

if matches:
    for match in matches:
        print(match)
else:
    print(-1)

#Re.start() & Re.end()#

import re

s = input()
k = input()

matches = list(re.finditer(r'(?={})'.format(k), s))

if matches:
    for match in matches:
        print((match.start(), match.start() + len(k) - 1))
else:
    print((-1, -1))

#Regex Substitution#

import re

def replace_logical_operators(text):

    text = re.sub(r'(?<= )&&(?= )', 'and', text)

    text = re.sub(r'(?<= )\|\|(?= )', 'or', text)
    return text

n = int(input())

for _ in range(n):
    print(replace_logical_operators(input()))

#Validating Roman Numerals#

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'.

#Validating phone numbers#

import re

n = int(input())

for _ in range(n):
    phone_number = input().strip()
    if re.fullmatch(r'[789]\d{9}', phone_number):
        print("YES")
    else:
        print("NO")

#Validating and Parsing Email Addresses#

import re
import email.utils

email_pattern = r'^[a-zA-Z][\w\.-]+@[a-zA-Z]+\.[a-zA-Z]{1,3}$'

n = int(input())

for _ in range(n):
    parsed_email = email.utils.parseaddr(input())
    if re.match(email_pattern, parsed_email[1]):
        print(email.utils.formataddr(parsed_email))

#Hex Color Code#

import re

n = int(input())

hex_color_pattern = re.compile(r'(?<!^)(#(?:[a-fA-F0-9]{3}|[a-fA-F0-9]{6}))(?:[^\w]|$)')

for _ in range(n):
    s = input()
    matches = hex_color_pattern.findall(s)
    for match in matches:
        print(match)

#HTML Parser - Part 1#

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] else 'None'}")
    
    def handle_endtag(self, tag):
        print(f"End   : {tag}")
    
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] else 'None'}")

n = int(input())
html_data = ""

for _ in range(n):
    html_data += input().strip()

parser = MyHTMLParser()
parser.feed(html_data)

#HTML Parser - Part 2#

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if "\n" in data:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values#

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

    def handle_startendtag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")


html = ""

for _ in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Validating UID#

import re

def is_valid_uid(uid):

    if len(uid) != 10:
        return False

    if len([char for char in uid if char.isupper()]) < 2:
        return False

    if len([char for char in uid if char.isdigit()]) < 3:
        return False

    if not uid.isalnum():
        return False

    if len(set(uid)) != 10:
        return False

    return True

test = int(input().strip())
for _ in range(test):
    employee_uid = input().strip()
    if is_valid_uid(employee_uid):
        print("Valid")
    else:
        print("Invalid")

#Validating Credit Card Numbers#

import re

n = int(input())

for _ in range(n):
    s = input().strip()

    if re.match(r"^[456]\d{3}(-?\d{4}){3}$", s) and not re.search(r"(\d)\1{3,}", s.replace("-", "")):
        print("Valid")
    else:
        print("Invalid")

#Validating Postal Codes#

regex_integer_in_range = r"^[1-9][\d]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	# Do not delete 'r'.

#Matrix Script#

#!/bin/python3

import re

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])

matrix = []
for i in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

decoded_script = []
for col in range(m):
    for row in range(n):
        decoded_script.append(matrix[row][col])

decoded_string = ''.join(decoded_script)

cleaned_string = re.sub(r'(?<=\w)([^\w]+)(?=\w)', ' ', decoded_string)

print(cleaned_string)

#XML 1 - Find the Score#

def get_attr_number(node):
    attr_num = 0
    for i in node.iter():
        attr_num += len(i.attrib)
    return attr_num

#XML2 - Find the Maximum Depth#

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for i in elem:
        depth(i, level)

#Standardize Mobile Number Using Decorators#

def wrapper(f):
    def fun(l):
        standardized = []
        for number in l:
            number = number[-10:]
            standardized.append(f"+91 {number[:5]} {number[5:]}")
        return f(standardized)
    return fun

#Decorators 2 - Name Directory#

def person_lister(f):
    def inner(people):
        return [f(person) for person in sorted(people, key=lambda x: int(x[2]))]
    return inner

#Arrays#

def arrays(arr):
    # complete this function
    # use numpy.array
    arr.reverse()
    numpy_arr = numpy.array(arr,float)
    return numpy_arr

#Shape and Reshape#

import numpy as np

arr = list(map(int, input().split()))

reshaped_arr = np.array(arr).reshape(3, 3)

print(reshaped_arr)

#Transpose and Flatten#

import numpy as np

n, m = map(int, input().split())
array = np.array([input().split() for _ in range(n)], int)

print(np.transpose(array))
print(array.flatten())

#Concatenate#

import numpy as np

n, m, p = map(int, input().split())

array_1 = np.array([input().split() for _ in range(n)], int)

array_2 = np.array([input().split() for _ in range(m)], int)

result = np.concatenate((array_1, array_2), axis=0)

print(result)

#Zeros and Ones#

import numpy as np
dimensions = tuple(map(int, input().split()))

zeros_array = np.zeros(dimensions, dtype=int)
ones_array = np.ones(dimensions, dtype=int)

print(zeros_array)
print(ones_array)

#Eye and Identity#

import numpy as np
N, M = map(int, input().split())
matrix = np.eye(N, M)
print(str(matrix).replace('1', ' 1').replace('0', ' 0'))

#Array Mathematics#

import numpy as np

n, m = map(int, input().split())

A = np.array([list(map(int, input().split())) for _ in range(n)])
B = np.array([list(map(int, input().split())) for _ in range(n)])

print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)

#Floor, Ceil and Rint#

import numpy
numpy.set_printoptions(legacy='1.13')

A=numpy.array(list(map(float,input().split())))

print(numpy.floor(A),numpy.ceil(A),numpy.rint(A),sep='\n')

#Sum and Prod#

import numpy as np

n, m = map(int, input().split())
array = np.array([input().split() for _ in range(n)], int)
result = np.prod(np.sum(array, axis=0))
print(result)

#Min and Max#

import numpy

n, m = map(int, input().split())
array = []

for _ in range(n):
    array.append(list(map(int, input().split())))

arr = numpy.array(array)
print(numpy.max(numpy.min(array, axis = 1)))

#Mean, Var, and Std#

import numpy

n, m = map(int, input().split())
arr = []
for _ in range(n):
    arr.append(list(map(int, input().split())))

arr = numpy.array(arr)

print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
print(round(numpy.std(arr), 11))

#Dot and Cross#

import numpy as np

n = int(input())
matrix_a = np.array([list(map(int, input().split())) for _ in range(n)])
matrix_b = np.array([list(map(int, input().split())) for _ in range(n)])

result = np.dot(matrix_a, matrix_b)
print(result)

#Inner and Outer#

import numpy as np

A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))

inner_product = np.inner(A, B)
outer_product = np.outer(A, B)

print(inner_product)
print(outer_product)

#Polynomials#

import numpy as np

coefficients = list(map(float, input().split()))
x = float(input())

result = np.polyval(coefficients, x)

print(result)

#Linear Algebra#

import numpy as np

n = int(input())
matrix = [list(map(float, input().split())) for _ in range(n)]
determinant = np.linalg.det(matrix)

print(round(determinant, 2))

#Birthday Cake Candles#

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    max_height = max(candles)
    return candles.count(max_height)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps#

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    if v1 == v2:
        
        return "YES" if x1 == x2 else "NO"
    else:
        
        if (x2 - x1) % (v1 - v2) == 0 and (x1 - x2) * (v1 - v2) < 0:
            return "YES"
        else:
            return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising#

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    shared = 5
    cumulative_likes = 0
    for day in range(n):
        likes = shared // 2
        cumulative_likes += likes
        shared = likes * 3
    return cumulative_likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive Digit Sum#

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    p = sum(map(int, list(n))) * k
    while p >= 10:
        p = sum(map(int, str(p)))
    return p

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort 1#

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    value_to_insert = arr[-1]
    i = n - 2
    while i >= 0 and arr[i] > value_to_insert:
        arr[i + 1] = arr[i]
        print(' '.join(map(str, arr)))
        i -= 1
    arr[i + 1] = value_to_insert
    print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort 2#

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)