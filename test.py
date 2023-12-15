print(pow(2,3)-1)

# 1
def solution(x, y):
    for num in x:
        if num not in y:
            return num
    for num in y:
        if num not in x:
            return num
    return -1

# 2 ion flux relabeling
def find_parent_node(h, node, max_root):
    if node >= max_root:
        return -1
    # right     
    if max_root-1 == node:
        return max_root
    #left    
    if max_root - pow(2,h-1) == node:
        return max_root
        
    while h != 2:
        if node < max_root - pow(2,h-1):
            return find_parent_node(h-1,node, max_root - pow(2,h-1))
        else:
            return find_parent_node(h-1,node, max_root - 1)
    return -1
        

def solution(h, q):
    max_root = pow(2,h)-1
    # Your code here
    return [find_parent_node(h, node, max_root) for node in q]

# 3 number station coded messages
def solution(l, t):
    # return first beginning and end indexes in l whose values add up to t
    for start in range(len(l)):
        total = 0
        for current, e in enumerate(l[start:]):
            total += e
            if total == t:
                return [start, start + current]
            if total > t:
                break
    return [-1, -1]

from fractions import Fraction
from fractions import gcd

# Function to calculate determinant
def determinant(matrix, n):
    if n == 1:
        return matrix[0][0]
    
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = Fraction(0)
    for i in range(n):
        minor = [row[:i] + row[i+1:] for row in matrix[1:]]
        cofactor = ((-1)**i) * determinant(minor, n-1)
        det += matrix[0][i] * cofactor
    return det

# Function to calculate the inverse of a matrix
def inverse(matrix):
    n = len(matrix)
    det = determinant(matrix, n)
    
    if det == 0:
        return "Inverse does not exist"
    
    # Calculate adjoint
    adjoint = []
    for i in range(n):
        adjoint_row = []
        for j in range(n):
            minor = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            cofactor = ((-1)**(i+j)) * determinant(minor, n-1)
            adjoint_row.append(cofactor)
        adjoint.append(adjoint_row)
    
    # Transpose the cofactor matrix to get adjoint
    adjoint = list(map(list, zip(*adjoint)))
    
    # Calculate inverse using the formula: 1/det(A) * adjoint(A)
    inv = [[adjoint[i][j] * Fraction(1, det) for j in range(n)] for i in range(n)]
    
    return inv

# Function to find LCM of two numbers
def lcm(a):
    # least common multiple for array
    for i, x in enumerate(a):
        lcm = x if i == 0 else lcm * x // gcd(lcm, x)
    return lcm

def solution(m):
    # Your code here
    terminal = [not any(row) for row in m]

    if terminal.count(True) == 1:
        return [1, 1]
    count_terminal = 0
    idx_terminal = []
    idx_not_terminal = []
    
    for row in m:
      sum_row = sum(row)
      if sum_row > 0:
        for i in range(len(row)):
          if row[i] > 0:
            row[i] = Fraction(row[i], sum_row)
    
    for indx, row in enumerate(m):
        if not any(row):
            idx_terminal.append(indx)
            count_terminal += 1
    
    for idx in range(len(m)):
        if idx not in idx_terminal:
            idx_not_terminal.append(idx)
            
    I = [[1 if i == j else 0 for j in range((len(idx_not_terminal)))] for i in range(len(idx_not_terminal))]
    R = [[m[i][j] for j in idx_terminal] for i in idx_not_terminal]
    Q = [[m[i][j] for j in idx_not_terminal] for i in idx_not_terminal]
    
    sub_matrix = [[I[i][j] - Q[i][j] for j in range(len(Q[0]))] for i in range(len(I))]
    F = inverse(sub_matrix)
    
    FR_res = [[sum(a * b for a, b in zip(F_row, R_col)) for R_col in zip(*R)] for F_row in F]
    
    # Extract numerators and common denominator
    lcm_denominator = lcm([x.denominator for x in FR_res[0]])
    
    numerators = [int(frac * lcm_denominator) for frac in FR_res[0]]
    numerators.append(lcm_denominator)
    
    return numerators
    