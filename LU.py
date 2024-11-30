import numpy as np

def read_in(infile):
    with open(infile, 'r') as file:
        lines = file.readlines()
        # get n
        n = int(lines[0].strip())
        # get a
        a = [[int(num) for num in line.split()] for line in lines[1:n + 1]]
        a = np.array(a)
        # get b
        b = int(lines[n + 1].strip())
        # get a list of the right-hand side matrices
        right_list = []
        for iter in range(b):
            right = (lines[(n + 2) + iter].split())
            rightrow = np.array(right)
            rightcol = rightrow.reshape(-1, 1)  # turn to column matrices
            right_list.append(rightcol)
    return n, a, b, right_list

def partial_pivot(matrix):
    matrix = matrix.astype(np.float64)  # ensure float
    n = len(matrix)
    P = np.eye(n)
    all_factors = []

    for i in range(n - 1):
        max_row = np.argmax(np.abs(matrix[i:, i])) + i
        if max_row != i:
            matrix[[i, max_row]] = matrix[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
        factorlist = []
        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            factorlist.append(factor)
            matrix[j, i:] -= factor * matrix[i, i:]
        sorted_factor_list = sorted(factorlist, key=lambda x: abs(x), reverse=True)
        all_factors.append(sorted_factor_list)

    all_factors_final_list = [item for sublist in all_factors for item in sublist]
    L = np.eye(n)
    count = 0
    for i in range(n - 1):
        for iteration in range(i + 1, n):
            L[iteration, i] = all_factors_final_list[count]
            count += 1
    U = matrix
    return L, U, P

def factor(A, n, pivot=np.array([])):
    pivot = np.eye(n)
    num_rows = A.shape[0]
    num_col = A.shape[1]
    if num_rows != n or num_col != n:
        print('Matrix A must be square')
        return
    A2 = partial_pivot(A)
    L = A2[0]
    U = A2[1]
    P = A2[2]
    pivot = P
    A = [L, U]
    return A, n, pivot

def solve(A, n, pivot, b, x=0):
    p = pivot
    L = A[0]
    U = A[1]
    ys = np.zeros((n, 1))
    xs = np.zeros((n, 1))
    results_list = []
    b_numeric = np.array(b, dtype=float)
    btrue = np.matmul(p, b_numeric)
    y = np.linalg.solve(L, btrue)
    x = np.linalg.solve(U, y)
    x_rounded = np.round(x, decimals=6)
    results_list.append(x_rounded)
    return results_list

def main():
    infile = 'lu1.dat'  # Specify the file to read
    read_file = read_in(infile)
    rankn = read_file[0]
    amat = read_file[1]
    numright = read_file[2]
    rightlist = read_file[3]
    factorit = factor(amat, rankn)
    newaMat = factorit[0]
    L = newaMat[0]
    U = newaMat[1]
    LU_combine = np.zeros((rankn, rankn))
    for i in range(rankn):
        for j in range(rankn):
            if i > j:
                LU_combine[i, j] = L[i, j]
            else:
                LU_combine[i, j] = U[i, j]
    print("L\\U =", end=" ")
    for i in range(len(LU_combine)):
        print(" ".join(f"{value:.2f}" for value in LU_combine[i]))
    print('\n')
    newpiv = factorit[2]
    rank = factorit[1]
    for i in range(numright):
        solved = solve(newaMat, rank, newpiv, rightlist[i])
        solved_flattened = np.squeeze(solved)
        print('b = ', end=' ')
        for i in np.squeeze(rightlist[i]):
            print(i, end=' ')
        print('\n')
        print('x = ', end=' ')
        for i in solved_flattened:
            print(i, end=' ')
        print('\n')

if __name__ == "__main__":
    main()
