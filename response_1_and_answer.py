import numpy as np
from sympy import symbols, Eq, Matrix, solve, sympify

def string_to_augmented_matrix(equations):
    equation_list = equations.split('\n')
    equation_list = [x for x in equation_list if x != '']
    coefficients = []

    ss = ''
    for c in equations:
        if c in 'abcdefghijklmnopqrstuvwxyz':
            if c not in ss:
                ss += c + ' '
    ss = ss[:-1]
    variables = symbols(ss)
    for equation in equation_list:
        sides = equation.replace(' ', '').split('=')
        left_side = sympify(sides[0])
        coefficients.append([left_side.coeff(variable) for variable in variables])
        coefficients[-1].append(int(sides[1]))
    augmented_matrix = Matrix(coefficients)
    augmented_matrix = np.array(augmented_matrix).astype("float32")

    A, B = augmented_matrix[:, :-1], augmented_matrix[:, -1].reshape(-1, 1)

    return ss, A, B

def swap_rows(M, row_index_1, row_index_2):
    M = M.copy()
    if row_index_1 != row_index_2:
        M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

def get_index_first_non_zero_value_from_column(M, column, starting_row):
    column_array = M[starting_row:, column]
    for i, val in enumerate(column_array):
        if not np.isclose(val, 0, atol=1e-4):
            index = i + starting_row
            return index
    return -1

def get_index_first_non_zero_value_from_row(M, row, augmented=False):
    M = M.copy()
    if augmented:
        M = M[:, :-1]
    row_array = M[row]
    for i, val in enumerate(row_array):
        if not np.isclose(val, 0, atol=1e-3):
            return i
    return -1

def augmented_matrix(A, B):
    augmented_M = np.hstack((A, B))
    return augmented_M

def row_echelon_form(A, B):
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        return 'Singular system'

    A = A.copy()
    B = B.copy()

    A = A.astype('float64')
    B = B.astype('float64')

    num_rows = len(A)

    M = augmented_matrix(A, B)
    for row in range(num_rows):
        pivot_candidate = M[row, row]
        if np.isclose(pivot_candidate, 0):
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(M, row, row)
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate)
            pivot = M[row, row]
        else:
            pivot = pivot_candidate
        M[row] = (1 / pivot) * M[row]
        for j in range(row + 1, num_rows):
            value_below_pivot = M[j, row]
            M[j] = M[j] - value_below_pivot * M[row]

    return M

def back_substitution(M):
    M = M.copy()
    num_rows = M.shape[0]
    for row in reversed(range(num_rows)):
        substitution_row = M[row]
        index = get_index_first_non_zero_value_from_row(M, row, augmented=True)
        for j in range(row):
            row_to_reduce = M[j]
            value = row_to_reduce[index - 1]
            row_to_reduce = row_to_reduce - value * substitution_row
            M[j, :] = row_to_reduce

    solution = M[:, -1]

    return solution

def gaussian_elimination(A, B):
    row_echelon_M = row_echelon_form(A, B)
    if not isinstance(row_echelon_M, str):
        solution = back_substitution(row_echelon_M)
    return solution

equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

variables, A, B = string_to_augmented_matrix(equations)

sols = gaussian_elimination(A, B)

if not isinstance(sols, str):
    for variable, solution in zip(variables.split(' '), sols):
        print(f"{variable} = {solution:.4f}")
else:
    print(sols)
"""
answer for response 2
x = -1.6584
y = 2.3577
w = -1.3685
z = 1.1855
"""
