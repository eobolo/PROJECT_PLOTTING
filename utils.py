from sympy import symbols, Eq, Matrix, solve, sympify
import numpy as np

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
    augmented_matrix = np.array(augmented_matrix).astype("float64")

    A, B = augmented_matrix[:,:-1], augmented_matrix[:,-1].reshape(-1,1)
    
    return ss, A, B

