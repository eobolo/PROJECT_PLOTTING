import numpy as np
from sympy import symbols, Eq, Matrix, solve, sympify

class GaussianElimination:
    """
    This class provides methods for solving systems of linear equations using Gaussian elimination.
    """

    def __init__(self, equations):
        """
        Initialize the GaussianElimination class.

        Args:
            equations (str): A string containing the system of linear equations.
        """
        self.equations = equations

    def string_to_augmented_matrix(self):
        """
        Convert the system of linear equations into an augmented matrix.

        Returns:
            variables (str): A string containing the variables in the system.
            A (numpy array): The coefficient matrix.
            B (numpy array): The constant matrix.
        """
        equation_list = self.equations.split('\n')
        equation_list = [x for x in equation_list if x != '']
        coefficients = []

        # Extract variables from the equations
        variables = ''
        for c in self.equations:
            if c in 'abcdefghijklmnopqrstuvwxyz':
                if c not in variables:
                    variables += c + ' '
        variables = variables[:-1]

        # Create symbols for the variables
        variables = symbols(variables)

        # Iterate over each equation
        for equation in equation_list:
            sides = equation.replace(' ', '').split('=')
            left_side = sympify(sides[0])

            # Extract coefficients for each variable
            coefficients.append([left_side.coeff(variable) for variable in variables])
            coefficients[-1].append(int(sides[1]))

        # Create the augmented matrix
        augmented_matrix = Matrix(coefficients)
        augmented_matrix = np.array(augmented_matrix).astype("float32")

        # Split the augmented matrix into coefficient and constant matrices
        A, B = augmented_matrix[:, :-1], augmented_matrix[:, -1].reshape(-1, 1)

        return variables, A, B

    def swap_rows(self, M, row_index_1, row_index_2):
        """
        Swap two rows in a matrix.

        Args:
            M (numpy array): The matrix.
            row_index_1 (int): The index of the first row.
            row_index_2 (int): The index of the second row.

        Returns:
            M (numpy array): The matrix with the rows swapped.
        """
        M = M.copy()
        if row_index_1 != row_index_2:
            M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
        return M

    def get_index_first_non_zero_value_from_column(self, M, column, starting_row):
        """
        Get the index of the first non-zero value in a column.

        Args:
            M (numpy array): The matrix.
            column (int): The column index.
            starting_row (int): The starting row index.

        Returns:
            index (int): The index of the first non-zero value.
        """
        column_array = M[starting_row:, column]
        for i, val in enumerate(column_array):
            if not np.isclose(val, 0, atol=1e-4):
                index = i + starting_row
                return index
        return -1

    def get_index_first_non_zero_value_from_row(self, M, row, augmented=False):
        """
        Get the index of the first non-zero value in a row.

        Args:
            M (numpy array): The matrix.
            row (int): The row index.
            augmented (bool): Whether the matrix is augmented.

        Returns:
            index (int): The index of the first non-zero value.
        """
        M = M.copy()
        if augmented:
            M = M[:, :-1]
        row_array = M[row]
        for i, val in enumerate(row_array):
            if not np.isclose(val, 0, atol=1e-3):
                return i
        return -1

    def row_echelon_form(self, A, B):
        """
        Convert the matrix into row echelon form.

        Args:
            A (numpy array): The coefficient matrix.
            B (numpy array): The constant matrix.

        Returns:
            M (numpy array): The matrix in row echelon form.
        """
        det_A = np.linalg.det(A)
        if np.isclose(det_A, 0):
            return 'Singular system'

        A = A.copy()
        B = B.copy()

        A = A.astype('float64')
        B = B.astype('float64')

        num_rows = len(A)

        M = np.hstack((A, B))
        for row in range(num_rows):
            pivot_candidate = M[row, row]
            if np.isclose(pivot_candidate, 0):
                first_non_zero_value_below_pivot_candidate = self.get_index_first_non_zero_value_from_column(M, row, row)
                M = self.swap_rows(M, row, first_non_zero_value_below_pivot_candidate)
                pivot = M[row, row]
            else:
                pivot = pivot_candidate
            M[row] = (1 / pivot) * M[row]
            for j in range(row + 1, num_rows):
                value_below_pivot = M[j, row]
                M[j] = M[j] - value_below_pivot * M[row]

        return M

    def back_substitution(self, M):
        """
        Solve the system using back substitution.

        Args:
            M (numpy array): The matrix in row echelon form.

        Returns:
            solution (numpy array): The solution to the system.
        """
        M = M.copy()
        num_rows = M.shape[0]
        for row in reversed(range(num_rows)):
            substitution_row = M[row]
            index = self.get_index_first_non_zero_value_from_row(M, row, augmented=True)
            for j in range(row):
                row_to_reduce = M[j]
                value = row_to_reduce[index]
                row_to_reduce[-1] = row_to_reduce[-1] - value * substitution_row[-1]
                M[j, -1] = row_to_reduce[-1]

        solution = M[:, -1]

        return solution

    def gaussian_elimination(self):
        """
        Solve the system using Gaussian elimination.

        Returns:
            solution (numpy array): The solution to the system.
        """
        variables, A, B = self.string_to_augmented_matrix()
        row_echelon_M = self.row_echelon_form(A, B)
        if not isinstance(row_echelon_M, str):
            solution = self.back_substitution(row_echelon_M)
            return variables, solution
        else:
            return row_echelon_M


# Example usage
equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

gaussian_elimination_solver = GaussianElimination(equations)
solution = gaussian_elimination_solver.gaussian_elimination()

if not isinstance(solution, str):
    variables, solution_values = solution
    for variable, solution_value in zip(variables.split(' '), solution_values):
        print(f"{variable} = {solution_value:.4f}")
else:
    print(solution)
------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-6-48669e837ec5> in <cell line: 208>()
    208 if not isinstance(solution, str):
    209     variables, solution_values = solution
--> 210     for variable, solution_value in zip(variables.split(' '), solution_values):
    211         print(f"{variable} = {solution_value:.4f}")
    212 else:

AttributeError: 'tuple' object has no attribute 'split
----------------------------------------------------------------------------------------
