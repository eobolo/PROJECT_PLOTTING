import numpy as np
from sympy import symbols, Eq, Matrix, solve, sympify

class GaussianElimination:
    """
    A class used to solve systems of linear equations using Gaussian Elimination.

    Attributes:
    ----------
    equations : str
        A string containing the system of linear equations.
    variables : str
        A string containing the variables in the system of linear equations.
    A : numpy.ndarray
        The coefficient matrix of the system of linear equations.
    B : numpy.ndarray
        The constant matrix of the system of linear equations.

    Methods:
    -------
    string_to_augmented_matrix(equations)
        Converts a string of linear equations into an augmented matrix.
    swap_rows(M, row_index_1, row_index_2)
        Swaps two rows in a matrix.
    get_index_first_non_zero_value_from_column(M, column, starting_row)
        Finds the index of the first non-zero value in a column of a matrix.
    get_index_first_non_zero_value_from_row(M, row, augmented=False)
        Finds the index of the first non-zero value in a row of a matrix.
    augmented_matrix(A, B)
        Creates an augmented matrix from a coefficient matrix and a constant matrix.
    row_echelon_form(A, B)
        Converts a matrix into row echelon form.
    back_substitution(M)
        Solves a system of linear equations in row echelon form using back substitution.
    gaussian_elimination(A, B)
        Solves a system of linear equations using Gaussian Elimination.
    """

    def __init__(self, equations):
        """
        Initializes the GaussianElimination class.

        Parameters:
        ----------
        equations : str
            A string containing the system of linear equations.
        """
        self.equations = equations
        self.variables, self.A, self.B = self.string_to_augmented_matrix(equations)

    def string_to_augmented_matrix(self, equations):
        """
        Converts a string of linear equations into an augmented matrix.

        Parameters:
        ----------
        equations : str
            A string containing the system of linear equations.

        Returns:
        -------
        variables : str
            A string containing the variables in the system of linear equations.
        A : numpy.ndarray
            The coefficient matrix of the system of linear equations.
        B : numpy.ndarray
            The constant matrix of the system of linear equations.
        """
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

    def swap_rows(self, M, row_index_1, row_index_2):
        """
        Swaps two rows in a matrix.

        Parameters:
        ----------
        M : numpy.ndarray
            The matrix.
        row_index_1 : int
            The index of the first row.
        row_index_2 : int
            The index of the second row.

        Returns:
        -------
        M : numpy.ndarray
            The matrix with the rows swapped.
        """
        M = M.copy()
        if row_index_1 != row_index_2:
            M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
        return M

    def get_index_first_non_zero_value_from_column(self, M, column, starting_row):
        """
        Finds the index of the first non-zero value in a column of a matrix.

        Parameters:
        ----------
        M : numpy.ndarray
            The matrix.
        column : int
            The index of the column.
        starting_row : int
            The starting row index.

        Returns:
        -------
        index : int
            The index of the first non-zero value in the column.
        """
        column_array = M[starting_row:, column]
        for i, val in enumerate(column_array):
            if not np.isclose(val, 0, atol=1e-4):
                index = i + starting_row
                return index
        return -1

    def get_index_first_non_zero_value_from_row(self, M, row, augmented=False):
        """
        Finds the index of the first non-zero value in a row of a matrix.

        Parameters:
        ----------
        M : numpy.ndarray
            The matrix.
        row : int
            The index of the row.
        augmented : bool, optional
            Whether the matrix is augmented (default is False).

        Returns:
        -------
        index : int
            The index of the first non-zero value in the row.
        """
        M = M.copy()
        if augmented:
            M = M[:, :-1]
        row_array = M[row]
        for i, val in enumerate(row_array):
            if not np.isclose(val, 0, atol=1e-3):
                return i
        return -1

    def augmented_matrix(self, A, B):
        """
        Creates an augmented matrix from a coefficient matrix and a constant matrix.

        Parameters:
        ----------
        A : numpy.ndarray
            The coefficient matrix.
        B : numpy.ndarray
            The constant matrix.

        Returns:
        -------
        augmented_M : numpy.ndarray
            The augmented matrix.
        """
        augmented_M = np.hstack((A, B))
        return augmented_M

    def row_echelon_form(self, A, B):
        """
        Converts a matrix into row echelon form.

        Parameters:
        ----------
        A : numpy.ndarray
            The coefficient matrix.
        B : numpy.ndarray
            The constant matrix.

        Returns:
        -------
        M : numpy.ndarray
            The matrix in row echelon form.
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
        Solves a system of linear equations in row echelon form using back substitution.

        Parameters:
        ----------
        M : numpy.ndarray
            The matrix in row echelon form.

        Returns:
        -------
        solution : numpy.ndarray
            The solution to the system of linear equations.
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

    def gaussian_elimination(self, A, B):
        """
        Solves a system of linear equations using Gaussian Elimination.

        Parameters:
        ----------
        A : numpy.ndarray
            The coefficient matrix.
        B : numpy.ndarray
            The constant matrix.

        Returns:
        -------
        solution : numpy.ndarray
            The solution to the system of linear equations.
        """
        row_echelon_M = self.row_echelon_form(A, B)
        if not isinstance(row_echelon_M, str):
            solution = self.back_substitution(row_echelon_M)
        return solution

    def solve(self):
        """
        Solves the system of linear equations.

        Returns:
        -------
        solution : numpy.ndarray
            The solution to the system of linear equations.
        """
        solution = self.gaussian_elimination(self.A, self.B)
        return solution

# Example usage
equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

gaussian_elimination = GaussianElimination(equations)
solution = gaussian_elimination.solve()

if not isinstance(solution, str):
    for variable, value in zip(gaussian_elimination.variables.split(' '), solution):
        print(f"{variable} = {value:.4f}")
else:
    print(solution)

------------------------------------------------------
x = -1.5414
y = -0.5223
w = -0.1210
z = 1.1855
------------------------------------------------------
