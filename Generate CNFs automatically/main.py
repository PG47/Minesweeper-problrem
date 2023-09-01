from itertools import combinations
import os

#Function that generate list of CNF base on the matrix
def generate_minesweeper_CNF(grid):
    CNF = []
    rows, cols = len(grid), len(grid[0])
    added_clauses = set()

    # Helper function to get adjacent cells
    def get_adjacent_cells(row, col):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if 0 <= row + dr < rows and 0 <= col + dc < cols:
                    yield (row + dr, col + dc)

    # Helper function to add clauses
    def add_clause(clause):
        if clause not in added_clauses:
            CNF.append(clause)
            added_clauses.add(clause)

    # Helper function to check if the cell '0' is 100% safe or not
    def is_safe(row, col):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if 0<= row + dr < rows and 0 <= col + dc < cols:
                    if(grid[row+dr][col+dc] != 0 and grid[row+dr][col+dc] != 'X'):
                        return False
        return True

    # Add clue constraints
    for row in range(rows):
        for col in range(cols):
            clue = grid[row][col]
            if clue > 0:
                clue_neighbors = []
                add_clause(f'¬C{row}_{col}')
                for r, c in get_adjacent_cells(row, col):
                    if grid[r][c] == 0:
                        clue_neighbors.append(f'C{r}_{c}')
                if (clue > len(clue_neighbors)):
                    return None
                elif (clue == len(clue_neighbors)):
                    for cell in clue_neighbors:
                        add_clause(cell)
                else:
                    add_clause(' OR '.join(clue_neighbors))
                    for combo in combinations(clue_neighbors, clue+1):
                        add_clause(' OR '.join(list(f'¬{x}' for x in combo)))
                    if(clue>1):
                        for combo in combinations(clue_neighbors, len(clue_neighbors)-clue+1):
                            add_clause(' OR '.join(combo))

    # Add non-mine constraints for uncovered cells
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 0 and is_safe(row,col):
                CNF.append(f'¬C{row}_{col}')

    return CNF

#Function that help you read the matrix in the input file
def read_matrix_from_file(file_path):
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"The input file '{file_path}' does not exist.")

	matrix = []
	with open(file_path, 'r') as file:
		num_cols = None
		for line in file:
			row = list(map(int, line.strip().split()))

			# Verify if the number of columns is consistent for each row
			if num_cols is None:
				num_cols = len(row)
			elif len(row) != num_cols:
				raise ValueError("Invalid matrix. Number of columns in each row must be the same.")

			matrix.append(row)

	# Verify if the number of rows is consistent
	if not all(len(row) == num_cols for row in matrix):
		raise ValueError("Invalid matrix. Number of rows must be the same.")

	return matrix



#MAIN PROGRAM
# Input file paths
file_path = 'input.txt'

try:
    # Read the minesweeper matrix from input file
	matrix = read_matrix_from_file(file_path)

    # Begin to create a list of CNF base on the input matrix
	CNF = generate_minesweeper_CNF(matrix)

    # Print all clause in the CNF to the console
	for clause in CNF:
		print(clause)
except FileNotFoundError as e:
	print(e)
except ValueError as e:
	print(e)


