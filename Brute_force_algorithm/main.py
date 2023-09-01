from itertools import combinations
import time


#Function that generate list of CNF base on the matrix
def generate_minesweeper_CNF(matrix):
    CNF = []
    rows, cols = len(matrix), len(matrix[0])
    added_clauses = []
    un_list=[]

    # Helper function to get adjacent cells
    def get_neighbors(row, col):
        list=[]
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if(0 <= row + dr < rows and 0 <= col + dc < cols and matrix[row+dr][col+dc] == 0):
                    list.append((row+dr)*cols + col+dc+1)
        return list

    # Helper function to add clauses
    def add_clause(clause):
        if clause and clause not in added_clauses:
            CNF.append(clause)
            added_clauses.append(clause)

    # Helper function to check if the cell '0' is 100% safe or not
    def is_safe(row, col):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if 0<= row + dr < rows and 0 <= col + dc < cols:
                    if(matrix[row+dr][col+dc] != 0):
                        return False
        return True

    # Add clue constraints
    for row in range(rows):
        for col in range(cols):
            clue = matrix[row][col]
            if clue > 0:
                CNF.append([-(row * cols + col + 1)])
                un_list.append(-(row * cols + col + 1))
                clue_neighbors = get_neighbors(row, col)
                if(clue > len(clue_neighbors)):
                    return None
                elif(clue == len(clue_neighbors)):
                    for cell in clue_neighbors:
                        add_clause([cell])
                        if cell not in un_list:
                            un_list.append(cell)
                else:
                    add_clause(clue_neighbors)
                    for combo in combinations(clue_neighbors, clue+1):
                        add_clause(list(-x for x in combo))
                    if(clue>1):
                        for combo in combinations(clue_neighbors, len(clue_neighbors)-clue+1):
                            add_clause(list(x for x in combo))

            # Add non-mine constraints for uncovered cells
            elif clue == 0 and is_safe(row,col):
                CNF.append([-(row * cols + col + 1)])
                if(-(row * cols + col + 1) not in un_list):
                    un_list.append(-(row * cols + col + 1))
    return CNF,un_list

# Brute force algorithm solve
def Brute_force_solve(CNF,n,unlist):
    # Help function to check if the state is valid or not
    def check(state):
        return all(any(var in state for var in clause) for clause in CNF)
    range_check = list(range(1, n+1))
    init_state=[-i for i in range(1,n+1)]
    for i in range(len(unlist)):
        x=abs(unlist[i])
        init_state[x-1]=unlist[i]
        range_check.remove(x)
    for i in range(2**len(range_check)):
        state = init_state[:]
        for j in range(len(range_check)):
            num = range_check[j] * (1 if (bool(i & (1 << j))) else -1)
            state[abs(num)-1]=num
        if check(state):
            return state
    return None

#Function solve CNFs using Brute Force algorithm
def solve_minesweeper_CNF(CNF,matrix,un_list):
    rows, cols = len(matrix), len(matrix[0])
    solution = Brute_force_solve(CNF,rows*cols,un_list)
    if solution!=None:
        model = list(solution)
        solved_matrix = [[' ' for _ in range(cols)] for _ in range(rows)]
        for lit in model:
            row, col = divmod(abs(lit) - 1, cols)
            if matrix[row][col]!=0 :
                solved_matrix[row][col] = matrix[row][col]
            elif lit > 0:
                solved_matrix[row][col] = 'X'
            else:
                solved_matrix[row][col] = '0'
        return solved_matrix
    else:
        return None

#Function that help you read the matrix in the input file
def read_matrix_from_file(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(int, line.strip().split()))
            matrix.append(row)
    return matrix

#Function that help you print the matrix to the output file
def write_matrix_to_file(file_path, matrix):
    with open(file_path, 'w') as file:
        for row in matrix:
            file.write(' '.join(str(cell) for cell in row) + '\n')


# Input and Output file paths
input_file_path = 'input.txt'
output_file_path = 'output.txt'

# Read the minesweeper matrix from input file
input_matrix = read_matrix_from_file(input_file_path)
rows, cols = len(input_matrix), len(input_matrix[0])

#Begin to create a list of CNF base on the input matrix
CNF,un_list= generate_minesweeper_CNF(input_matrix)

#Solve the CNF
start_time = time.time()
solved_matrix = solve_minesweeper_CNF(CNF,input_matrix,un_list)
end_time = time.time()

if solved_matrix:
    print("Solution found:")
    running_time = end_time - start_time
    print(f"Running time: {running_time:.6f} seconds.")
    write_matrix_to_file(output_file_path, solved_matrix)
else:
    print("No valid solution found.")
