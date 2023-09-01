from itertools import combinations
import  heapq
import time


#Function that generate list of CNF base on the matrix
def generate_minesweeper_CNF(matrix):
    CNF = []
    rows, cols = len(matrix), len(matrix[0])
    added_clauses = []
    mine_zone = []

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
                clue_neighbors = get_neighbors(row, col)
                if(clue > len(clue_neighbors)):
                    return None
                elif(clue == len(clue_neighbors)):
                    for cell in clue_neighbors:
                        add_clause([cell])
                else:
                    add_clause(clue_neighbors)
                    mine_zone.append((clue_neighbors, clue))
                    for combo in combinations(clue_neighbors, clue+1):
                        add_clause(list(-x for x in combo))
                    if(clue>1):
                        for combo in combinations(clue_neighbors, len(clue_neighbors)-clue+1):
                            add_clause(list(x for x in combo))

            # Add non-mine constraints for uncovered cells
            elif clue == 0 and is_safe(row,col):
                CNF.append([-(row * cols + col + 1)])
    return CNF,mine_zone

# A* algorithm solve
def A_solve(CNF,zone):
    #heuristic function
    def heuristic(state):
        return len(CNF) - sum(any(var in state for var in clause) for clause in CNF)

    #check is goal state function
    def is_goal(state):
        return all(any(var in state for var in clause) for clause in CNF)

    #function that put combo mines in list places
    def put_mine(state,comb,list):
        for c in list:
            if c in comb:
                state[c-1]=c
            else:
                state[c-1]=-c
        return state

    #function that help you create the successors
    def generate_successors(state,cell):
        successors = []
        list,mines=mine_zone[cell]
        for comb in combinations(list,mines):
            successor = state[:]
            new_succesor=put_mine(successor,comb,list)
            successors.append(new_succesor)
        return successors

    #Function optimize by creating the valid initial state and lock all the un-change cell
    def optimize():
        num_vars = max(abs(var) for clause in CNF for var in clause)
        initial_state = [-i for i in range(1, num_vars + 1)]
        flist = []
        for clause in CNF:
            if len(clause) == 1:
                if clause[0]>0:
                    flist.append(abs(clause[0]))
                initial_state[abs(clause[0]) - 1] = clause[0]
        return initial_state, flist

    initial_state, flist = optimize()
    mine_zone=[]
    for list,k in zone:
        n_list=[num for num in list if num not in flist]
        n_k=k-(len(list)-len(n_list))
        if (n_list,n_k) not in mine_zone and n_k>0:
            mine_zone.append((n_list,n_k))
    open_list = [(heuristic(initial_state), 0, initial_state)]
    closed_set = set()
    closed_set.add(tuple(initial_state))
    while open_list:
        _, st, current_state = heapq.heappop(open_list)
        if is_goal(current_state):
            return {i + 1: val for i, val in enumerate(current_state)}
        if(st<len(mine_zone)):
            successors = generate_successors(current_state,st)
            for successor_state in successors:
                if(tuple(successor_state) not in closed_set):
                    closed_set.add(tuple(successor_state))
                    heapq.heappush(open_list, (heuristic(successor_state)+st+1, st+1, successor_state))

    return None

#Function solve CNFs using A* algorithm
def solve_minesweeper_CNF(CNF, rows, cols,matrix,mine_zone):
    solution = A_solve(CNF,mine_zone)
    if solution!=None:
        model = list(solution.values())
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
CNF,mine_zone = generate_minesweeper_CNF(input_matrix)

#Solve the CNF
start_time = time.time()
solved_matrix = solve_minesweeper_CNF(CNF,rows,cols,input_matrix,mine_zone)
end_time = time.time()

if solved_matrix:
    print("Solution found:")
    running_time = end_time - start_time
    print(f"Running time: {running_time:.6f} seconds.")
    write_matrix_to_file(output_file_path, solved_matrix)
else:
    print("No valid solution found.")
