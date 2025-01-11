# Implémentation de l'algorithme A* et des heuristiques
import heapq
import sys

import cProfile
import inspect
from typing import List, Callable, Dict,Union, Set, Any, Tuple
from heuristics import manhattan_distance,  hamming_distance, manhattan_metric, hamming_metric, linear_conflict_distance
from heuristics_util import get_and_filter_line, linear_conflict_on_multiple_lines
from parser import is_solvable, is_linear_solvable
from utils import display_puzzle
from test_goal_generator import generate_goal, generate_goal_linear

class PuzzleState:
    def __init__(self, puzzle, goal, size):
        self.puzzle = puzzle
        self.size = size
        self.goal = goal
        self.goal_positions = precompute_goal_positions(self.goal, self.size)
        self.column_indices = [[k * size + j for k in range(size)] for j in range(size)]
        self.row_goal_indices = precompute_goal_indices(self.goal_positions, is_row=True)
        self.col_goal_indices = precompute_goal_indices(self.goal_positions, is_row=False)
        self.goal_positions = precompute_goal_positions(goal, size)
        self.manhattan_precomputed = precompute_distance_dictionary(self.goal_positions, size, manhattan_metric)
        self.hamming_precomputed = precompute_distance_dictionary(self.goal_positions, size, hamming_metric)
        
        self.goal_columns_computed = precompute_goal_columns(goal, size)
        self.goal_rows_computed = precompute_goal_rows(goal, size)

def precompute_goal_positions(goal: List[int], size: int)-> Dict[int, tuple[int, int]]:
    """
    Precomputes the positions of tiles in the goal state.

    Maps each tile value to its (column, row) position in the goal state grid.

    goal: A list of integers representing the tiles in the goal state.
    size: The dimension of the grid (e.g., 4 for a 4x4 puzzle).
    return: A dictionary where keys are tile values and values are tuples (column, row) representing their positions.
    """
    goal_positions = {}
    for i, tile in enumerate(goal):
        goal_positions[tile] = (i % size, i // size)
    return goal_positions


def	precompute_goal_columns(goal: List[int], size: int)->Dict[int, Dict[str, Union[List[int], Set[int]]]]:
    """
    Precomputes the columns of the goal state and organizes them into a structured format.

    For each column index, creates a dictionary containing:
    - 'values': A list of tile values in the column.
    - 'set': A set of unique tile values in the column.

    goal: A list of integers representing the tiles in the goal state.
    size: The dimension of the grid (e.g., 4 for a 4x4 puzzle).
    return: A dictionary where keys are column indices, and values are dictionaries with 'values' and 'set' of tile values.
    """
    goal_columns = {}
    for index in range(size):
        column = [goal[index + j * size] for j in range(size)]
        goal_columns[index] = {'values': column, 'set': set(column)}
    return goal_columns

def precompute_goal_rows(goal: List[int], size: int)->Dict[int, Dict[str, Union[List[int], Set[int]]]]:
    """
    Precomputes the rows of the goal state and organizes them into a structured format.

    For each column index, creates a dictionary containing:
    - 'values': A list of tile values in the column.
    - 'set': A set of unique tile values in the column.

    goal: A list of integers representing the tiles in the goal state.
    size: The dimension of the grid (e.g., 5 for a 5x5 puzzle).
    return: A dictionary where keys are row indices, and values are dictionaries with 'values' and 'set' of tile values.
    """
    goal_rows = {}
    for index in range(size):
        row = [goal[j + index * size] for j in range(size)]
        goal_rows[index]  = {'values': row, 'set': set(row)}
    return goal_rows

def precompute_goal_indices(goal_positions:Dict[int, tuple[int, int]], is_row=True):
    """
    Precompute goal indices for rows or columns.
    :param goal_positions: Dictionary mapping tiles to their goal positions.
    :param is_row: True to compute row indices, False for column indices.
    :return: Dictionary mapping tiles to their goal row or column index.
    """
    index_type = 0 if is_row else 1
    return {tile: pos[index_type] for tile, pos in goal_positions.items()}


def precompute_distance_dictionary(goal_positions: Dict[int, tuple[int, int]], size: int, distance_func) -> Dict[int, Dict[int, int]]:
    """
    Precomputes heuristic distances for a given heuristic function.

    Args:
        goal_positions (dict): A dictionary mapping each tile to its goal position as (x, y).
        size (int): The size of the puzzle (e.g., 4 for a 4x4 puzzle).
        distance_func (callable): A function that computes the distance between two points (x1, y1) and (x2, y2).

    Returns:
        dict: A nested dictionary with precomputed distances.
            Outer key: Tile value
            Inner key: Current position
            Value: Distance from the current position to the tile's goal position
    """
    distance_dictionary = {}
    for tile in range(size * size):
        tile_distance = {}
        goal_x, goal_y = goal_positions[tile]
        for position in range(size * size):
            tile_x, tile_y = position % size, position // size
            distance = distance_func(goal_x, goal_y, tile_x, tile_y)
            tile_distance[position] = distance
        distance_dictionary[tile] = tile_distance
    return distance_dictionary


def update_Manhattan_distance(last_h: int, zero_position: int, target_position: int, moved_tile:int, manhattan_precomputed: Dict[int, Dict[int, int]])-> int:
    """
    Update the Manhattan distance after moving a tile to the zero position.
    last_h: the Manhattan distance of the last state
    zero_position: the position of the zero tile before the move
    target_position: the new position of the zero tile after the move
    moved_tile: the tile that moved to the zero position
    manhattan_precomputed (Dict[int, Dict[int, int]]): Precomputed Manhattan distances
        where the outer key is the tile value, the inner key is the position, 
        and the value is the Manhattan distance to the tile's goal position.
    return: int: The updated Manhattan distance after the move. 	
    """

    last_h -= manhattan_precomputed[moved_tile][target_position]
    last_h += manhattan_precomputed[moved_tile][zero_position]

    return last_h

def get_heuristic_function(heuristic_name: str):
    """
    Renvoie la fonction heuristique en fonction du nom spécifié.
    """
    try:
        if heuristic_name == "manhattan":
            return manhattan_distance
        elif heuristic_name == "hamming":
            return hamming_distance
        elif heuristic_name == "linear_conflict":
            return linear_conflict_distance #modified
        else:
            raise ValueError("Heuristic not supported")
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit()

def solve_puzzle(algorithm, puzzle, size, heuristic_name, inversions, snail):
    """
    Sélectionne la meilleure stratégie de résolution en fonction du nombre d'inversions.
    - algorithm: le choix algorithmique (A-star, greedy ou IDA ou uniform-cost)
    - puzzle : état initial du puzzle sous forme de liste
    - size : taille du puzzle
    - heuristic_name : nom de l'heuristique à utiliser
    - inversions : nombre d'inversions dans le puzzle
    """
    if snail:
        goal = generate_goal(size)
        # Vérification de la solvabilité
        if not is_solvable(puzzle, size, goal):
            print("Le puzzle n'est pas résolvable.")
            return None
    else:
        goal = generate_goal_linear(size)
        if not is_linear_solvable(puzzle, size): 
            print("Le puzzle n'est pas résolvable.")
            return None

    # Vérification de la solvabilité
    # if not is_solvable(puzzle, size, goal): #modif ici
    #     print("Le puzzle n'est pas résolvable.")
    #     return None

    # Choix de l'heuristique
    heuristic_func = get_heuristic_function(heuristic_name)

    # Choix de la stratégie en fonction des inversions
    try:
        if algorithm == "A-star" or "greedy" or "uniform-cost":
            return solve_puzzle_algorithm(puzzle, goal, size, heuristic_func, algorithm)
        elif algorithm == "IDA":
            return ida_star(puzzle, goal, size, heuristic_func)
        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")
    except NotImplementedError as e:
        print(f"Erreur : {e}")
        sys.exit()

def get_neighbors(puzzle: List[int], size: int, zero_index: int, last_move: str=None) -> List[Tuple[int, int]]:
    """
    Generate a list of possible neighbor states for the given puzzle by moving the blank tile (0).
    
    Each neighbor is represented by a new puzzle configuration and the direction of the move made to 
    reach that configuration.

    Args:
        puzzle (List[int]): The current state of the puzzle represented as a flat list of integers.
        Each element corresponds to a tile, with 0 representing the blank tile.
        size (int): The size of the puzzle (e.g., 3 for a 3x3 puzzle).
        zero_index (int): The index of the blank tile (0) in the puzzle.
        last_move (str, optional): The direction of the last move made ("up", "down", "left", "right").
            This helps prevent reversing the last move (default is None).

    Returns:
        List[Tuple[List[int], str]]: A list of tuples, where each tuple contains:
            - A new puzzle state (List[int]) after moving the blank tile.
            - A string representing the direction of the move made to reach that state ("up", "down", "left", or "right").
    
    """
    neighbors = []
    x, y = zero_index % size, zero_index // size

    moves = [
        (x - 1, y, "left") if x > 0 and last_move != "right" else None,
        (x + 1, y, "right") if x < size - 1 and last_move != "left" else None,
        (x, y - 1, "up") if y > 0 and last_move != "down" else None,
        (x, y + 1, "down") if y < size - 1 and last_move != "up" else None
    ]

    for move in moves:
        if move is not None:
            new_x, new_y, direction = move
            new_index = new_y * size + new_x
            new_puzzle = puzzle[:]
            new_puzzle[zero_index], new_puzzle[new_index] = new_puzzle[new_index], new_puzzle[zero_index]
            neighbors.append((new_puzzle, direction))
    return neighbors

def solve_puzzle_algorithm(puzzle: List[int], goal: List[int], size: int, heuristic_func: Callable[[List[int], Dict[int, Dict[int, int]], int], int], algorithm: str) -> Dict[str, Any]:
    """
    Unified function to solve a puzzle using either A* or Greedy algorithm.
    "A-star" : Utilise f=g+h comme priorité.
    Uniform-Cost Search : Utilisez uniquement g(x) (le coût réel accumulé) comme critère de priorité.
    Greedy Search : Utilisez uniquement h(x) (l’heuristique) comme critère de priorité.

    Parameters:
    - puzzle: Initial puzzle state as a flattened list.
    - goal: Goal puzzle state as a flattened list.
    - size: Size of the puzzle (e.g., 3 for 3x3).
    - heuristic_func: Function to calculate heuristic cost.
    - algorithm: The algorithm to use ('A-star' or 'greedy').

    Returns:
    - A dictionary containing the solution path, number of moves, states explored, and max states in memory.
    """
    print(f"{algorithm} search: Goal state\n")
    display_puzzle(goal, size)
    print(f"{algorithm} search: Initial state\n")
    display_puzzle(puzzle, size)

    # Precomputations
    column_indices = [[k * size + j for k in range(size)] for j in range(size)]
    goal_positions = precompute_goal_positions(goal, size)
    row_goal_indices = precompute_goal_indices(goal_positions, is_row=True)
    col_goal_indices = precompute_goal_indices(goal_positions, is_row=False)
    goal_columns_computed = precompute_goal_columns(goal, size)
    goal_rows_computed = precompute_goal_rows(goal, size)
    manhattan_precomputed = precompute_distance_dictionary(goal_positions, size, manhattan_metric)
    hamming_precomputed = precompute_distance_dictionary(goal_positions, size, hamming_metric)

    # Initialization
    open_set = []
    visited = set()
    came_from = {}

    zero_position = puzzle.index(0)
    if heuristic_func is manhattan_distance:
        initial_h = heuristic_func(puzzle, manhattan_precomputed, size)
    elif heuristic_func is hamming_distance:
        initial_h = hamming_distance(puzzle, goal, goal_positions, size)
    elif heuristic_func is linear_conflict_distance:
        initial_h = linear_conflict_distance(puzzle, goal_positions, size)
        initial_h += manhattan_distance(puzzle, manhattan_precomputed, size)
    else:
        initial_h = heuristic_func(puzzle, manhattan_precomputed, size)

    start_tuple = tuple(puzzle)
    goal_tuple = tuple(goal)

    if algorithm == "A-star":
        heapq.heappush(open_set, (initial_h, start_tuple, 0, None, zero_position))  # f = g + h
    elif algorithm == "greedy":
        heapq.heappush(open_set, (initial_h, start_tuple, None, zero_position))  # Only h
    elif algorithm == "uniform-cost":
        heapq.heappush(open_set, (0, start_tuple, None, zero_position))  # Only g

    visited.add(start_tuple)
    came_from[start_tuple] = None

    states_explored = 0
    max_states_in_memory = 1

    # Search loop
    while open_set:
        if algorithm == "A-star":
            f, current_tuple, g, last_move, zero_position = heapq.heappop(open_set)
        elif algorithm == "greedy":
            f, current_tuple, last_move, zero_position = heapq.heappop(open_set)
            g = 0  # Greedy ignores g
        elif algorithm == "uniform-cost":
            g, current_tuple, last_move, zero_position = heapq.heappop(open_set)
            f = g  # Uniform-cost uses only g

        current = list(current_tuple)

        if current_tuple == goal_tuple:
            solution_path = []
            while current_tuple is not None:
                solution_path.append(list(current_tuple))
                current_tuple = came_from.get(current_tuple)
            solution_path.reverse()
            return {
                "path": solution_path,
                "moves": len(solution_path) - 1,
                "states_explored": states_explored,
                "max_states_in_memory": max_states_in_memory
            }

        neighbors = get_neighbors(current, size, zero_position, last_move)
        for neighbor, move in neighbors:
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                came_from[neighbor_tuple] = current_tuple

                if move == "left":
                    new_empty_position = zero_position - 1
                elif move == "right":
                    new_empty_position = zero_position + 1
                elif move == "up":
                    new_empty_position = zero_position - size
                elif move == "down":
                    new_empty_position = zero_position + size

                # Heuristic calculations
                if heuristic_func is hamming_distance:
                    tile_being_moved = current[new_empty_position]
                    new_tile_position = new_empty_position
                    new_heuristic = f - g if algorithm == "A-star" else f
                    new_heuristic -= hamming_precomputed[tile_being_moved][new_tile_position]
                    new_heuristic += hamming_precomputed[tile_being_moved][zero_position]
                elif heuristic_func is manhattan_distance or heuristic_func is linear_conflict_distance:
                    tile_being_moved = current[new_empty_position]
                    new_tile_position = new_empty_position
                    new_heuristic = f - g if algorithm == "A-star" else f
                    new_heuristic -= manhattan_precomputed[tile_being_moved][new_tile_position]
                    new_heuristic += manhattan_precomputed[tile_being_moved][zero_position]

                    if heuristic_func is linear_conflict_distance:
                        if heuristic_func is linear_conflict_distance:
                            tile_being_moved = current[new_empty_position]
                            new_tile_position = new_empty_position
                            new_heuristic = f
                            modified_column = []
                            modified_row = []

                            if move == 'left' or move == 'right':

                                tmp = zero_position % size
                                modified_column.append(tmp)

                                if move == 'left':
                                    modified_column.append(tmp - 1)
                                else:
                                    modified_column.append(tmp + 1)

                                current_modified_line_filtered = get_and_filter_line(current, size, modified_column, column_indices, goal_columns_computed, is_row=False)
                                goal_modified_line = [goal_columns_computed[modified_column[0]]['values'],goal_columns_computed[modified_column[1]]['values']]
                                
                                neighbor_modified_line_filtered = get_and_filter_line(neighbor, size, modified_column, column_indices, goal_columns_computed, is_row=False)
                                
                                LC_removal = linear_conflict_on_multiple_lines(current_modified_line_filtered, goal_modified_line, col_goal_indices)
                                LC_add = linear_conflict_on_multiple_lines(neighbor_modified_line_filtered, goal_modified_line, col_goal_indices)

                                new_heuristic -= LC_removal
                                new_heuristic += LC_add
                            else:
                                tmp = zero_position // size
                                modified_row.append(tmp)
                                if move == "up":
                                    modified_row.append(tmp - 1)
                                else:
                                    modified_row.append(tmp + 1)

                                current_modified_line_filtered = get_and_filter_line(current, size, modified_row, column_indices, goal_rows_computed, is_row=True)
                                goal_modified_line = [goal_rows_computed[modified_row[0]]['values'],goal_rows_computed[modified_row[1]]['values']]

                                neighbor_modified_line_filtered = get_and_filter_line(neighbor, size, modified_row, column_indices, goal_rows_computed, is_row=True)

                                
                                LC_removal = linear_conflict_on_multiple_lines(current_modified_line_filtered, goal_modified_line, row_goal_indices)
                                LC_add = linear_conflict_on_multiple_lines(neighbor_modified_line_filtered, goal_modified_line, row_goal_indices)

                                new_heuristic -= LC_removal
                                new_heuristic += LC_add
                            pass
                else:
                    new_heuristic = heuristic_func(neighbor, goal_positions, size)

                new_g = g + 1  # Coût réel pour Uniform-Cost et A*
                new_h = new_heuristic  # Coût heuristique
                if algorithm == "A-star":
                    new_f = new_g + new_h
                    heapq.heappush(open_set, (new_f, neighbor_tuple, new_g, move, new_empty_position))
                elif algorithm == "greedy":
                    heapq.heappush(open_set, (new_h, neighbor_tuple, move, new_empty_position))
                elif algorithm == "uniform-cost":
                    heapq.heappush(open_set, (new_g, neighbor_tuple, move, new_empty_position))

        max_states_in_memory = max(max_states_in_memory, len(open_set))
        states_explored += 1

    return None

def ida_star(puzzle: List[int], goal: List[int], size: int,heuristic_func: Callable[[List[int], Dict[int, Dict[int, int]], int], int]) -> List[List[int]]:
    state = PuzzleState(puzzle, goal, size)

    print("IDA search: Goal state\n")
    display_puzzle(goal, size)
    print("IDA search: Initial state\n")
    display_puzzle(puzzle, size)

    
    goal_positions = state.goal_positions
    manhattan_precomputed = state.manhattan_precomputed


    if heuristic_func is manhattan_distance:
        threshold = heuristic_func(puzzle, manhattan_precomputed, size)
    elif heuristic_func is hamming_distance:
        threshold = hamming_distance(puzzle, goal,goal_positions, size)
    elif heuristic_func is linear_conflict_distance:
        threshold = linear_conflict_distance(puzzle, goal_positions, size)
        tmp = manhattan_distance(puzzle, manhattan_precomputed, size)
        threshold += tmp

    f = None

    while True:
        result = dfs(state, puzzle, goal, size, heuristic_func, 0, threshold, [],f)

        if isinstance(result, dict):
            return result
        if result == float('inf'):
            return None
        threshold = result

def dfs(state:PuzzleState,puzzle: List[int], goal: List[int], size: int, heuristic_func: Callable[[List[int], Dict[int, Dict[int, int]], int], int], 
        g:int, 
        threshold:int, 
        path: List[List[int]],
        f
        )-> Union[Dict[str, Any], int]:
    """
    Depth-First Search 
    Parameter:
    state: class PuzzleState, a container
    puzzle: A list of int (in flattened notation) representing the puzzle to be solved
    goal: A list of int (in flattened notation) representing the final state
    heuristic_func : a function for computing the heuristic cost
    g: The cost,
    threshold: The maximum depth dfs searches
    path: A list of list of int, representing the solution
    f: int the heuristics cost
    """
    goal_positions = state.goal_positions
    manhattan_precomputed = state.manhattan_precomputed
    hamming_precomputed = state.hamming_precomputed
    column_indices = state.column_indices
    row_goal_indices = state.row_goal_indices
    col_goal_indices = state.col_goal_indices
    goal_columns_computed = state.goal_columns_computed
    goal_rows_computed = state.goal_rows_computed

    if f is None:
        if heuristic_func is manhattan_distance:
            h = heuristic_func(puzzle, manhattan_precomputed, size)
        elif heuristic_func is hamming_distance:
            h = hamming_distance(puzzle, goal,goal_positions, size)
        elif heuristic_func is linear_conflict_distance:
            h = linear_conflict_distance(puzzle, goal_positions, size)
            tmp = manhattan_distance(puzzle, manhattan_precomputed, size)
            h += tmp
        f = g + h

    if f > threshold:
        return f
    if puzzle == goal:
        return {
            "path": path + [puzzle],
            "moves": len(path),
            "states_explored": len(path),
            "max_states_in_memory": len(path)
        }

    
    min_cost = float('inf')
    zero_position = puzzle.index(0)

    for neighbor, move in get_neighbors(puzzle, size, zero_position, None):
        if path and neighbor == path[-1]:
            continue 

        if move == "left":
            new_empty_position = zero_position - 1
        elif move == "right":
            new_empty_position = zero_position + 1
        elif move == "up":
            new_empty_position = zero_position - size
        elif move == "down":
            new_empty_position = zero_position + size

        if heuristic_func is hamming_distance:
            tile_being_moved = puzzle[new_empty_position]
            new_tile_position = new_empty_position
            new_heuristic = f - g
            new_heuristic -= hamming_precomputed[tile_being_moved][new_tile_position]
            new_heuristic += hamming_precomputed[tile_being_moved][zero_position]

        elif heuristic_func is manhattan_distance or heuristic_func is linear_conflict_distance:
            tile_being_moved = puzzle[new_empty_position]
            new_tile_position = new_empty_position
            new_heuristic = f - g
            new_heuristic -= manhattan_precomputed[tile_being_moved][new_tile_position]
            new_heuristic += manhattan_precomputed[tile_being_moved][zero_position]
            if heuristic_func is linear_conflict_distance:
                tile_being_moved = puzzle[new_empty_position]
                new_tile_position = new_empty_position
                new_heuristic = f - g

                modified_column = []
                modified_row = []
                if move == 'left' or move == 'right':
                    tmp = zero_position % size
                    modified_column.append(tmp)
                    if move == 'left':
                        modified_column.append(tmp - 1)
                    else:
                        modified_column.append(tmp + 1)

                    
                    current_modified_line_filtered = get_and_filter_line(puzzle, size, modified_column, column_indices, goal_columns_computed, is_row=False)
                    goal_modified_line = [goal_columns_computed[modified_column[0]]['values'],goal_columns_computed[modified_column[1]]['values']]
                            
                    neighbor_modified_line_filtered = get_and_filter_line(neighbor, size, modified_column, column_indices, goal_columns_computed, is_row=False)
                            
                    LC_removal = linear_conflict_on_multiple_lines(current_modified_line_filtered, goal_modified_line, col_goal_indices)
                    LC_add = linear_conflict_on_multiple_lines(neighbor_modified_line_filtered, goal_modified_line, col_goal_indices)
                            

                    new_heuristic -= LC_removal
                    new_heuristic += LC_add
                else:
                    tmp = zero_position // size
                    modified_row.append(tmp)
                    if move == "up":
                        modified_row.append(tmp - 1)
                    else:
                        modified_row.append(tmp + 1)

                    
                    current_modified_line_filtered = get_and_filter_line(puzzle, size, modified_row, column_indices, goal_rows_computed, is_row=True)
                    goal_modified_line = [goal_rows_computed[modified_row[0]]['values'],goal_rows_computed[modified_row[1]]['values']]

                    neighbor_modified_line_filtered = get_and_filter_line(neighbor, size, modified_row, column_indices, goal_rows_computed, is_row=True)

                            
                    LC_removal = linear_conflict_on_multiple_lines(current_modified_line_filtered, goal_modified_line, row_goal_indices)
                    LC_add = linear_conflict_on_multiple_lines(neighbor_modified_line_filtered, goal_modified_line, row_goal_indices)

                    new_heuristic -= LC_removal
                    new_heuristic += LC_add
        else:
            new_heuristic = heuristic_func(neighbor, goal, goal_positions, size)
        
        g_new = g + 1
        f_new = g_new + new_heuristic
        
        result = dfs(state, neighbor, goal, size, heuristic_func, g_new, threshold,path + [puzzle], f_new)
        if isinstance(result, dict):
            return result
        min_cost = min(min_cost, result)
    return min_cost