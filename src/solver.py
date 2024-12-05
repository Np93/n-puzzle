# Implémentation de l'algorithme A* et des heuristiques
import heapq
import sys

import cProfile
import inspect
from typing import List, Callable, Dict, Any
from heuristics import manhattan_distance, linear_conflict, dynamic_misplaced_heuristic, hamming_distance, manhattan_metric, hamming_metric, linear_conflict_distance
from heuristics_util import get_and_filter_line, linear_conflict_on_multiple_lines
from parser import is_solvable
from utils import display_puzzle
from test_goal_generator import generate_goal

class PuzzleState:
	def __init__(self, puzzle, goal, size):
		self.puzzle = puzzle
		self.size = size
		self.goal = goal
		self.goal_positions = precompute_goal_positions(self.goal, self.size)
		self.column_indices = [[k * size + j for k in range(size)] for j in range(size)]
		self.row_goal_indices = precompute_goal_indices(self.goal_positions, is_row=True)
		self.col_goal_indices = precompute_goal_indices(self.goal_positions, is_row=False)


# Implementation d'optimisation
def precompute_goal_positions(goal, size):
	goal_positions = {}
	for i, tile in enumerate(goal):
		goal_positions[tile] = (i % size, i // size)
	return goal_positions

# at tester ATTN set is not preserving the order of the goal
def	precompute_goal_columns(goal, size):
	goal_columns = {}
	for index in range(size):
		column = [goal[index + j * size] for j in range(size)]
		goal_columns[index] = {'values': column, 'set': set(column)}
	return goal_columns

def precompute_goal_rows(goal, size):
	goal_rows = {}
	for index in range(size):
		row = [goal[j + index * size] for j in range(size)]
		goal_rows[index]  = {'values': row, 'set': set(row)}
	return goal_rows

def precompute_goal_indices(goal_positions, is_row=True):
	"""
	Precompute goal indices for rows or columns.
	:param goal_positions: Dictionary mapping tiles to their goal positions.
	:param is_row: True to compute row indices, False for column indices.
	:return: Dictionary mapping tiles to their goal row or column index.
	"""
	index_type = 0 if is_row else 1
	return {tile: pos[index_type] for tile, pos in goal_positions.items()}

def precompute_manhattan_dictionnary(goal_positions, size):
	manhattan_dictionnary = {}
	for tile in range(size * size):
		tile_distance = {}
		goal_x, goal_y = goal_positions[tile]
		for position in range(size * size):
			tile_x, tile_y = position % size, position // size
			distance = abs(goal_x - tile_x) + abs(goal_y - tile_y)
			tile_distance[position]= distance
		manhattan_dictionnary[tile]= tile_distance
	return manhattan_dictionnary

# generalisation
def precompute_distance_dictionary(goal_positions, size, distance_func):
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


def get_new_blank_position(zero_position, move, size):
	if move == 'left':
		return zero_position - 1
	elif move == 'right':
		return zero_position + 1
	elif move == 'up':
		return zero_position - size
	elif move == 'down':
		return zero_position + size

def get_moved_tile(puzzle, zero_position, move, size):
	# Determine the new position of the blank tile after the move
	if move == "left":
		new_zero_position = zero_position - 1
	elif move == "right":
		new_zero_position = zero_position + 1
	elif move == "up":
		new_zero_position = zero_position - size
	elif move == "down":
		new_zero_position = zero_position + size

	# The tile at the new zero position is the one that will move to the original zero position
	moved_tile = puzzle[new_zero_position]
	return moved_tile, new_zero_position


def update_Manhattan_distance(last_h, zero_position, target_position, moved_tile, manhattan_precomputed):
	"""
	Update the Manhattan distance after moving a tile to the zero position.
	last_h: the Manhattan distance of the last state
	zero_position: the position of the zero tile before the move
	target_position: the new position of the zero tile after the move
	tile: the tile that moved to the zero position
	"""

	last_h -= manhattan_precomputed[moved_tile][target_position]
	last_h += manhattan_precomputed[moved_tile][zero_position]

	return last_h

# fin implementation optimisation
# not done
def combined_heuristic(puzzle, goal, size):
	return manhattan_distance(puzzle, goal, size) + linear_conflict(puzzle, goal, size)

def get_heuristic_function(heuristic_name):
	"""
	Renvoie la fonction heuristique en fonction du nom spécifié.
	"""
	if heuristic_name == "manhattan":
		return manhattan_distance
	elif heuristic_name == "manhattan_conflict":
		return combined_heuristic
	elif heuristic_name == "dynamic_misplaced":
		return dynamic_misplaced_heuristic #a voir ?
	elif heuristic_name == "hamming":
		return hamming_distance
	elif heuristic_name == "linear_conflict":
		return linear_conflict_distance #modified
	else:
		raise ValueError("Heuristic not supported")

def solve_puzzle(puzzle, size, heuristic_name, inversions):
	"""
	Sélectionne la meilleure stratégie de résolution en fonction du nombre d'inversions.
	- puzzle : état initial du puzzle sous forme de liste
	- size : taille du puzzle
	- heuristic_name : nom de l'heuristique à utiliser
	- inversions : nombre d'inversions dans le puzzle
	"""
	goal = generate_goal(size)


	# Vérification de la solvabilité
	if not is_solvable(puzzle, size, goal): #modif ici
		print("Le puzzle n'est pas résolvable.")
		return None

	# Choix de l'heuristique
	heuristic_func = get_heuristic_function(heuristic_name)

	# Choix de la stratégie en fonction des inversions
	if inversions < 50000:

		#return greedy_search(puzzle, goal, size, heuristic_func)

		return a_star(puzzle, goal, size, heuristic_func)
		#return ida_star(puzzle, goal, size, heuristic_func)

	""" elif 50 <= inversions <= 100:
		print("Utilisation de la Recherche Gloutonne")
		return greedy_search(puzzle, goal, size, heuristic_func)
	else:
		print("Utilisation de IDA* pour de grandes inversions")
		return ida_star(puzzle, goal, size, heuristic_func) """


def get_neighbors(puzzle, size, zero_index, last_move=None):
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

def a_star(puzzle: List[int], goal: List[int], size: int,heuristic_func: Callable[[List[int], Dict[int, Dict[int, int]], int], int]) -> List[List[int]]:
	"""
	A* search algorithm for solving puzzles

	Parameter:
	puzzle: a list of int (in flattened notation) representing the puzzle to be solved
	goal: a list of int (in flattened notation) representing the final state
	heuristic_func : a function for computing the heuristic cost
	"""

	display_puzzle(goal, size)
	display_puzzle(puzzle, size)

	column_indices = [[k * size + j for k in range(size)] for j in range(size)]
	goal_positions = precompute_goal_positions(goal, size)
	row_goal_indices = precompute_goal_indices(goal_positions, is_row=True)
	col_goal_indices = precompute_goal_indices(goal_positions, is_row=False)
	goal_columns_computed = precompute_goal_columns(goal, size)
	goal_rows_computed = precompute_goal_rows(goal, size)
	manhattan_precomputed = precompute_distance_dictionary(goal_positions, size, manhattan_metric)
	hamming_precomputed = precompute_distance_dictionary(goal_positions, size, hamming_metric)


	open_set = []
	visited = set()
	came_from = {}

	zero_position = puzzle.index(0)
	if heuristic_func is manhattan_distance:
		initial_h = heuristic_func(puzzle, manhattan_precomputed, size)
	elif heuristic_func is hamming_distance:
		initial_h = hamming_distance(puzzle, goal,goal_positions, size)
	elif heuristic_func is linear_conflict_distance:
		initial_h = linear_conflict_distance(puzzle, goal_positions, size)
		tmp = manhattan_distance(puzzle, manhattan_precomputed, size)
		initial_h += tmp
	else:
		print(f"another one {heuristic_func}")
		initial_h = heuristic_func(puzzle, manhattan_precomputed, size)


	start_tuple = tuple(puzzle)
	goal_tuple = tuple(goal)

	heapq.heappush(open_set, (initial_h, start_tuple, 0, None, zero_position))
	visited.add(start_tuple)
	came_from[start_tuple] = None

	states_explored = 0
	max_states_in_memory = 1

	while open_set:
		f, current_tuple, g, last_move, zero_position = heapq.heappop(open_set)
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


				if heuristic_func is hamming_distance:
					tile_being_moved = current[new_empty_position]
					new_tile_position = new_empty_position
					new_heuristic = f - g
					new_heuristic -= hamming_precomputed[tile_being_moved][new_tile_position]
					new_heuristic += hamming_precomputed[tile_being_moved][zero_position]


				elif heuristic_func is manhattan_distance or heuristic_func is linear_conflict_distance: #linear-conflict with manhattan
					tile_being_moved = current[new_empty_position]
					new_tile_position = new_empty_position
					new_heuristic = f - g
					new_heuristic -= manhattan_precomputed[tile_being_moved][new_tile_position]
					new_heuristic += manhattan_precomputed[tile_being_moved][zero_position]


					if heuristic_func is linear_conflict_distance: # optimisé
						tile_being_moved = current[new_empty_position]
						new_tile_position = new_empty_position
						new_heuristic = f - g
						modified_column = []
						modified_row = []

						if move == 'left' or move == 'right': #is row False
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
				else:
					new_heuristic = heuristic_func(neighbor, goal, goal_positions, size)
				heapq.heappush(open_set, (g + 1 + new_heuristic, neighbor_tuple, g + 1, move, new_empty_position))
		max_states_in_memory = max(max_states_in_memory, len(open_set))
		states_explored += 1
	return None


def greedy_search(puzzle: List[int], goal: List[int], size: int,heuristic_func: Callable[[List[int], Dict[int, Dict[int, int]], int], int]) -> List[List[int]]:
	"""
	greedy search algorithm for solving puzzles, from A-star
	Parameter:
	puzzle: a list of int (in flattened notation) representing the puzzle to be solved
	goal: a list of int (in flattened notation) representing the final state
	heuristic_func : a function for computing the heuristic cost
	"""
	display_puzzle(goal, size)
	display_puzzle(puzzle, size)

	column_indices = [[k * size + j for k in range(size)] for j in range(size)]
	goal_positions = precompute_goal_positions(goal, size)
	row_goal_indices = precompute_goal_indices(goal_positions, is_row=True)
	col_goal_indices = precompute_goal_indices(goal_positions, is_row=False)
	goal_columns_computed = precompute_goal_columns(goal, size)
	goal_rows_computed = precompute_goal_rows(goal, size)
	manhattan_precomputed = precompute_distance_dictionary(goal_positions, size, manhattan_metric)
	hamming_precomputed = precompute_distance_dictionary(goal_positions, size, hamming_metric)

	open_set = []
	visited = set()
	came_from = {}

	zero_position = puzzle.index(0)
	if heuristic_func is manhattan_distance:
		initial_h = heuristic_func(puzzle, manhattan_precomputed, size)
	elif heuristic_func is hamming_distance:
		initial_h = hamming_distance(puzzle, goal,goal_positions, size)
	elif heuristic_func is linear_conflict_distance:
		initial_h = linear_conflict_distance(puzzle, goal_positions, size)
		tmp = manhattan_distance(puzzle, manhattan_precomputed, size)
		initial_h += tmp
	else:
		print(f"another one {heuristic_func}")
		initial_h = heuristic_func(puzzle, manhattan_precomputed, size)

	start_tuple = tuple(puzzle)
	goal_tuple = tuple(goal)


	heapq.heappush(open_set, (initial_h, start_tuple, None, zero_position)) # I remove g
	visited.add(start_tuple)
	came_from[start_tuple] = None
	states_explored = 0
	max_states_in_memory = 1

	while open_set:
		f, current_tuple, last_move, zero_position = heapq.heappop(open_set)
		current = list(current_tuple)
		states_explored += 1
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

				if heuristic_func is hamming_distance:
					tile_being_moved = current[new_empty_position]
					new_tile_position = new_empty_position
					new_heuristic = f
					new_heuristic -= hamming_precomputed[tile_being_moved][new_tile_position]
					new_heuristic += hamming_precomputed[tile_being_moved][zero_position]

				elif heuristic_func is manhattan_distance or heuristic_func is linear_conflict_distance: #linear-conflict with manhattan
					tile_being_moved = current[new_empty_position]
					new_tile_position = new_empty_position
					new_heuristic = f
					new_heuristic -= manhattan_precomputed[tile_being_moved][new_tile_position]
					new_heuristic += manhattan_precomputed[tile_being_moved][zero_position]

					if heuristic_func is linear_conflict_distance: # optimisé
						tile_being_moved = current[new_empty_position]
						new_tile_position = new_empty_position
						new_heuristic = f

						modified_column = []
						modified_row = []
						if move == 'left' or move == 'right': #is row False

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
							
							""" current_modified_line = get_modified_line(current, size, modified_column, column_indices, False)
							current_modified_line_filtered = filter_column_candidates_tuple(current_modified_line, goal_columns_computed, modified_column, size)
							goal_modified_line = [goal_columns_computed[modified_column[0]]['values'],goal_columns_computed[modified_column[1]]['values']]

							LC_on_line_test_0 = linear_conflict_on_line(current_modified_line_filtered[0], goal_modified_line[0], col_goal_indices)
							LC_on_line_test_1 = linear_conflict_on_line(current_modified_line_filtered[1], goal_modified_line[1], col_goal_indices)
							LC_removal = LC_on_line_test_0 + LC_on_line_test_1

							neighbor_modified_line = get_modified_line(neighbor, size, modified_column, column_indices, False)
							neighbor_modified_line_filtered = filter_column_candidates_tuple(neighbor_modified_line, goal_columns_computed, modified_column, size)

							LC_on_line_test_0 = linear_conflict_on_line(neighbor_modified_line_filtered[0], goal_modified_line[0], col_goal_indices)
							LC_on_line_test_1 = linear_conflict_on_line(neighbor_modified_line_filtered[1], goal_modified_line[1], col_goal_indices)
							LC_add = LC_on_line_test_0 + LC_on_line_test_1 """

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

							""" current_modified_line = get_modified_line(current, size, modified_row, column_indices, True)
							current_modified_line_filtered = filter_row_canditates_tuple(current_modified_line, goal_rows_computed, modified_row, size)
							goal_modified_line = [goal_rows_computed[modified_row[0]]['values'],goal_rows_computed[modified_row[1]]['values']]

							LC_on_line_test_0 = linear_conflict_on_line(current_modified_line_filtered[0], goal_modified_line[0], row_goal_indices)
							LC_on_line_test_1 = linear_conflict_on_line(current_modified_line_filtered[1], goal_modified_line[1], row_goal_indices)
							LC_removal = LC_on_line_test_0 + LC_on_line_test_1

							neighbor_modified_line = get_modified_line(neighbor, size, modified_row, column_indices, True)
							neighbor_modified_line_filtered = filter_row_canditates_tuple(neighbor_modified_line, goal_rows_computed, modified_row, size)

							LC_on_line_test_0 = linear_conflict_on_line(neighbor_modified_line_filtered[0], goal_modified_line[0], row_goal_indices)
							LC_on_line_test_1 = linear_conflict_on_line(neighbor_modified_line_filtered[1], goal_modified_line[1], row_goal_indices)
							LC_add = LC_on_line_test_0 + LC_on_line_test_1 """

							new_heuristic -= LC_removal
							new_heuristic += LC_add
				else:
					new_heuristic = heuristic_func(neighbor, goal, goal_positions, size)
				heapq.heappush(open_set, ( new_heuristic, neighbor_tuple, move, new_empty_position))

		max_states_in_memory = max(max_states_in_memory, len(open_set))
		states_explored += 1
	return None

def ida_star(puzzle: List[int], goal: List[int], size: int,heuristic_func: Callable[[List[int], Dict[int, Dict[int, int]], int], int]) -> List[List[int]]:
	state = PuzzleState(puzzle, goal, size)
	#precomputing everything to be sent to dfs
	goal_positions = precompute_goal_positions(goal, size)
	manhattan_precomputed = precompute_distance_dictionary(goal_positions, size, manhattan_metric)
	hamming_precomputed = precompute_distance_dictionary(goal_positions, size, hamming_metric)
	column_indices = [[k * size + j for k in range(size)] for j in range(size)]
	row_goal_indices = precompute_goal_indices(goal_positions, is_row=True)
	col_goal_indices = precompute_goal_indices(goal_positions, is_row=False)
	goal_columns_computed = precompute_goal_columns(goal, size)
	goal_rows_computed = precompute_goal_rows(goal, size)

	if heuristic_func is manhattan_distance:
		threshold = heuristic_func(puzzle, manhattan_precomputed, size)
	elif heuristic_func is hamming_distance:
		threshold = hamming_distance(puzzle, goal,goal_positions, size)
	elif heuristic_func is linear_conflict_distance:
		threshold = linear_conflict_distance(puzzle, goal_positions, size)
		tmp = manhattan_distance(puzzle, manhattan_precomputed, size)
		threshold += tmp
	else:
		# to modify or remove
		print(f"another one {heuristic_func}")
		threshold = heuristic_func(puzzle, manhattan_precomputed, size)

	f = None

	while True:
		result = dfs(state, puzzle, goal, size, heuristic_func, 0, threshold, [],
			goal_positions,
			manhattan_precomputed,
			hamming_precomputed,
			column_indices,
			row_goal_indices,
			col_goal_indices,
			goal_columns_computed,
			goal_rows_computed,
			f
			)

		if isinstance(result, dict):
			return result
		if result == float('inf'):
			return None
		threshold = result

def dfs(state:PuzzleState,puzzle, goal, size, heuristic_func, g, threshold, path,
		goal_positions,
		manhattan_precomputed,
		hamming_precomputed,
		column_indices,
		row_goal_indices,
		col_goal_indices,
		goal_columns_computed,
		goal_rows_computed,
		f
		):
	"""
	Depth-First Search 
	Parameter:
	state: class PuzzleState
	puzzle: a list of int (in flattened notation) representing the puzzle to be solved
	goal: a list of int (in flattened notation) representing the final state
	heuristic_func : a function for computing the heuristic cost
	g: int, the cost,
	threshold: int, the maximum depth dfs searches
	path: 
	goal_positions:
	manhattan_precomputed:
	hamming_precomputed:
	column_indices:
	row_goal_indices:
	col_goal_indices:
	goal_columns_computed:
	goal_row_computed:
	f: int the heuristics cost
	"""
	
	if f is None:
		if heuristic_func is manhattan_distance:
			h = heuristic_func(puzzle, manhattan_precomputed, size)
		elif heuristic_func is hamming_distance:
			h = hamming_distance(puzzle, goal,goal_positions, size)
		elif heuristic_func is linear_conflict_distance:
			h = linear_conflict_distance(puzzle, goal_positions, size)
			tmp = manhattan_distance(puzzle, manhattan_precomputed, size)
			h += tmp
		else:
			# to modify or remove
			print(f"another one {heuristic_func}")
			h = heuristic_func(puzzle, manhattan_precomputed, size)

		f = g + h

	#pruning
	if f > threshold:
		return f
	# Check if the goal state is reached
	if puzzle == goal:
		return {
			"path": path + [puzzle],
			"moves": len(path),
			"states_explored": len(path),  # Optional: track visited states
			"max_states_in_memory": len(path)  # Optional: track memory usage
		}

	#explore neighbor
	min_cost = float('inf')
	zero_position = puzzle.index(0)

	column_indices = state.column_indices
	row_goal_indices = state.col_goal_indices
	col_goal_indices = state.col_goal_indices

	for neighbor, move in get_neighbors(puzzle, size, zero_position, None):
		if path and neighbor == path[-1]:
			continue  # Skip the immediate parent

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

					current_modified_line_filtered = get_and_filter_line(current, size, modified_column, column_indices, goal_columns_computed, is_row=False)
					goal_modified_line = [goal_columns_computed[modified_column[0]]['values'],goal_columns_computed[modified_column[1]]['values']]
							
					neighbor_modified_line_filtered = get_and_filter_line(neighbor, size, modified_column, column_indices, goal_columns_computed, is_row=False)
							
					LC_removal = linear_conflict_on_multiple_lines(current_modified_line_filtered, goal_modified_line, col_goal_indices)
					LC_add = linear_conflict_on_multiple_lines(neighbor_modified_line_filtered, goal_modified_line, col_goal_indices)
							
					""" current_modified_line = get_modified_line(puzzle, size, modified_column, column_indices, False)# to check... (before current)
					current_modified_line_filtered = filter_column_candidates_tuple(current_modified_line, goal_columns_computed, modified_column, size)
					goal_modified_line = [goal_columns_computed[modified_column[0]]['values'],goal_columns_computed[modified_column[1]]['values']]

					LC_on_line_test_0 = linear_conflict_on_line(current_modified_line_filtered[0], goal_modified_line[0], col_goal_indices)
					LC_on_line_test_1 = linear_conflict_on_line(current_modified_line_filtered[1], goal_modified_line[1], col_goal_indices)
					LC_removal = LC_on_line_test_0 + LC_on_line_test_1

					neighbor_modified_line = get_modified_line(neighbor, size, modified_column, column_indices, False)
					neighbor_modified_line_filtered = filter_column_candidates_tuple(neighbor_modified_line, goal_columns_computed, modified_column, size)

					LC_on_line_test_0 = linear_conflict_on_line(neighbor_modified_line_filtered[0], goal_modified_line[0], col_goal_indices)
					LC_on_line_test_1 = linear_conflict_on_line(neighbor_modified_line_filtered[1], goal_modified_line[1], col_goal_indices)
					LC_add = LC_on_line_test_0 + LC_on_line_test_1 """

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
					
					""" current_modified_line = get_modified_line(puzzle, size, modified_row, column_indices, True)	# to check... (before current)
					current_modified_line_filtered = filter_row_canditates_tuple(current_modified_line, goal_rows_computed, modified_row, size)
					goal_modified_line = [goal_rows_computed[modified_row[0]]['values'],goal_rows_computed[modified_row[1]]['values']]

					LC_on_line_test_0 = linear_conflict_on_line(current_modified_line_filtered[0], goal_modified_line[0], row_goal_indices)
					LC_on_line_test_1 = linear_conflict_on_line(current_modified_line_filtered[1], goal_modified_line[1], row_goal_indices)
					LC_removal = LC_on_line_test_0 + LC_on_line_test_1

					neighbor_modified_line = get_modified_line(neighbor, size, modified_row, column_indices, True)
					neighbor_modified_line_filtered = filter_row_canditates_tuple(neighbor_modified_line, goal_rows_computed, modified_row, size)

					LC_on_line_test_0 = linear_conflict_on_line(neighbor_modified_line_filtered[0], goal_modified_line[0], row_goal_indices)
					LC_on_line_test_1 = linear_conflict_on_line(neighbor_modified_line_filtered[1], goal_modified_line[1], row_goal_indices)
					LC_add = LC_on_line_test_0 + LC_on_line_test_1 """

					new_heuristic -= LC_removal
					new_heuristic += LC_add
		else:
			new_heuristic = heuristic_func(neighbor, goal, goal_positions, size)
		# compute new f
		g_new = g + 1
		f_new = g_new + new_heuristic
		# recursive call
		result = dfs(state,
			neighbor, goal, size, heuristic_func, g_new, threshold,
			path + [puzzle], goal_positions,manhattan_precomputed,
			hamming_precomputed, row_goal_indices, col_goal_indices,
			column_indices, goal_columns_computed, goal_rows_computed,
			f_new
		)
		if isinstance(result, dict):
			return result
		min_cost = min(min_cost, result)
	return min_cost
