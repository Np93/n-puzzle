# Génération et gestion des puzzles
import random, sys
from goal_generator import generate_goal, generate_goal_array
from utils import display_puzzle
from parser import is_solvable

def generate_puzzle(size, iterations=1000, solvable=True):
	"""
	Génère un puzzle de taille `size` x `size`.
	- Si `solvable` est True, le puzzle est résolvable.
	- Si `solvable` est False, le puzzle est non résolvable.
	"""
	# Crée un puzzle résolvable de base en appliquant des mouvements aléatoires
	
	puzzle = generate_goal(size)
	goal= generate_goal(size)
	
	for i in range(iterations):
		print(f"puzzle n {i}")
		display_puzzle(puzzle, size)
		puzzle = swap_random_tile(puzzle, size)
	print(f"puzzle n {iterations}")
	display_puzzle(puzzle, size)	

	if solvable:
		tmp = is_solvable(puzzle, size, goal)
		if tmp:
			print("IS SOLVABLE")
		else:
			print("IS NOT SOLVABLE")
		return puzzle if is_solvable(puzzle, size, goal) else generate_puzzle(size, iterations, solvable=True)
	else:
		if is_solvable(puzzle, size, goal):
			puzzle = make_unsolvable(puzzle)
		return puzzle

def swap_random_tile(puzzle, size):
	"""
	Échange la case vide (0) avec une case adjacente aléatoire.
	"""
	zero_index = puzzle.index(0)
	possible_swaps = []
	
	# Cases adjacentes
	if zero_index % size > 0:            # Gauche
		possible_swaps.append(zero_index - 1)
	if zero_index % size < size - 1:      # Droite
		possible_swaps.append(zero_index + 1)
	if zero_index >= size:                # Haut
		possible_swaps.append(zero_index - size)
	if zero_index < size * (size - 1):    # Bas
		possible_swaps.append(zero_index + size)
	
	
	swap_idx = random.choice(possible_swaps)
	puzzle[zero_index], puzzle[swap_idx] = puzzle[swap_idx], puzzle[zero_index]
	return puzzle

def make_unsolvable(puzzle):
	"""
	Rend un puzzle non résolvable en échangeant deux tuiles adjacentes (sauf la case vide).
	"""
	if puzzle[0] != 0 and puzzle[1] != 0:
		# Échange deux cases adjacentes au début
		puzzle[0], puzzle[1] = puzzle[1], puzzle[0]
	else:
		# Si les premières cases incluent la case vide, échange deux autres cases
		puzzle[-1], puzzle[-2] = puzzle[-2], puzzle[-1]
	return puzzle


def count_inversions(puzzle, goal):
	"""
	Compte le nombre d'inversions dans le puzzle.
	Une inversion est un couple (i, j) tel que i < j et puzzle[i] > puzzle[j].
	"""
	goal_positions = {value: idx for idx, value in enumerate(goal)}
	
	# Map the current puzzle state to the order of the goal state
	puzzle_order = [goal_positions[tile] for tile in puzzle if tile != 0]
	
	inversions = 0
	
	for i in range(len(puzzle_order) - 1):
		for j in range(i + 1, len(puzzle_order)):
			if puzzle_order[i] > puzzle_order[j]:
				inversions += 1
	return inversions