# Génération et gestion des puzzles
import random, sys
from test_goal_generator import generate_goal, generate_goal_array
from utils import display_puzzle
from parser import is_solvable

def generate_puzzle(size, iterations=1000, solvable=True):
	"""
	Génère un puzzle de taille `size` x `size`.
	- Si `solvable` est True, le puzzle est résolvable.
	- Si `solvable` est False, le puzzle est non résolvable.
	"""
	# Crée un puzzle résolvable de base en appliquant des mouvements aléatoires
	""" puzzle = list(range(1, size * size)) + [0] 
	print(f"list puzzle {puzzle}") """
	puzzle = generate_goal(size)
	goal= generate_goal(size)
	
	for i in range(iterations):
		print(f"puzzle n {i}")
		display_puzzle(puzzle, size)
		puzzle = swap_random_tile(puzzle, size)
	print(f"puzzle n {iterations}")
	display_puzzle(puzzle, size)	
	
	
	# Vérifie si le puzzle est résolvable et ajuste si nécessaire
	if solvable:
		tmp = is_solvable(puzzle, size, goal)
		if tmp:
			print("IS SOLVABLE")
		else:
			print("IS NOT SOLVABLE")
		# Si le puzzle généré n'est pas résolvable, régénère-le
		return puzzle if is_solvable(puzzle, size, goal) else generate_puzzle(size, iterations, solvable=True)
	else:
		# Si le puzzle est résolvable, le rendre non résolvable
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
	
	# Échange avec une case choisie au hasard parmi les possibilités
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

def count_inversions_ori(puzzle):
	"""
	Compte le nombre d'inversions dans le puzzle.
	Une inversion est un couple (i, j) tel que i < j et puzzle[i] > puzzle[j].
	"""
	inversions = 0
	puzzle_no_zero = [tile for tile in puzzle if tile != 0]
	for i in range(len(puzzle_no_zero) - 1):
		for j in range(i + 1, len(puzzle_no_zero)):
			if puzzle_no_zero[i] > puzzle_no_zero[j]:
				inversions += 1
	return inversions

def count_inversions(puzzle, goal):
	"""
	Compte le nombre d'inversions dans le puzzle.
	Une inversion est un couple (i, j) tel que i < j et puzzle[i] > puzzle[j].
	"""
	goal_positions = {value: idx for idx, value in enumerate(goal)}
	
	# Map the current puzzle state to the order of the goal state
	puzzle_order = [goal_positions[tile] for tile in puzzle if tile != 0]
	# Compte le nombre d'inversions
	inversions = 0
	
	for i in range(len(puzzle_order) - 1):
		for j in range(i + 1, len(puzzle_order)):
			if puzzle_order[i] > puzzle_order[j]:
				inversions += 1
	return inversions

def is_solvable_ori(puzzle, size):
	"""
	Vérifie si le puzzle est résolvable en fonction du nombre d'inversions et de la position de la case vide.
	"""
	inversions = count_inversions(puzzle)
	if size % 2 == 1:
		# Si la taille est impaire, le puzzle est résolvable si le nombre d'inversions est pair
		return inversions % 2 == 0
	else:
		# Si la taille est paire, la solubilité dépend de la ligne de la case vide
		empty_row = puzzle.index(0) // size
		return (inversions + empty_row) % 2 == 1
	
	
def is_solvable_tbc(puzzle, size, goal):
	"""
	Vérifie si le puzzle est résolvable en fonction du nombre d'inversions et de la position de la case vide.
	- puzzle : liste représentant l'état du puzzle
	- size : taille du puzzle (par exemple, 3 pour 3x3)
	Retourne : True si le puzzle est résolvable, False sinon
	"""

	goal_positions = {value: idx for idx, value in enumerate(goal)}
	# print(f"is_solv goal position { goal_positions}")
	# Map the current puzzle state to the order of the goal state
	puzzle_order = [goal_positions[tile] for tile in puzzle if tile != 0]
	"""     print(f"is_solv puzzle order { puzzle_order}")
	display_puzzle(puzzle, size)
	display_puzzle(goal, size) """
	
	# Compte le nombre d'inversions
	inversions = 0
	#puzzle_no_zero = [tile for tile in puzzle if tile != 0]
	for i in range(len(puzzle_order) - 1):
		for j in range(i + 1, len(puzzle_order)):
			if puzzle_order[i] > puzzle_order[j]:
				inversions += 1
	print(f"is_solv there are {inversions} inversions")
	if size % 2 == 1:
		# Si la taille est impaire, le puzzle est résolvable si le nombre d'inversions est pair
		tmp = inversions % 2
		if tmp % 2 != 0:
			print(f" impair not solvable")
			sys.exit()
		return inversions % 2 == 0
	else:
		# Si la taille est paire, la solubilité dépend de la ligne de la case vide
		empty_row = puzzle.index(0) // size
		if (inversions + empty_row) % 2 != 1:
			print(f" pair not solvable")
			sys.exit()
		return (inversions + empty_row) % 2 == 1