import yaml
import sys
import os
import cProfile
from pathlib import Path
from puzzle_generator import generate_puzzle, count_inversions
from generate_linear_puzzle import generate_linear_puzzle
from solver import solve_puzzle
from utils import display_puzzle
from parser import parse_puzzle
from goal_generator import generate_goal, generate_goal_linear
from txtchecker import TXT_Checker

def load_config(config_path="config.yaml"):
	try:
		with open(config_path, "r") as f:
			config = yaml.safe_load(f)
		return config
	except FileNotFoundError:
		print(f"Configuration file {config_path} not found.")
		sys.exit(1)

def main():
	config = load_config()
	
	size = config.get("size", 3) or 3
	iterations = config.get("iterations", 10) or 10
	heuristic = config.get("heuristic", "manhattan") or "manhattan"
	puzzle_file = config.get("file")
	solvable = config.get("solvable", True)
	algorithm = config.get("algorithm", "A-star") or "A-star"
	snail = config.get("snail", True)

	if size < 3:
		print("Can't generate and resolvate a puzzle with size lower than 2. It says so in the help. Dummy.")
		return

	if puzzle_file:
		puzzle_file_path = Path("..") / puzzle_file
	else:
		puzzle_file_path = None

	if snail:
		# Chargement du puzzle
		if puzzle_file and Path(puzzle_file).is_file():
			if os.stat(puzzle_file).st_size > 0:
				puzzle_test = TXT_Checker(puzzle_file)
				test = puzzle_test.is_Puzzle()
				if test:
					puzzle_test.save_clean_version(puzzle_file)
					#print(f"le fichier {puzzle_file} est ok ? {test}")
				if puzzle_file and Path(puzzle_file).is_file():
					print(f"Loading puzzle from {puzzle_file}")
					try:
						puzzle, size = parse_puzzle(puzzle_file)
					except ValueError as e:
						print(f"Erreur lors du parsing du fichier : {e}")
						sys.exit(1)
			else:
				print(f"Generating a random solvable puzzle of size {size} with {iterations} iterations.")
				puzzle = generate_puzzle(size, iterations, solvable)
		else:
			print(f"Generating a random solvable puzzle of size {size} with {iterations} iterations.")
			puzzle = generate_puzzle(size, iterations, solvable)
	else:
		puzzle = generate_linear_puzzle(size, iterations, solvable)

	if snail:
		tmp = generate_goal(size)
	else:
		tmp = generate_goal_linear(size)
	inversions = count_inversions(puzzle, tmp)
	# Résolution du puzzle avec profiling
	print(f"Solving puzzle using the {algorithm} with {heuristic} heuristic")
	#cProfile.runctx('solve_puzzle(algorithm,puzzle, size, heuristic, inversions)', globals(), locals())
	solution = solve_puzzle(algorithm, puzzle, size, heuristic, inversions, snail)

	if solution:
		print("Puzzle solved!")
		print("Solution path:")
		for step in solution['path']:
			display_puzzle(step, size)
		print(f"Number of moves: {solution['moves']}")
		print(f"Total states explored: {solution['states_explored']}")
		print(f"Maximum states in memory: {solution['max_states_in_memory']}")
	else:
		print("No solution found.")
	sys.exit()

if __name__ == "__main__":
	main()