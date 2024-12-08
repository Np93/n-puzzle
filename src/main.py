# Point d'entrée principal pour exécuter le programme
import yaml
import sys
from pathlib import Path
from puzzle_generator import generate_puzzle, count_inversions
from solver import solve_puzzle
from utils import display_puzzle
from parser import parse_puzzle

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
    
    size = config.get("size", 3)
    iterations = config.get("iterations", 10)
    heuristic = config.get("heuristic", "manhattan")
    puzzle_file = config.get("file")
    solvable = config.get("solvable", True)

    if size < 3:
        print("Can't generate and resolvate a puzzle with size lower than 2. It says so in the help. Dummy.")
        return

    if puzzle_file:
        puzzle_file_path = Path("..") / puzzle_file
    else:
        puzzle_file_path = None

    # Chargement du puzzle
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

    # Affichage du puzzle initial
    print("Initial Puzzle State:")
    display_puzzle(puzzle, size)

    inversions = count_inversions(puzzle)
    # Résolution du puzzle
    print(f"Solving puzzle using {heuristic} heuristic...")
    solution = solve_puzzle(puzzle, size, heuristic, inversions)

    # Affichage des résultats
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

if __name__ == "__main__":
    main()