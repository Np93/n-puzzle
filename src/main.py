# Point d'entrée principal pour exécuter le programme
import yaml
import sys
from pathlib import Path
from src.puzzle_generator import generate_solvable_puzzle
from src.solver import solve_puzzle
from src.parser import parse_puzzle

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

    if puzzle_file:
        puzzle_file_path = Path("..") / puzzle_file
    else:
        puzzle_file_path = None

    # Chargement du puzzle
    if puzzle_file and Path(puzzle_file).is_file():
        print(f"Loading puzzle from {puzzle_file}")
        puzzle = parse_puzzle(puzzle_file)
    else:
        print(f"Generating a random solvable puzzle of size {size} with {iterations} iterations.")
        puzzle = generate_solvable_puzzle(size, iterations)

    # Affichage du puzzle initial
    print("Initial Puzzle State:")
    display_puzzle(puzzle, size)

    # Résolution du puzzle
    print(f"Solving puzzle using {heuristic} heuristic...")
    solution = solve_puzzle(puzzle, size, heuristic)

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