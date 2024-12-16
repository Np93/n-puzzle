# Lecture et parsing des fichiers .txt contenant les puzzles
import sys
from test_goal_generator import generate_goal, generate_goal_array
from utils import display_puzzle

def parse_puzzle(puzzle_file):
    """
    Lit un fichier de puzzle et le retourne sous forme de liste si le format est valide.
    - puzzle_file : chemin du fichier contenant le puzzle
    Retourne : liste représentant l'état du puzzle (ex: [1, 2, 3, 4, 5, 6, 7, 8, 0])
    Lève une ValueError si le format est incorrect.
    """
    puzzle = []
    
    # Lecture du fichier et vérification du contenu
    with open(puzzle_file, "r") as f:
        for line in f:
            row = line.strip().split()
            # Vérification que chaque élément est un entier
            if not all(cell.isdigit() for cell in row):
                raise ValueError("Le puzzle contient des éléments non numériques.")
            puzzle.extend(int(cell) for cell in row)
    
    # Vérification de la taille du puzzle (doit être carré)
    length = len(puzzle)
    size = int(length ** 0.5)
    if size * size != length:
        raise ValueError("Le puzzle n'est pas au bon format carré.")
    
    # Vérification de la plage de valeurs attendues et des doublons
    required_numbers = set(range(size * size))
    if set(puzzle) != required_numbers:
        raise ValueError(f"Le puzzle doit contenir chaque nombre de 0 à {size * size - 1} sans doublons.")
    
    goal = generate_goal(size)
    if not is_solvable(puzzle, size, goal):
        print("parser: Le puzzle n'est pas résolvable.")
        sys.exit(1)
    
    return puzzle, size


    

def is_solvable(puzzle, size, goal):
    """
    Vérifie si le puzzle est résolvable en fonction du nombre d'inversions et de la position de la case vide.
    - puzzle : liste représentant l'état du puzzle
    - size : taille du puzzle (par exemple, 3 pour 3x3)
    Retourne : True si le puzzle est résolvable, False sinon
    """

    goal_positions = {value: idx for idx, value in enumerate(goal)}
    puzzle_order = [goal_positions[tile] for tile in puzzle if tile != 0]
    
    inversions = 0
    for i in range(len(puzzle_order) - 1):
        for j in range(i + 1, len(puzzle_order)):
            if puzzle_order[i] > puzzle_order[j]:
                inversions += 1
    print(f"there are {inversions} inversions")
    if size % 2 == 1:
        # Si la taille est impaire, le puzzle est résolvable si le nombre d'inversions est pair
        return inversions % 2 == 0
    elif size % 4 == 0:
        # Si la taille est paire, la solubilité dépend de la ligne de la case vide
        empty_row = puzzle.index(0) // size
        row_from_bottom = (size - 1) - empty_row
        return (inversions +row_from_bottom) % 2 == 1 # avant empty row
    else:
        empty_row = puzzle.index(0) // size
        row_from_bottom = (size - 1) - empty_row
        return (inversions +row_from_bottom) % 2 == 0 # avant empty row
    
def is_solvable_gpt(puzzle, size, goal):
    """
    Vérifie si le puzzle est solvable en utilisant l'ordre en spirale pour la configuration finale.
    - puzzle : liste représentant l'état du puzzle
    - size : taille du puzzle (par exemple, 4 pour 4x4)
    Retourne : True si le puzzle est résolvable, False sinon
    """
    # Générer la configuration finale en spirale
    

    # Calcul des positions finales des tuiles
    goal_positions = {value: idx for idx, value in enumerate(goal)}

    # Ordre du puzzle basé sur les indices finaux
    puzzle_order = [goal_positions[tile] for tile in puzzle if tile != 0]

    # Calcul des inversions
    inversions = 0
    for i in range(len(puzzle_order) - 1):
        for j in range(i + 1, len(puzzle_order)):
            if puzzle_order[i] > puzzle_order[j]:
                inversions += 1

    # Localisation de la case vide
    empty_row = puzzle.index(0) // size
    row_from_bottom = (size - 1) - empty_row  # Distance depuis le bas

    # Condition de solvabilité
    if size % 2 == 1:
        # Taille impaire : nombre d'inversions pair
        return inversions % 2 == 0
    else:
        # Taille paire : (inversions + distance case vide) pair
        return (inversions + row_from_bottom) % 2 == 0
