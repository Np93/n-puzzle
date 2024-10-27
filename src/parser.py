# Lecture et parsing des fichiers .txt contenant les puzzles
import sys

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
    
    if not is_solvable(puzzle, size):
        print("Le puzzle n'est pas résolvable.")
        sys.exit(1)
    
    return puzzle, size

def is_solvable(puzzle, size):
    """
    Vérifie si le puzzle est résolvable en fonction du nombre d'inversions et de la position de la case vide.
    - puzzle : liste représentant l'état du puzzle
    - size : taille du puzzle (par exemple, 3 pour 3x3)
    Retourne : True si le puzzle est résolvable, False sinon
    """
    # Compte le nombre d'inversions
    inversions = 0
    puzzle_no_zero = [tile for tile in puzzle if tile != 0]
    for i in range(len(puzzle_no_zero) - 1):
        for j in range(i + 1, len(puzzle_no_zero)):
            if puzzle_no_zero[i] > puzzle_no_zero[j]:
                inversions += 1
    
    if size % 2 == 1:
        # Si la taille est impaire, le puzzle est résolvable si le nombre d'inversions est pair
        return inversions % 2 == 0
    else:
        # Si la taille est paire, la solubilité dépend de la ligne de la case vide
        empty_row = puzzle.index(0) // size
        return (inversions + empty_row) % 2 == 1