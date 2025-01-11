import random

def generate_linear_puzzle(size, iterations=1000, solvable=True):
    """
    Génère un puzzle de taille `size` x `size`.
    - Si `solvable` est True, le puzzle est résolvable.
    - Si `solvable` est False, le puzzle est non résolvable.
    """
    # Crée un puzzle résolvable de base en appliquant des mouvements aléatoires
    puzzle = list(range(1, size * size)) + [0]
    for _ in range(iterations):
        puzzle = swap_random_tile(puzzle, size)
    
    # Vérifie si le puzzle est résolvable et ajuste si nécessaire
    if solvable:
        # Si le puzzle généré n'est pas résolvable, régénère-le
        return puzzle if is_solvable(puzzle, size) else generate_puzzle(size, iterations, solvable=True)
    else:
        # Si le puzzle est résolvable, le rendre non résolvable
        if is_solvable(puzzle, size):
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

def count_inversions(puzzle):
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

def is_solvable(puzzle, size):
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