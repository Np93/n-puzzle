# Implémentation des heuristiques admissibles

def manhattan_distance(puzzle, goal, size):
    """
    Calcule la distance de Manhattan entre `puzzle` et `goal`.
    - puzzle : état actuel du puzzle sous forme de liste
    - goal : état final du puzzle sous forme de liste
    - size : taille du puzzle (ex: 3 pour un puzzle 3x3)
    """
    distance = 0
    for i, tile in enumerate(puzzle):
        if tile == 0:
            continue  # Ignore la case vide
        goal_index = goal.index(tile)
        x1, y1 = i % size, i // size
        x2, y2 = goal_index % size, goal_index // size
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def linear_conflict(puzzle, goal, size):
    conflicts = 0
    for row in range(size):
        row_conflicts = []
        for col in range(size):
            tile = puzzle[row * size + col]
            if tile != 0 and (tile - 1) // size == row:  # Dans la même ligne cible
                row_conflicts.append(tile)
        conflicts += count_conflicts(row_conflicts)
    for col in range(size):
        col_conflicts = []
        for row in range(size):
            tile = puzzle[row * size + col]
            if tile != 0 and tile % size == col + 1:  # Dans la même colonne cible
                col_conflicts.append(tile)
        conflicts += count_conflicts(col_conflicts)
    return conflicts * 2  # Chaque conflit ajoute 2 à l'heuristique

def count_conflicts(tiles):
    conflicts = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                conflicts += 1
    return conflicts

def misplaced_tiles(puzzle, goal, size):
    misplaced = sum(1 for i, tile in enumerate(puzzle) if tile != 0 and tile != goal[i])
    return misplaced

def dynamic_misplaced_heuristic(puzzle, goal, size):
    misplaced = misplaced_tiles(puzzle, goal, size)
    # Pondération dynamique qui augmente l'impact de l'heuristique en fonction du nombre de tuiles mal placées
    weight = 1 + (misplaced / (size * size))
    return misplaced * weight

def hamming_distance(puzzle, goal, size):
    """
    Calcule le nombre de tuiles mal placées par rapport à l'état cible.
    """
    return sum(1 for i, tile in enumerate(puzzle) if tile != 0 and tile != goal[i])