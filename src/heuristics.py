# Implémentation des heuristiques admissibles
import cProfile
manhattan_cache = {}

def manhattan_metric(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def hamming_metric(x1,y1,x2,y2):
    if x1 == x2 and y1 == y2:
        return 0
    else:
        return 1

def manhattan_distance(puzzle, manhattan_precomputed, size):
    total_distance = 0
    for position, tile in enumerate(puzzle):
        if tile != 0:
            total_distance += manhattan_precomputed[tile][position]
    return total_distance

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

# not done
def misplaced_tiles(puzzle, goal, size):
    misplaced = sum(1 for i, tile in enumerate(puzzle) if tile != 0 and tile != goal[i])
    return misplaced

def dynamic_misplaced_heuristic(puzzle, goal, size):
    misplaced = misplaced_tiles(puzzle, goal, size)
    # Pondération dynamique qui augmente l'impact de l'heuristique en fonction du nombre de tuiles mal placées
    weight = 1 + (misplaced / (size * size))
    return misplaced * weight

def hamming_distance(puzzle, goal, goal_positions, size):
    """
    Calcule le nombre de tuiles mal placées par rapport à l'état cible.
    """
    distance = 0
    for idx, tile in enumerate(puzzle):
        if tile != 0:  # Skip the blank tile
            goal_x, goal_y = goal_positions[tile]
            current_x, current_y = idx % size, idx // size
            if (goal_x, goal_y) != (current_x, current_y):
                distance += 1
    
    return distance

# test pour linear conflict (a row) 
def linear_conflict_row(current_row, goal_row):
    relevant_tiles = [tile for tile in current_row if tile in goal_row]

    goal_indices = [goal_row.index(tile) for tile in relevant_tiles]

    inversions = 0
    for i in range(len(goal_indices)):
        for j in range(i+ 1, len(goal_indices)):
            if goal_indices[i] > goal_indices[j]:
                inversions += 1
    
    return inversions * 2


def linear_conflict_distance(puzzle, goal_positions, size):
    def get_rows_and_columns(puzzle_list, size): # ok
        # Extract rows
        rows = [puzzle[i * size : (i + 1) * size] for i in range(size)]

        # Extract columns
        columns = [[puzzle[k * size + j] for k in range(size)] for j in range(size)]
        return rows, columns
    
    # attn (column, row) so goal_positions[tile][1] instead of goal_positions[tile][0]
    def filter_row_candidates(puzzle_rows, goal_positions):
        filtered_rows = []
        
        for row_index, row in enumerate(puzzle_rows):
            
            filtered_row = [tile for tile in row if tile != 0 and goal_positions[tile][1] == row_index]
            filtered_rows.append(filtered_row)
        return filtered_rows
    
    # attn (column, row) so goal_positions[tile][0] instead of goal_positions[tile][1]
    def filter_column_candidates(puzzle_columns, goal_positions):
        filtered_columns = []
        for col_index, column in enumerate(puzzle_columns):
            filtered_column = [tile for tile in column if tile != 0 and goal_positions[tile][0] == col_index]
            filtered_columns.append(filtered_column)
        return filtered_columns
    

    def count_conflicts(tiles, goal_positions, is_row=True):
        """
        Count the number of linear conflicts in a given row or column.
        """
        conflicts = 0
        
        for index, line in enumerate(tiles):
            """ print(f"**tiles {tiles} index {index} line {line}")
            print(f"goal position[{index}] {goal_positions[index]}") """
            for i in range(len(line)):
                for j in range(i + 1, len(line)):
                    
                    tile1 = tiles[index][i]
                    tile2 = tiles[index][j]
                    
                    if tile1 == 0 or tile2 == 0:
                        continue
                    if is_row: #attn order
                        goal_pos1 = goal_positions[tile1][0]
                        goal_pos2 = goal_positions[tile2][0]
                    else:
                        goal_pos1 = goal_positions[tile1][1]
                        goal_pos2 = goal_positions[tile2][1]
                    
                    # Check for row/column conflict
                    if is_row:
                        if goal_pos1 > goal_pos2:
                            conflicts += 1
                    else:
                        if goal_pos1 > goal_pos2:
                            conflicts += 1
        return conflicts

    
    puzzle_rows, puzzle_columns = get_rows_and_columns(puzzle, size) 
    filtered_rows = filter_row_candidates(puzzle_rows, goal_positions)
    filtered_columns = filter_column_candidates(puzzle_columns, goal_positions)
    row_conflicts = count_conflicts(filtered_rows, goal_positions, True)
    column_conflicts = count_conflicts(filtered_columns, goal_positions, False)
    return 2 * (row_conflicts + column_conflicts)