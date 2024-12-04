import sys
from utils import display_puzzle

def filter_row_candidates(puzzle_rows, goal_positions, modified_line_index):
	filtered_rows = []
	i = 0
	for index in modified_line_index:
		row = puzzle_rows[i]
		filtered_row = [tile for tile in row if tile != 0 and goal_positions[tile][1] == index]
		filtered_rows.append(filtered_row)
		i +=1
	return filtered_rows	

def filter_row_canditates_tuple1(puzzle_rows, goal_tuple, modified_line_index, size):
	"""
	Filters the rows based on the tiles present in the corresponding goal row.
	- puzzle_rows: List of rows from the puzzle.
	- goal_tuple: Tuple representing the goal state of the puzzle.
	- modified_line_index: Indices of rows to filter (list of row indices).
	- size: Size of the puzzle (e.g., 4 for a 4x4 puzzle).
	"""
	filtered_rows = []
	#print(f"-tup-> index {modified_line_index}")
	i = 0
	for index in modified_line_index:
		row = puzzle_rows[i]
		
		goal_row = [goal_tuple[j + index * size] for j in range(size)]

		filtered_row = [tile for tile in row if tile != 0 and tile in goal_row]
		filtered_rows.append(filtered_row)
		i += 1
	return filtered_rows

def filter_column_candidates(puzzle_columns, goal_positions, modified_line_index): # a tester, semble ok
	filtered_columns = []
	#print(f"-----> index {modified_line_index}")
	
	i = 0
	for index in modified_line_index:
		column = puzzle_columns[i]
		#print(f"Processing column {column} for index {index}")

		filtered_column = [tile for tile in column if tile != 0 and goal_positions[tile][0] == index]
		filtered_columns.append(filtered_column)
		i +=1
	return filtered_columns

def filter_column_candidates_tuple1(puzzle_columns, goal_tuple, modified_line_index, size):
	filtered_columns = []
	goal_columns = {}
	for index in modified_line_index:
		goal_columns[index] = {goal_tuple[index + j * size] for j in range(size)}
	#print(f"-tup-> index {modified_line_index}")
	i = 0
	for index in modified_line_index:
		column = puzzle_columns[i]
		
		#goal_column = [goal_tuple[index + j * size] for j in range(size)]
		filtered_column = [tile for tile in column if tile != 0 and tile in goal_columns[index]]
		#filtered_column = [tile for tile in column if tile != 0 and tile in goal_column]
		filtered_columns.append(filtered_column)
		i +=1
	return filtered_columns

def filter_column_candidates_tuple(puzzle_columns, goal_columns, modified_line_index, size):
	filtered_columns = []
	#print(f"goal columns in filter... {goal_columns}")
	#print(f"-tup-> index {modified_line_index}")
	i = 0
	for index in modified_line_index:
		column = puzzle_columns[i]
		filtered_column = [tile for tile in column if tile != 0 and tile in goal_columns[index]['set']]
		filtered_columns.append(filtered_column)
		i +=1
	return filtered_columns

def filter_row_canditates_tuple(puzzle_rows, goal_rows, modified_line_index, size):
	"""
	Filters the rows based on the tiles present in the corresponding goal row.
	- puzzle_rows: List of rows from the puzzle.
	- goal_tuple: Tuple representing the goal state of the puzzle.
	- modified_line_index: Indices of rows to filter (list of row indices).
	- size: Size of the puzzle (e.g., 4 for a 4x4 puzzle).
	"""
	filtered_rows = []
	#print(f"-tup-> index {modified_line_index}")
	i = 0
	for index in modified_line_index:
		row = puzzle_rows[i]
		filtered_row = [tile for tile in row if tile != 0 and tile in goal_rows[index]['set']]
		filtered_rows.append(filtered_row)
		i += 1
	return filtered_rows


def count_linear_conflict(filtered_lines, goal_positions, state_positions, is_row=True):
	total_conflict = 0
	for line in filtered_lines:
		conflicts = 0
		for i in range(len(line)):
			for j in range(i + 1, len(line)):
				tile1, tile2 = line[i], line[j]
				if tile1 == 0 or tile2 == 0:  # Skip empty tile
					continue
				g_pos1 = goal_positions[tile1][0 if is_row else 1] # inversion of (x,y)
				g_pos2 = goal_positions[tile2][0 if is_row else 1]
				
				s_pos1 = state_positions[tile1][0 if is_row else 1]
				s_pos2 = state_positions[tile2][0 if is_row else 1]
				#print(f"*ori*--->i,j {i,j} gpos1 {g_pos1} : gpos2 {g_pos2}  ------ s_pos1 {s_pos1} : s_pos2 {s_pos2}")
				if g_pos1 < g_pos2 and s_pos1 > s_pos2:
					conflicts += 1
				elif g_pos1 > g_pos2 and s_pos1 < s_pos2:
					conflicts += 1
		total_conflict += conflicts
	return 2 * total_conflict

def get_modified_line1(puzzle, size, modified_line_index, is_row= True):
	modified_line = []
	for index in modified_line_index:
		if is_row:
			rows = puzzle[index * size : (index + 1) * size]
			modified_line.append(rows)
			#print(modified_line)
		else:
			colums = [puzzle[k * size + index] for k in range(size)] # bordelique
			
			modified_line.append(colums)
	return modified_line

def get_modified_line(puzzle, size, modified_line_index, column_indice, is_row=True):
	modified_line = []
	#modified_line1 = []
	# If it's for rows
	
	if is_row:
		# Use list slicing for row extraction
		for index in modified_line_index:
			modified_line.append(puzzle[index * size:(index + 1) * size])
	
	# If it's for columns
	else:
		
		# Use list comprehension to generate columns
		for index in modified_line_index:
			#modified_line1.append([puzzle[k * size + index] for k in range(size)])
			#print(f" before error Index: {index}, column_indices: {column_indice[index]}")
			try:
				modified_line.append([puzzle[k] for k in column_indice[index]])
			except Exception as e:
				print(f"Error: {e}")
				print(f"Index: {index}, column_indices: {column_indice[index]}")
				print(f"Puzzle: {puzzle}")
				sys.exit()
			""" if modified_line != modified_line1:
				print(f"error get_modified_line")
				sys.exit() """
			
			""" print(f"puzzle type {type(puzzle)} and \nvalue {puzzle}")
			print(f"column_indices: {column_indice}") """
	return modified_line





def count_linear_conflict_tuple(filtered_lines, goal_tuple, state_tuple, is_row=True):
	total_conflict = 0
	size = 4
	for line in filtered_lines:
		conflicts = 0
		for i in range(len(line)):
			for j in range(i + 1, len(line)):
				tile1, tile2 = line[i], line[j]
				if tile1 == 0 or tile2 == 0:  # Skip empty tile
					continue
				if is_row:
					g_pos1 = goal_tuple.index(tile1) % size
					g_pos2 = goal_tuple.index(tile2) % size #semble ok
					s_pos1 = state_tuple.index(tile1) % size #a verifier 
					s_pos2 = state_tuple.index(tile2) % size
				else:
					g_pos1 = goal_tuple.index(tile1) // size
					g_pos2 = goal_tuple.index(tile2) // size
					s_pos1 = state_tuple.index(tile1) // size
					s_pos2 = state_tuple.index(tile2) // size
				
				print(f"*tup*--->i,j {i,j} gpos1 {g_pos1} : gpos2 {g_pos2}  ------ s_pos1 {s_pos1} : s_pos2 {s_pos2}")
				""" g_pos1 = goal_positions[tile1][0 if is_row else 1] # inversion of (x,y)
				g_pos2 = goal_positions[tile2][0 if is_row else 1]

				s_pos1 = state_positions[tile1][0 if is_row else 1]
				s_pos2 = state_positions[tile2][0 if is_row else 1] """
				""" if g_pos1 < g_pos2 and s_pos1 > s_pos2:
					conflicts += 1
				elif g_pos1 > g_pos2 and s_pos1 < s_pos2:
					conflicts += 1 """
		total_conflict += conflicts
	return 2 * total_conflict

def compute_LC_on_line_tuple(puzzle, modified_line_index, goal_tuple,state_tuple, size, current_state= True, is_row= True):
	""" print(f"test compute_lC goal position {goal_positions}")
	print(f"test compute_lC state position {state_positions}")
	print(f"test compute_lC modified line index {modified_line_index}") """
	# en cours
	modified_line = get_modified_line(puzzle, size, modified_line_index, is_row)
	if is_row:
		filtered_row = filter_row_canditates_tuple(modified_line, goal_tuple, modified_line_index, size)
		#print(f"filtered row {filtered_row}")
		total = count_linear_conflict_tuple(filtered_row, goal_tuple, state_tuple, is_row)
	else:
		filtered_columns =filter_column_candidates_tuple(modified_line, goal_tuple, modified_line_index, size)
		#print(f"filtered column {filtered_columns}")
		total = count_linear_conflict_tuple(filtered_columns, goal_tuple, state_tuple, is_row)
	return total

def linear_conflict_on_line_ok(current_filtered, goal, goal_position, is_row = True):
	"""
	To test
	Compute the linear conflict for a single row or column.

	:param current: List of current tile positions in a row/column.
	:param goal: List of goal tile positions in the same row/column.
	:return: Linear conflict distance.
	"""
	conflict_count = 0
	size_current_line = len(current_filtered)
	for index in range(size_current_line):
		tile_current = current_filtered[index]
		if current_filtered[index] == goal[index]:
			#print(f"invariant[{index}] {current_filtered[index]} {goal[index]}")
			continue
		
		#print(f"tile {tile_current}")
		goal_position = goal.index(tile_current)
		#print(f"lcol goal.index{tile_current} is {goal_position}")
		for j in range(index + 1, size_current_line):
			other_tile = current_filtered[j]
			other_goal_position = goal.index(other_tile)
			#print(f"for tile_current {tile_current} and other tile {other_tile} ")
			if (goal_position < other_goal_position and index > j) or (goal_position > other_goal_position and index < j):
				conflict_count += 1
	#sys.exit()
	return 2*conflict_count



def linear_conflict_on_line(current_filtered, goal, goal_indices):
	"""
	To test
	Compute the linear conflict for a single row or column.

	:param current: List of current tile positions in a row/column.
	:param goal: List of goal tile positions in the same row/column.
	:return: Linear conflict distance.
	"""
	conflict_count = 0
	size_current_line = len(current_filtered)

	for index in range(size_current_line):
		tile_current = current_filtered[index]
		if tile_current == goal[index]:
			continue
		""" if current_filtered[index] == goal[index]:
			#print(f"invariant[{index}] {current_filtered[index]} {goal[index]}")
			continue """
		
		#print(f"tile {tile_current}")
		#tmp = goal.index(tile_current)
		#goal_position_current = goal_positions[tile_current][0 if is_row else 1] # to check the last part
		#goal_position_current = goal_indices[current_filtered[index]]
		goal_position_current = goal_indices[tile_current]
		""" if tmp != goal_position_current:
			print(f"error old version {tmp} new {goal_position_current}")
			sys.exit() """
		
		for j in range(index + 1, size_current_line):
			other_tile = current_filtered[j]
			#goal_position_other = goal_positions[other_tile][0 if is_row else 1]
			goal_position_other = goal_indices[other_tile]
			#goal_position_other = goal_indices[current_filtered[j]]
			
			if (goal_position_current < goal_position_other and index > j) or (goal_position_current > goal_position_other and index < j):
				conflict_count += 1
	
	return 2*conflict_count