# Fonctions utilitaires diverses (par exemple, affichage du puzzle)
import sys

def display_puzzle(puzzle, size):
	"""
	Affiche le puzzle dans un format de grille lisible.
	- puzzle : liste représentant l'état actuel du puzzle
	- size : taille du puzzle (ex : 3 pour un puzzle 3x3)
	"""
	max_width = len(str(size * size - 1))  # Largeur maximale des nombres (important pour l'alignement)

	for y in range(size):
		row = [
			str(puzzle[x + y * size]).rjust(max_width) if puzzle[x + y * size] != 0 else " " * max_width
			for x in range(size)
		]
		print(" ".join(row))
	print()  # Saut de ligne pour séparer les états du puzzle

def get_total_size(obj, seen=None):
	"""Recursively finds the memory size of objects, including container contents."""
	size = sys.getsizeof(obj)
	if seen is None:
		seen = set()
	obj_id = id(obj)
	if obj_id in seen:
		return 0  # Avoid double-counting the same object
	# Mark this object as seen
	seen.add(obj_id)

	# If it's a container, sum size of each contained item
	if isinstance(obj, dict):
		size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())
	elif isinstance(obj, (list, tuple, set)):
		size += sum(get_total_size(i, seen) for i in obj)
	
	return size