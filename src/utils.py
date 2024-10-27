# Fonctions utilitaires diverses (par exemple, affichage du puzzle)

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