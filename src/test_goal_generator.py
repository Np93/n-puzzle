# Générateur du puzzle goal en escargot (ou snail)

def generate_goal_array(size):
    goal_array = [[0 for col in range(size) ] for row in range(size)]
    tile = 1
    tile_max = size * size -1
    left, right, top, bottom = 0, size-1, 0, size-1
    x, y = 0,0 # starting position
    direction = "right"

    while tile <= tile_max:
        goal_array[y][x] = tile
        tile += 1
        if direction == "right":
            if x < right:
                x +=1
            else:
                direction = "down"
                y += 1
                top += 1 # in this way it should not rewrite the line
        elif direction == "down":
            if y < bottom:
                y += 1
            else:
                direction = "left"
                right -= 1
                x -= 1
        elif direction == "left":
            if x > left:
                x -= 1
            else:
                direction = "up"
                bottom -= 1
                y -= 1
        elif direction == "up":
            if y > top:
                y -= 1
            else:
                direction = "right"
                left += 1
                x += 1

    return goal_array
    

def generate_goal(size):
    array = generate_goal_array(size)
    goal_list = [tile for sub in array for tile in sub]
    return goal_list

def generate_goal_linear(size):
    """
    Génère l'état final d'un puzzle linéaire.
    Les nombres sont disposés en ligne de gauche à droite, de haut en bas.
    """
    # Crée une liste de 1 à size*size-1, suivi de 0 pour la dernière case
    goal_list = [i for i in range(1, size * size)] + [0]
    return goal_list