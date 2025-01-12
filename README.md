# N-puzzle

Le but de ce projet est de résoudre le jeu du N-puzzle (« taquin » en français) en utilisant l'algorithme de recherche A* ou l'une de ses variantes.

## Description

Le N-puzzle est un casse-tête constitué d'une grille de taille `N x N` contenant des tuiles numérotées de 1 à `(N^2 - 1)` et une case vide. L'objectif est de réorganiser les tuiles pour atteindre une configuration cible, en déplaçant une tuile à la fois dans la case vide.

### Algorithmes disponibles

1. **A*** : Trouve toujours le chemin optimal en combinant coût déjà parcouru et coût estimé vers la solution.
   - Avantage : Assure une solution optimale si l'heuristique est admissible.
   
2. **Greedy Best-First Search** : Ne considère que le coût estimé vers la solution.
   - Avantage : Plus rapide que A* dans de nombreux cas, mais peut ne pas donner une solution optimale.

3. **Uniform-Cost Search** : Explore les nœuds en fonction du coût déjà parcouru, sans prendre en compte l'heuristique.
   - Avantage : Utile si aucune bonne heuristique n'est disponible.

4. **IDA*** (Iterative Deepening A*) : Combinaison de la recherche en profondeur et d'A*, avec une faible consommation de mémoire.
   - Avantage : Plus économique en mémoire tout en restant efficace pour des puzzles de grande taille.

### Résolutions possibles

- **Snail** (escargot) : Les tuiles sont organisées dans un motif en spirale partant du coin supérieur gauche.
(3x3)
```
1 2 3
8 0 4
7 6 5
```
- **Linear** (linéaire) : Les tuiles sont organisées de gauche à droite et de haut en bas, formant une grille ordonnée.
(3x3)
```
1 2 3
4 5 6
7 8 0
```
---

## Fonctionnalités

- Résolution de puzzles de taille personnalisable (3x3, 4x4, 15x15, etc.).
- Algorithmes disponibles :
  - A*
  - Greedy Best-First Search
  - Uniform-Cost Search
  - IDA*
- Support de différentes heuristiques :
  - Distance de Manhattan
  - hamming
  - linear_conflict
  - dynamic_misplace
- Lecture d'une configuration initiale depuis un fichier texte.
- Génération aléatoire de puzzles résolubles ou pas.
- Résolutions **snail** et **linear**.
- Visualisation de la résolution pas à pas.
- Gestion des performances et du temps d'exécution.

---

## Prérequis

- poetry

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/Np93/n-puzzle.git
   cd n-puzzle
   ```
2. installer les dependance poetry :
    ```bash
    poetry install
    ```
3. configurer selon votre volonter le fichier config.yaml
4. lancer le programme :
    ```bash
    poetry run python3 src/main.py
    ```

## Contributeurs
- Np93[https://github.com/Np93]
- jsollet[https://github.com/jsollet]

## Licence
Ce projet est sous licence MIT. Vous êtes libre d'utiliser, modifier et distribuer ce projet, tant que vous incluez une copie de la licence. Voir le fichier LICENSE[https://github.com/Np93/n-puzzle/blob/main/LICENSE] pour plus de détails.

## Sujet
pour plus d'information vous trouverez le sujet ici[https://github.com/Np93/n-puzzle/blob/main/sujet/sujet.pdf]