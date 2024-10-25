# Nom de votre programme Python
NAME := n_puzzle

# Commande pour supprimer les fichiers
RM := rm -rf

# Cible principale pour exécuter le programme
all: ${NAME}

# Exécute le programme principal sans réinstaller les dépendances
${NAME}:
	@poetry run python src/main.py

# Nettoyage des fichiers temporaires (caches Python)
clean:
	@${RM} __pycache__
	@${RM} .mypy_cache
	@${RM} .pytest_cache

# Supprime toutes les installations de dépendances Poetry et les caches
fclean: clean
	@${RM} poetry.lock
	@poetry cache clear --all pypi  # Vide le cache Poetry (optionnel)

# Nettoie et réinstalle les dépendances, puis exécute le programme
re: fclean
	@poetry install
	@make all