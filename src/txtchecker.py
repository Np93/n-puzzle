import sys
import os
from typing import List, Tuple

class TXT_Checker:
    def __init__(self, file_path: str, delimiter: List =[' ', '\t'], encoding: str ='utf-8', comment: str = '#') -> None:
        """
		Initialise la classe TXT_Checker, qui s'occupe de tester la conformité du fichier à analyser.

		Paramètres :
		str: file_path, le fichier .str à analyser.
		List: delimiter, le delimiteur entre les données
		str: encoding, le format de l'alphabet
        str: comment, le caractère de début d'un commentaire
        list[str]: lines, le contenu du fichier brute 
        list[str]: cleanedLines, le contenu du fichier purgé des commentaires
        Boolean: error_found, le drapeau indiquant une éventuelle error
        int: puzzle_Size_N, la taille d'une ligne du jeu n-puzzle
        """

        self.file_path = file_path
        self.delimiter = delimiter
        self.encoding = encoding
        self.comment = comment
        self.lines = []
        self.cleanedLines = []
        self.error_found = False

        self.puzzle_Size_N = 0

    def load_file(self) -> None:
        """
        Charge le fichier dans lines
        """
        try:
            with open(self.file_path, 'r') as file:
                self.lines = file.readlines()
            if not self.lines:
                print(f"Error: Empty file:")
                self.error_found = True
                return False
            return True
        except Exception as e:
            print(f"Error: Error opening file: {e}")
            return False
        
    def remove_comment(self) -> None:
        """
        Enlève les commentaires commençant par le caractère 'comment' et crée une liste cleanedLines
        """
        for line_Number, line in enumerate(self.lines, start=1):
            if self.comment in line:
                cleanedLine = line.split(self.comment,1)[0].rstrip()
                self.cleanedLines.append(cleanedLine)
            else:
                self.cleanedLines.append(line.rstrip())
    
    def remove_empty_line(self) -> None:
        """
        Enlèle les lignes vides (dues principallement à des lignes de commentaires)
        """
        for line_number in range(len(self.cleanedLines) - 1, -1, -1):
            if self.cleanedLines[line_number] == '':
                self.cleanedLines.pop(line_number)

    def check_delimiter(self) -> None:
        """
        Vérifie que chaque ligne ne contient que des chiffres espacés d'espace ou de tabs
        """
        forbiden_Characters = 0
        for line_Number, line in enumerate(self.cleanedLines, start=1):
            modified_Line = line.translate(str.maketrans('', '', '0123456789 \t'))
            forbiden_Characters += len(modified_Line)
        if forbiden_Characters > 0:
            self.error_found = True
            print("Error: Non numeric characters were found")
            return False
        return True

    def check_Structure(self) -> None:
        """
        Vérifie la structure du fichier
            la première ligne doit contenir le nombre d'entiers sur une ligne.
            les autres lignes doivent représenter le puzzle avec les conditions suivantes:
                - Chaque nombre est un entier unique et compris entre [0, (N^2 -1)]
            Si aucune erreur de détecté alors le fichier est bien un puzzle
        """
        first_line = self.cleanedLines[0].strip()
        if first_line.isdigit():
            number = int(first_line)
            self.puzzle_Size_N = number
        else:
            self.error_found = True
            return False
        
        expected_value = set(range(number * number))

        for line_Number, line in enumerate(self.cleanedLines[1:], start=1):
            line_Items = line.strip().split()
            if len(line_Items) == self.puzzle_Size_N and all(item for item in line_Items):
                # Try to convert all items to integers
                try:
                    numbers = set(int(item) for item in line_Items)
                except ValueError:
                    print(f"Error: Line {line_Number} contains non-integer values.")
                    self.error_found = True
                    return False
                # Check if the set of numbers has the same size as the list (to detect duplicates)
                if len(numbers) != self.puzzle_Size_N:
                    print(f"Error: Line {line_Number} contains duplicate values.")
                    self.error_found = True
                    return False
                # Check if all numbers are in the valid range
                if not numbers.issubset(expected_value):
                    print(f"Error: Line {line_Number} contains invalid numbers. Expected range: 0 to {number**2 - 1}.")
                    self.error_found = True
                    return False
        
            else:
                print(f"Error: Line {line_Number} contains a different number of integer. Expected number: {number}.")
                self.error_found = True
                return False
        print("All lines are correctly formatted.")
        return True
    
    def is_Puzzle(self) -> None:
        """
        Vérifie que le fichier est bien un puzzle, et si oui crée une version nettoyée (qu'il faut sauvegarder après)
        """
        if self.load_file():
            self.remove_comment()
            self.remove_empty_line()
            self.check_delimiter()
            self.check_Structure()
            return not self.error_found
        else:
            return True
    
    def save_clean_version(self, file_path: str) -> None:
        try:
            with open(file_path, 'w') as file:
                for line in self.cleanedLines[1:]:
                    file.write( line + '\n')
            print(f"Cleaned file saved successfully to {file_path}")
        except Exception as e:
            print(f"Error: Error saving file: {e}")
            
        
def main():
    txt_test = TXT_Checker('puzzle1.txt')
    test = txt_test.is_Puzzle()
    if test:
        txt_test.save_clean_version('puzzle.txt')
    print(test)
    """ txt_test.load_file()
    txt_test.remove_comment()
    #print(txt_test.lines)
    print(txt_test.cleanedLines)
    txt_test.remove_empty_line()
    print(txt_test.cleanedLines)
    txt_test.check_delimiter()
    txt_test.check_Structure()
    print(f"Error found {txt_test.error_found}") 
    print(f"Error found {txt_test.is_Puzzle()}") """

if __name__ == "__main__":
    main()