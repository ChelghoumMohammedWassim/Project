import random as rnd

class Genome:
    def __init__(self, selection, operation=rnd.randint(1, 3), fit=0):
        self.selection: list[int] = selection
        self.fit= fit
        self.operation = operation

    def print(self):
        return (self.selection, self.operation)