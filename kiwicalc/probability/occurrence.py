class Occurrence:
    def __init__(self, chance: float = 1, identifier: str = ""):
        self._chance = chance
        self._identifier = identifier

    @property
    def chance(self):
        return self._chance

    @chance.setter
    def chance(self, chance: float):
        if 0 <= chance <= 1:
            self._chance = chance
        else:
            warnings.warn(f"Occurrence._chance(setter): failed to set since "
                          f" the probability must be in the range 0 to 1, got {chance} ")

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str):
        self._identifier = identifier

    def intersection(self, *occurrences):
        """
        :param occurrences: a collection of occurrences
        :return: returns their intersection of probabilities of type float
        """
        result = self.chance
        for occurrence in occurrences:
            if isinstance(occurrence, float) or isinstance(occurrence, int):
                occurrence = Occurrence(occurrence)
            result *= occurrence.chance
        return result

    def union(self, *occurrences):
        result = self.chance
        for occurrence in occurrences:
            result += occurrence.chance + result - \
                self.intersection(occurrence)
        return result

    def __str__(self):
        return f"probability: {self.chance} , _identifier: {self.identifier}"

    def __repr__(self):
        return f'Occurrence(_chance={self.chance},_identifier={self.identifier})'
