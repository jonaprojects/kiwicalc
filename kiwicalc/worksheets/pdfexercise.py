class PDFExercise:
    """
    This class represents an exercise in a PDF page.
    """
    __slots__ = ['__exercise', '__exercise_type',
                 '__dtype', '__solution', '__number', '__lang']

    def __init__(self, exercise: str, exercise_type: str, dtype: str, solution=None, number=None, lang="en"):
        self.__exercise = exercise
        self.__exercise_type = exercise_type
        self.__dtype = dtype
        self.__solution = solution
        self.__number = number
        self.__lang = lang

    @property
    def exercise(self):
        return self.__exercise

    @property
    def number(self):
        return self.__number

    @number.setter
    def number(self, number: int):
        self.__number = number

    @property
    def dtype(self):
        return self.__dtype

    @property
    def solution(self):
        return self.__solution

    @property
    def has_solution(self):
        return self.__solution is not None

    @property
    def lang(self):
        return self.__lang

    def __str__(self):
        return self.__exercise
