class PDFPage:
    def __init__(self, title="Worksheet", exercises=None):
        self.__title = title
        if exercises is None:
            self.__exercises = []
        else:
            self.__exercises = exercises

    @property
    def exercises(self):
        return self.__exercises

    @property
    def title(self):
        return self.__title

    def add(self, exercise):
        self.__exercises.append(exercise)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.__exercises):
            raise StopIteration
        temp = self.__index
        self.__index += 1
        return self.__exercises[temp]
