class PDFWorksheet:
    __slots__ = ['__pages', '__ordered', '__current_page',
                 '__lines', '__title', '__num_of_exercises']

    def __init__(self, title="Worksheet", ordered=True):
        self.__pages = [PDFPage(title=title)]
        self.__ordered = ordered
        self.__current_page = self.__pages[0]
        self.__lines = [[]]
        self.__title = title
        self.__num_of_exercises = 0

    @property
    def num_of_pages(self):
        return len(self.__pages)

    @property
    def pages(self):
        return self.__pages

    def del_last_page(self):
        if len(self.__pages):
            del self.__pages[-1]

    @property
    def current_page(self):
        return self.__current_page

    def add_exercise(self, exercise):
        self.__num_of_exercises += 1
        self.__current_page.add(exercise)
        if '\n' in exercise.__str__():
            lines = exercise.__str__().split('\n')
            if self.__ordered:
                exercise.number = self.__num_of_exercises
                self.__lines[-1].append(f"{exercise.number}.    {lines[0]}")
            else:
                self.__lines[-1].append(lines[0])

            for i in range(1, len(lines)):
                self.__lines[-1].append(lines[i])

            self.__lines[-1].append("")  # separator
        else:
            if self.__ordered:
                exercise.number = self.__num_of_exercises
                self.__lines[-1].append(
                    f"{exercise.number}.    {exercise.__str__()}")
            else:
                self.__lines[-1].append(exercise.__str__())

    def end_page(self):
        if any(exercise.has_solution for exercise in self.__current_page.exercises):
            solutions_string = []
            for index, exercise in enumerate(self.__current_page.exercises):
                if exercise.solution is None:
                    continue
                if not isinstance(exercise.solution, (int, float, str)) and isinstance(exercise.solution, Iterable):
                    str_solution = ",".join(str(solution)
                                            for solution in exercise.solution)
                    if self.__ordered:
                        solutions_string.append(
                            f"{exercise.number}.    {str_solution}")
                    else:
                        solutions_string.append(str_solution)
                else:
                    if not isinstance(exercise.solution, str):
                        str_solution = str(exercise.solution)
                    else:
                        str_solution = exercise.solution

                    if "\n" in str_solution:
                        lines = exercise.solution.split("\n")
                        solutions_string.append(
                            f"{exercise.number}. {lines[0]}" if self.__ordered else f"{lines[0]}")
                        for j in range(1, len(lines)):
                            solutions_string.append(lines[j])
                        solutions_string.append("")
                    else:
                        if self.__ordered:
                            solutions_string.append(
                                f"{exercise.number}.    {exercise.solution}")
                        else:
                            solutions_string.append(f"{exercise.solution}")

            self.__pages.append(
                PDFPage(title="Solutions", exercises=solutions_string))
            self.__lines.append(solutions_string)

    def next_page(self, title=None):
        if title is None:
            title = self.__title
        self.__pages.append(PDFPage(title))
        self.__current_page = self.__pages[-1]
        self.__lines.append([])

    def create(self, path: str = None):
        if path is None:
            path = generate_pdf_path()
        create_pages(path, self.num_of_pages, [
                     page.title for page in self.__pages], self.__lines)


def generate_pdf_path() -> str:
    path = f"worksheet1.pdf"
    index = 1
    while os.path.isfile(path):
        index += 1
        path = f"worksheet{index}.pdf"

    return path


# An alternative way to create a worksheet with a method rather than a class
def worksheet(path: str = None, dtype='linear', num_of_pages: int = 1, equations_per_page: int = 20, get_solutions=True,
              digits_after=0, titles=None):
    if path is None:
        path = generate_pdf_path()
    if dtype == 'linear':
        LinearEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                         after_point=digits_after, get_solutions=get_solutions, titles=titles)

    elif dtype == 'quadratic':
        QuadraticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                            digits_after=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'cubic':
        CubicEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                        digits_after=digits_after, get_solutions=get_solutions, titles=titles)
    elif dtype == 'quartic':
        QuarticEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
                                          digits_after=digits_after, get_solutions=get_solutions, titles=titles)

    elif dtype == 'polynomial':
        PolyEquation.random_worksheets(path=path, titles=titles, equations_per_page=equations_per_page,
                                       num_of_pages=num_of_pages, digits_after=digits_after,
                                       get_solutions=get_solutions)

    elif dtype == 'trigo':
        # TrigoEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
        #                                after_point=after_point, get_solutions=get_solutions, titles=titles)
        pass
    elif dtype == 'log':
        pass
        # LogEquation.random_worksheets(path=path, num_of_pages=num_of_pages, equations_per_page=equations_per_page,
        #                                 after_point=after_point, get_solutions=get_solutions, titles=titles)
    else:
        raise ValueError(f"worksheet(): unknown dtype {dtype}: expected 'linear', 'quadratic', 'cubic', "
                         f"'quartic', 'polynomial', 'trigo', 'log' ")


def create_pdf(path: str, title="Worksheet", lines=()) -> bool:
    try:
        c = Canvas(path)
        c.setFontSize(22)
        c.drawString(50, 800, title)
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFontSize(14)
        for index, line in enumerate(lines):
            textobject.textLine(f'{index + 1}. {line.strip()}')
            textobject.textLine('')
        c.drawText(textobject)
        # c.showPage()
        # c.setFontSize(22)
        # c.drawString(50, 800, title)
        c.showPage()
        c.save()
        return True
    except Exception as ex:
        warnings.warn(
            f"Couldn't create the pdf file due to a {ex.__class__} error")
        return False


def create_pages(path: str, num_of_pages: int, titles, lines):
    c = Canvas(path)
    for i in range(num_of_pages):
        c.setFontSize(22)
        c.drawString(50, 800, titles[i])
        textobject = c.beginText(2 * cm, 26 * cm)
        c.setFontSize(14)
        for index, line in enumerate(lines[i]):
            textobject.textLine(f'{lines[i][index]}')
            textobject.textLine('')
        c.drawText(textobject)
        c.showPage()
    c.save()
