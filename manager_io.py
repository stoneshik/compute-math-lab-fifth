from sympy import diff, latex, sin, Symbol


class Equation:
    """
    Класс обертка для функций
    """

    def __init__(self, equation_func) -> None:
        self.equation_func = equation_func

    def get_string(self) -> str:
        return latex(self.equation_func)

    def get_diff(self):
        return diff(self.equation_func)


class InputManager:
    def __init__(self) -> None:
        self._method_input: int = 1
        self._file_path: str = ''
        self._chosen_function = None

    def input(self) -> (list, None):
        print("Формат входного файла:\nКоличество точек N>0\nКоординаты X и Y через пробел\n")
        while True:
            print("Выберите способ ввода данных")
            print("1. Через консоль\n2. Через файл\n3. При помощи выбранной функции")
            num_variant = int(input("Введите номер выбранного варианта...\n"))
            if num_variant < 1 or num_variant > 3:
                print("Введен неправильной номер, повторите ввод")
                continue
            break
        if num_variant == 2:
            self._file_path: str = input("Введите название файла\n")
            return self._input_from_file()
        elif num_variant == 3:
            return self._input_from_function()
        return self._input_from_console()

    def _input_from_console(self) -> (list, None):
        while True:
            n: int = int(input("Введите количество вводимых точек...\n"))
            if n <= 1:
                print("Количество вводимых точек должно быть больше одной")
                continue
            break
        initial_data: list = [[], []]
        print("Вводите значения X и Y через пробел")
        for _ in range(n):
            x, y = (float(i) for i in input().split())
            if len(initial_data[0]) > 0 and x <= initial_data[0][-1]:
                print("Введенно X, которое меньше или равно предыдущему")
                return None
            initial_data[0].append(x)
            initial_data[1].append(y)
        return initial_data

    def _input_from_file(self) -> (list, None):
        initial_data: list = [[], []]
        with open(self._file_path, 'r', encoding='utf-8') as file:
            n: int = int(file.readline())
            if n <= 1:
                print("Количество вводимых точек должно быть больше одной")
                return None
            for _ in range(n):
                x, y = (float(i) for i in file.readline().split())
                if len(initial_data[0]) > 0 and x <= initial_data[0][-1]:
                    print("Введенно X, которое меньше или равно предыдущему")
                    return None
                initial_data[0].append(x)
                initial_data[1].append(y)
        return initial_data

    def _input_from_function(self) -> (list, None):
        x: Symbol = Symbol('x')
        functions: tuple = (
            Equation(x + 1),
            Equation(x ** 2 - 1),
            Equation(sin(x))
        )
        function = None
        while True:
            print("Выберите функцию:")
            [print(f"{i + 1}. y = {equation_iter.get_string()}") for i, equation_iter in enumerate(functions)]
            equation_num = int(input("Введите номер выбранной фукнции...\n"))
            if equation_num < 1 or equation_num > len(functions):
                print("Номер фукнции не найден, повторите ввод")
                continue
            function = functions[equation_num - 1]
            break
        while True:
            a, b = (float(i) for i in input("Введите значения для исследуемого интервала [a, b]...\n").split())
            if a >= b:
                print("Значение b должно быть больше a")
                continue
            break
        while True:
            n: int = int(input("Введите количество вводимых точек...\n"))
            if n <= 1:
                print("Количество вводимых точек должно быть больше одной")
                continue
            break
        h: float = (b - a) / (n - 1)
        x_values: list = []
        y_values: list = []
        for i in range(n):
            x_value: float = a + h * i
            x_values.append(x_value)
            y_values.append(function.equation_func.subs(x, x_value))
        return [x_values, y_values]
