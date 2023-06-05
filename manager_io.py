class InputManager:
    def __init__(self) -> None:
        self._method_input: int = 1
        self._file_path: str = ''

    def input(self) -> (list, None):
        print("Формат входного файла:\nКоличество точек N>0\nКоординаты X и Y через пробел\n")
        while True:
            print("Выберите способ ввода данных")
            print("1. Через консоль\n2. Через файл")
            num_variant = int(input("Введите номер выбранного варианта...\n"))
            if num_variant < 1 or num_variant > 2:
                print("Введен неправильной номер, повторите ввод")
                continue
            break
        if num_variant == 2:
            self._file_path: str = input("Введите название файла\n")
            return self._input_from_file()
        return self._input_from_console()

    def _input_from_console(self) -> (list, None):
        while True:
            n: int = int(input("Введите количество вводимых точек...\n"))
            if n <= 0:
                print("Количество вводимых точек должно быть больше нуля")
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
            if n <= 0:
                print("Количество вводимых точек должно быть больше нуля")
                return None
            for _ in range(n):
                x, y = (float(i) for i in file.readline().split())
                if len(initial_data[0]) > 0 and x <= initial_data[0][-1]:
                    print("Введенно X, которое меньше или равно предыдущему")
                    return None
                initial_data[0].append(x)
                initial_data[1].append(y)
        return initial_data
