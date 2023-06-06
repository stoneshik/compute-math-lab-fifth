import math
from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import Symbol

from manager_io import InputManager


class TableEndDifference:
    """
    Таблица конечных разностей
    """
    def __init__(self, initial_data: list, x: float) -> None:
        self._initial_data: list = initial_data
        self._x: float = x
        self._x_values: list = []
        self._x_zero_index: int = self._find_x_zero_index(x)
        self._table: list = self._create_table()

    @property
    def table(self) -> list:
        return self._table

    def _find_x_zero_index(self, x: float) -> int:
        x_last: float = self._initial_data[0][0]
        for i, x_iter in enumerate(self._initial_data[0]):
            if abs(x_iter - x) > abs(x_last - x):
                return i - 1
            x_last = x_iter
        return len(self._initial_data[0]) - 1

    def _create_table(self) -> list:
        n_all: int = len(self._initial_data[1]) - 1
        if self._x_zero_index < n_all / 2:
            n_table: int = self._x_zero_index
        elif self._x_zero_index == n_all / 2:
            n_table: int = int(n_all / 2)
        else:
            n_table: int = n_all - self._x_zero_index
        self._x_values: list = self._initial_data[0][self._x_zero_index - n_table:self._x_zero_index + n_table + 1]
        y_values: list = self._initial_data[1][self._x_zero_index - n_table:self._x_zero_index + n_table + 1]
        table: list = [y_values]
        for i in range(n_table * 2 - 1):
            table.append([y_i_plus_1 - y_i for y_i, y_i_plus_1 in zip(table[i][:-1], table[i][1:])])
        return table

    def print_table(self) -> (PrettyTable, None):
        table: PrettyTable = PrettyTable()
        n: int = len(self._table)
        if n < 1:
            return None
        elif n == 1:
            table.field_names = ['i', 'xi', 'yi']
            table.add_row(['0', str(self._x_values[0]), str(self._table[0][0])])
            return table
        else:
            field_names: list = ['i', 'xi', 'yi', 'Δyi']
            field_names.extend([f"Δ^{i}yi" for i in range(2, n)])
            table.field_names = field_names
        for i, x_i in enumerate(self._x_values):
            if n - i > 1:
                row: list = [f"{i - int(n / 2)}", str(x_i), str(self._table[0][i]), f"{self._table[1][i]}"]
            else:
                row: list = [f"{i - int(n / 2)}", str(x_i), str(self._table[0][i]), '-']
            row.extend([f"{self._table[j][i]}" if j < n - i else '-' for j in range(2, n)])
            table.add_row(row)
        return table


class SolutionMethod(ABC):
    """
    Базовый абстрактный класс для классов реализаций интерполяции
    """

    def __init__(self, field_names_table: list, name_method: str, color: str, initial_data: list) -> None:
        self._field_names_table = field_names_table
        self._name_method: str = name_method
        self._color: str = color
        self._initial_data: list = initial_data
        self._find_solution_x: float = 0.0
        self._find_solution_y: float = 0.0
        self._is_calc: bool = False

    @property
    def name_method(self) -> str:
        return self._name_method

    @property
    def color(self) -> str:
        return self._color

    @property
    def is_calc(self) -> bool:
        return self._is_calc

    @abstractmethod
    def calc_error(self) -> float:
        pass

    @abstractmethod
    def calc(self, x: float) -> (float, None):
        pass

    @abstractmethod
    def calc_with_output_result(self, x: float) -> (PrettyTable, None):
        pass

    def _check_equidistant_nodes(self) -> bool:
        if len(self._initial_data[0]) < 2:
            return False
        h: float = self._initial_data[0][1] - self._initial_data[0][0]
        for x_i, x_i_plus in zip(self._initial_data[0][1:-1], self._initial_data[0][2:]):
            if x_i_plus - x_i != h:
                return False
        return True

    def print_result(self) -> str:
        return f"Для x={self._find_solution_x} было вычисленно значение y={self._find_solution_y}\nПогрешность R={self.calc_error()}"


class LangrangeMethod(SolutionMethod):
    """
    Класс для интерполяции при помощи многочлена Лагранжа
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'li(x)', 'yi', 'li(x)*yi'], 'многочлен Лагранжа', 'blue', initial_data)

    def calc_error(self) -> float:
        f_max: float = max(self._initial_data[1])
        n: int = len(self._initial_data[0])
        x: float = self._find_solution_x
        return f_max / math.factorial(n + 1) * abs(sum((x - x_i for x_i in self._initial_data[0])))

    def calc(self, x: float) -> float:
        l_n: float = 0.0
        n: int = len(self._initial_data[0])
        for i, x_i, y_i in zip(range(n), self._initial_data[0], self._initial_data[1]):
            l_n_iter: float = 1.0
            for j, x_j in enumerate(self._initial_data[0]):
                if i == j:
                    continue
                l_n_iter *= (x - x_j) / (x_i - x_j)
            l_n += y_i * l_n_iter
        return l_n

    def calc_with_output_result(self, x: float) -> PrettyTable:
        table: PrettyTable = PrettyTable()
        table.field_names = self._field_names_table
        l_n: float = 0.0
        n: int = len(self._initial_data[0])
        table_end_difference: TableEndDifference = TableEndDifference(self._initial_data, x)
        print(table_end_difference.print_table())
        for i, x_i, y_i in zip(range(n), self._initial_data[0], self._initial_data[1]):
            l_n_iter: float = 1.0
            for j, x_j in enumerate(self._initial_data[0]):
                if i == j:
                    continue
                l_n_iter *= (x - x_j) / (x_i - x_j)
            l_n += y_i * l_n_iter
            table.add_row([i, l_n_iter, y_i, y_i * l_n_iter])
            i += 1
        self._find_solution_x: float = x
        self._find_solution_y: float = l_n
        self._is_calc: bool = True
        return table


class GaussMethod(SolutionMethod):
    """
    Класс для интерполяции при помощи многочлена Гаусса
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'li(x)', 'yi', 'li(x)*yi'], 'многочлен Гаусса', 'green', initial_data)

    def calc_error(self) -> float:
        return 1.0

    def calc(self, x: float) -> (float, None):
        if not self._check_equidistant_nodes():
            return None
        pass

    def calc_with_output_result(self, x: float) -> (PrettyTable, None):
        if not self._check_equidistant_nodes():
            return None
        table: PrettyTable = PrettyTable()
        table.field_names = self._field_names_table
        table_end_difference: TableEndDifference = TableEndDifference(self._initial_data, x)
        return table


def draw(methods: iter, initial_data: list, input_manager: InputManager) -> None:
    plt.figure()
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    plt.title(r'Графики полученных функций')
    x_values = numpy.arange(initial_data[0][0] - 0.2, initial_data[0][-1] + 0.2, 0.01)
    if input_manager.chosen_function is not None:
        x_symbol: Symbol = Symbol('x')
        y_values: list = [
            input_manager.chosen_function.equation_func.subs(x_symbol, x_iter) for x_iter in x_values]
        plt.plot(x_values, y_values, color='orange', label="Выбранная функция")
    c: int = 0
    for method in methods:
        if not method.is_calc:
            continue
        y_values = [method.calc(x_iter) for x_iter in x_values]
        try:
            plt.plot(x_values, y_values, linestyle='--',
                     color=f"{method.color}", label=f"{method.name_method}")
        except TypeError:
            x_values_error = numpy.arange(initial_data[0][0], initial_data[0][-1], 0.01)
            y_values_error = [method.calc(x_iter) for x_iter in x_values_error]
            plt.plot(x_values_error, y_values_error, linestyle='--',
                     color=f"{method.color}", label=f"{method.name_method}")
        c += 1
    if c > 0:
        plt.legend(loc='upper left')
    x_values = []
    y_values = []
    for x, y in zip(initial_data[0], initial_data[1]):
        x_values.append(x)
        y_values.append(y)
    plt.scatter(x_values, y_values, color='red', marker='o')
    plt.show()


def main():
    input_manager: InputManager = InputManager()
    initial_data = input_manager.input()
    if initial_data is None:
        return
    solution_methods = (
        LangrangeMethod(initial_data),
        #GaussMethod(initial_data)
    )
    #table_end_difference: TableEndDifference = TableEndDifference(initial_data)
    #if table_end_difference is None:
    #    return
    #print(table_end_difference.print_table())
    x_value: float = float(input("Введите значение x, для которого нужно вычислить приближённое значение функции\n"))
    if not initial_data[0][0] <= x_value <= initial_data[0][-1]:
        print("Значение x не попадает в заданный интервал")
        return
    for solution_method in solution_methods:
        result = solution_method.calc_with_output_result(x_value)
        if not solution_method.is_calc or result is None:
            continue
        print(result)
        print(solution_method.print_result())
    draw(solution_methods, initial_data, input_manager)


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()
