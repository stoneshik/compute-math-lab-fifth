import math
from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import diff, latex, exp, Symbol, ln

from manager_io import InputManager


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


class SolutionMethod(ABC):
    """
    Базовый абстрактный класс для классов реализаций интерполяции
    """

    def __init__(self, field_names_table: list, name_method: str, initial_data: list) -> None:
        self._field_names_table = field_names_table
        self._name_method: str = name_method
        self._initial_data: list = initial_data
        self._find_solution_x: float = 0.0
        self._find_solution_y: float = 0.0

    @property
    def name_method(self) -> str:
        return self._name_method

    @abstractmethod
    def calc(self, x: float) -> float:
        pass

    @abstractmethod
    def calc_with_output_result(self, x: float) -> PrettyTable:
        pass

    def print_result(self) -> str:
        return f"Для x={self._find_solution_x} было вычисленно значение y={self._find_solution_y}"


class LangrangeMethod(SolutionMethod):
    """
    Класс для интерполяции при помощи многочлена Лагранжа
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'li(x)', 'yi', 'li(x)*yi'], 'многочлен Лагранжа', initial_data)

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
        return table


class GaussMethod(SolutionMethod):
    """
    Класс для интерполяции при помощи многочлена Гаусса
    """
    def __init__(self, initial_data: list) -> None:
        super().__init__(['i', 'X', 'Y', 'P2(x)=ax^2+bx+c', 'εi'], 'phi = ax^2+bx+c', initial_data)

    def calc(self, x: float) -> float:
        pass

    def calc_with_output_result(self, x: float) -> PrettyTable:
        pass


def draw(methods: iter, initial_data: list) -> None:
    plt.figure()
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    plt.title(r'Графики полученных функций')
    x_values = numpy.arange(initial_data[0][0] - 0.2, initial_data[0][-1] + 0.2, 0.01)
    for method in methods:
        y_values = [method.calc(x_iter) for x_iter in x_values]
        try:
            plt.plot(x_values, y_values, linestyle='--', label=f"{method.name_method}")
        except TypeError:
            x_values_error = numpy.arange(initial_data[0][0], initial_data[0][-1], 0.01)
            y_values_error = [method.calc(x_iter) for x_iter in x_values_error]
            plt.plot(x_values_error, y_values_error, linestyle='--', label=f"{method.name_method}")
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
    x_value: float = float(input("Введите значение x, для которого нужно вычислить приближённое значение функции\n"))
    for solution_method in solution_methods:
        print(solution_method.calc_with_output_result(x_value))
        print(solution_method.print_result())
    draw(solution_methods, initial_data)


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()
