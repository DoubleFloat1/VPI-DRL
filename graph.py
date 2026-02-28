import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import List, Tuple


class Data:
    def __init__(self, matrix: List[List[float]]):
        self.matrix: ndarray[ndarray[float]] = np.array(matrix)
        self.rows: int = len(matrix)
        self.cols: int = len(matrix[0])
        self.matrix = self.matrix
    
    def get_col_mean(self) -> ndarray[float]:
        return self.matrix.mean(axis=0)
    
    def get_col_std(self) -> ndarray[float]:
        return self.matrix.std(axis=0)
        

def extract_file_data(filepath: str) -> Tuple[Data, int]:
    matrix: List[List[float]] = []
    train_step_amount: int = -1
    with open(filepath, 'r') as file:
        train_step_amount = int(file.readline())
        (rows, cols) = [int(x) for x in file.readline().split(' ')]
        for _ in range(rows):
            line = file.readline()
            row: List[float] = [float(x) for x in line.split(' ')]
            matrix.append(row)

    
    return Data(matrix), train_step_amount

def plot_file(filepath: str, x_vals: List[float] | ndarray[float] = None, color: str = None, label: str = None) -> None:
    data: Data
    train_step_amount: int
    data, train_step_amount = extract_file_data(filepath)
    mean: ndarray[float] = data.get_col_mean()
    std: ndarray[float] = data.get_col_std()

    if x_vals == None:
        x_vals = [float(i)*train_step_amount for i in range(data.cols)]

    plt.plot(x_vals, mean, color=color, label=label)

    y1: ndarray[float] = mean - (std / 2)
    y2: ndarray[float] = mean + (std / 2)
    plt.fill_between(x_vals, y1, y2, alpha=0.3, color=color)
    plt.legend()

def plot_details(title: str = None, x_axis_name: str = None, y_axis_name: str = None, show_legend: bool = True) -> None:
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)

def main():
    plot_details(
        title="Recompensa média obtida pelos algoritmos",
        x_axis_name="Passos de treinamento",
        y_axis_name="Recompensa média"
    )
    
    #plot_file("data/lunar_lander/vpidqn.txt", label="VPIDQN", color="blue")
    plot_file("data/breakout/dqn.txt", label="DQN", color="red")

    #plot_file("vpidqn.txt", label="VPIDQN", color="blue")
    #plot_file("dqn.txt", label="DQN", color="red")
    
    plt.show()

def test():
    with open("dqn.txt", 'r') as file:
        for line in file:
            print(len(line.split(' ')), end=" ")
        print()

if __name__ == "__main__":
    main()