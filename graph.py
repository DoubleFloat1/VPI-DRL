import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import List


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
        

def extract_file_data(filepath: str, skip_first_line: bool = True) -> Data:
    matrix: List[List[float]] = []
    with open(filepath, 'r') as file:
        if skip_first_line:
            file.readline()

        line: str
        for line in file:
            row: List[float] = [float(x) for x in line.split(' ')]
            matrix.append(row)
    
    return Data(matrix)

def plot_file(filepath: str, skip_first_line: bool = True,
              x_vals: List[float] | ndarray[float] = None, color: str = None, label: str = None) -> None:
    data: Data = extract_file_data(filepath, skip_first_line=skip_first_line)
    mean: ndarray[float] = data.get_col_mean()
    std: ndarray[float] = data.get_col_std()

    if x_vals == None:
        x_vals = [float(i+1)*10000 for i in range(data.cols)]

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
    
    plot_file("vpidqn.txt",
              color="blue",
              label="VPIDQN",
              skip_first_line=False)
    plot_file("dqn.txt",
              color="red",
              label="DQN",
              skip_first_line=False)
    
    plt.show()

def test():
    with open("dqn.txt", 'r') as file:
        for line in file:
            print(len(line.split(' ')), end=" ")
        print()

if __name__ == "__main__":
    main()