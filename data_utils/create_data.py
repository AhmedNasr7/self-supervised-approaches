import numpy as np 
import os 
from pathlib import Path

'''
This is only for creating simulation data with the same dimenstions for testing 
data shapes: (1, 1200) -- classes: 0, 1, 2


'''

path = "../data/"

def create_random_arrays(directory, n):
    os.makedirs(directory, exist_ok=True)
    
    for i in range(n):
        random_array = np.random.random((1, 1200)).astype(np.float32) # (c , l)
        filename = os.path.join(directory, f"array_{i}.npy")
        np.save(filename, random_array)

def main():
    num_arrays_per_directory = 100 # Change this value to the desired number of arrays per directory
    directories = ['0', '1', '2']
    
    for directory in directories:
        dir_path = path + directory

        create_random_arrays(dir_path, num_arrays_per_directory)
    
    print("Random arrays saved in directories: ", directories)


if __name__ == "__main__":
    main()
