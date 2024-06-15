import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree

def random_method(points, p):
    indices = np.random.choice(range(len(points)), size=p, replace=False)
    antennes = points[indices]
    return antennes

def min_distance_method(points, p):
    antennes = []

    first_index = np.random.choice(range(len(points)))
    antennes.append(points[first_index])
    
    tree = cKDTree(points)
    
    for _ in range(p - 1):
        distances, _ = tree.query(antennes, k=1)
        max_dist_index = np.argmax(distances)
        antennes.append(points[max_dist_index])
    
    return np.array(antennes)

def plot_solution(points, antennes):
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Villes')
    plt.scatter(antennes[:, 0], antennes[:, 1], color='red', label='Antennes')
    
    for point in points:
        distances = np.linalg.norm(point - antennes, axis=1)
        closest_antenne_index = np.argmin(distances)
        closest_antenne = antennes[closest_antenne_index]
        plt.plot([point[0], closest_antenne[0]], [point[1], closest_antenne[1]], 'red', linestyle='-', linewidth=0.5)
    
    plt.legend()
    plt.grid(True)
    plt.show()

def read_uflp_file(file_name):
    data = pd.read_csv(file_name, sep='\s+', header=None, skiprows=1)
    tabX = data[2].astype(float).values  
    tabY = data[3].astype(float).values  
    return np.column_stack((tabX, tabY)) 

# Main
nom_fichier = "inst_0.flp" 
points = read_uflp_file(nom_fichier)
p = 16000

antennes = min_distance_method(points, p)

plot_solution(points, antennes)