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

def improved_min_distance_method(points, p):
    antennes = []

    first_index = np.random.choice(range(len(points)))
    antennes.append(points[first_index])
    
    for _ in range(p - 1):
        max_min_distance = -1
        next_antenne = None
        
        for point in points:
            if any(np.array_equal(point, antenne) for antenne in antennes):
                continue
            
            min_distance = np.min([np.linalg.norm(point - antenne) for antenne in antennes])
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_antenne = point
                
        antennes.append(next_antenne)
    
    return np.array(antennes)

def plot_solution(points, antennes):
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Villes')
    plt.scatter(antennes[:, 0], antennes[:, 1], color='red', label='Antennes')
    
    max_distance = 0
    for point in points:
        distances = np.linalg.norm(point - antennes, axis=1)
        closest_antenne_index = np.argmin(distances)
        closest_antenne = antennes[closest_antenne_index]
        plt.plot([point[0], closest_antenne[0]], [point[1], closest_antenne[1]], 'red', linestyle='-', linewidth=0.5)
        
        # Mise à jour de la distance maximale
        max_distance = max(max_distance, np.min(distances))
    
    plt.legend()
    plt.grid(True)
    plt.title(f"Solution avec p = {len(antennes)} antennes\nRayon maximal des distances (Z) = {max_distance:.2f}")
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.show()

def read_uflp_file(file_name):
    data = pd.read_csv(file_name, sep='\s+', header=None, skiprows=1)
    tabX = data[2].astype(float).values  
    tabY = data[3].astype(float).values  
    return np.column_stack((tabX, tabY)) 

# Main
nom_fichier = "inst_1000.flp" 
points = read_uflp_file(nom_fichier)
p = 15  

antennes = improved_min_distance_method(points, p)

plot_solution(points, antennes)
