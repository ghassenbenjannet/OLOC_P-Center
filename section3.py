import numpy as np
import matplotlib.pyplot as plt

def random_method(points, p):
    indices = np.random.choice(range(len(points)), size=p, replace=False)
    antennes = points[indices]
    return antennes


def min_distance_method(points, p):
    antennes = []
    # Choisissez le premier point aléatoirement
    first_index = np.random.choice(range(len(points)))
    antennes.append(points[first_index])
    
    for _ in range(p - 1):
        max_dist = np.array([np.min([np.linalg.norm(point - ant) for ant in antennes]) for point in points])
        next_index = np.argmax(max_dist)
        antennes.append(points[next_index])
    
    return np.array(antennes)


def plot_solution(points, antennes):
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Villes')
    plt.scatter(antennes[:, 0], antennes[:, 1], color='red', label='Antennes')
    
    # Relier chaque ville à l'antenne la plus proche
    for point in points:
        # Calculer la distance de ce point à chaque antenne
        distances = np.linalg.norm(point - antennes, axis=1)
        # Trouver l'indice de l'antenne la plus proche
        closest_antenne_index = np.argmin(distances)
        # Dessiner une ligne entre la ville et l'antenne la plus proche
        closest_antenne = antennes[closest_antenne_index]
        plt.plot([point[0], closest_antenne[0]], [point[1], closest_antenne[1]], 'gray', linestyle='--', linewidth=0.5)
    
    plt.legend()
    plt.grid(True)
    plt.show()

# Main
points = np.random.rand(100, 2)  # 100 points aléatoires
p = 5  # Nombre d'antennes
#antennes = random_method(points, p)
antennes = min_distance_method(points, p)

plot_solution(points, antennes)
