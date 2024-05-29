using Plots
include("IO_UFLP.jl")  # Assure-toi que ce chemin vers le fichier est correct

# Fonction pour calculer la distance euclidienne
function dist(x1, y1, x2, y2)
    return sqrt((x2 - x1)^2 + (y2 - y1)^2)
end

# Fonction pour générer une solution aléatoire
function generate_random_solution(n, p)
    S = zeros(Int, n)  # tableau pour marquer les emplacements choisis pour les antennes
    indices = randperm(n)[1:p]  # choisir aléatoirement p indices
    for idx in indices
        S[idx] = 1
    end
    return S
end

# Dessiner la solution avec les connexions
function visualize_solution(tabX, tabY, S)
    scatter(tabX, tabY, color=:blue, label="Villes")
    for i in 1:length(tabX)
        if S[i] == 1
            scatter!([tabX[i]], [tabY[i]], color=:red, label="Antennes")
        end
    end
    for i in 1:length(tabX)
        min_dist = Inf
        nearest_j = i
        for j in 1:length(tabX)
            if S[j] == 1
                distance = dist(tabX[i], tabY[i], tabX[j], tabY[j])
                if distance < min_dist
                    min_dist = distance
                    nearest_j = j
                end
            end
        end
        if i != nearest_j
            plot!([tabX[i], tabX[nearest_j]], [tabY[i], tabY[nearest_j]], color=:green, label="Liaisons")
        end
    end
    display(plot!())
end

# Lire les données de l'instance et exécuter les visualisations
function main()
    nom_fichier = "inst_3000000.flp"  # Vérifiez le nom du fichier et le chemin
    tabX, tabY, f = Float64[], Float64[], Float64[]
    n = Lit_fichier_UFLP(nom_fichier, tabX, tabY, f)

    # Générer une solution aléatoire
    p = 4  # Nombre d'antennes à placer, ajustez selon le besoin
    S = generate_random_solution(n, p)
    
    # Visualiser la solution avec connexions
    visualize_solution(tabX, tabY, S)
end

main()