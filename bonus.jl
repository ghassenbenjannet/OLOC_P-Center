using Random
using JuMP
using GLPK
using DataFrames
include("IO_UFLP.jl")

# Calculer la distance entre deux points
function distance(x1, y1, x2, y2)
    return sqrt((x2 - x1)^2 + (y2 - y1)^2)
end

# Générer une solution aléatoire
function generer_solution_aleatoire(n, p)
    S = zeros(Int, n)
    indices = randperm(n)[1:p]
    for idx in indices
        S[idx] = 1
    end
    return S
end

# Calculer le rayon maximal
function calculer_rayon_max(tabX, tabY, S)
    rayon_max = 0.0
    for i in 1:length(tabX)
        distance_min = Inf
        for j in 1:length(tabX)
            if S[j] == 1
                dist = distance(tabX[i], tabY[i], tabX[j], tabY[j])
                if dist < distance_min
                    distance_min = dist
                end
            end
        end
        rayon_max = max(rayon_max, distance_min)
    end
    return rayon_max
end

# Ajouter les nouvelles contraintes
function ajouter_contraintes_nouvelles!(modele, tabX, tabY, d, x, y; budget=nothing, dmax=nothing)
    n = length(tabX)

    # Contraintes de budget fixé
    if budget !== nothing
        @constraint(modele, sum(y[j] for j in 1:n) <= budget)
    end

    # Contraintes de dmax
    if dmax !== nothing
        @constraint(modele, sum(d[i, j] * x[i, j] for i in 1:n, j in 1:n) <= dmax)
    end

    # Ajout des nouvelles inégalités
    if dmax !== nothing
        for i in 1:n, j in 1:n, k in 1:n
            if d[i, j] <= d[i, k] && d[i, k] <= dmax
                @constraint(modele, x[i, k] + x[j, k] <= 1)
            end
        end
    end
end

# Méthode bonus pour minimiser le max de la somme des distances avec budget fixé
function resoudre_p_center_bonus1(tabX, tabY, budget)
    n = length(tabX)
    d = [distance(tabX[i], tabY[i], tabX[j], tabY[j]) for i in 1:n, j in 1:n]
    modele = Model(GLPK.Optimizer)

    @variable(modele, 0 <= x[1:n, 1:n] <= 1)
    @variable(modele, 0 <= y[1:n] <= 1)
    @variable(modele, z >= 0)

    @constraint(modele, sum(y[j] for j in 1:n) <= budget)
    for i in 1:n
        @constraint(modele, sum(x[i, j] for j in 1:n) == 1)
    end

    for i in 1:n, j in 1:n
        @constraint(modele, x[i, j] <= y[j])
    end

    # Contrainte principale
    for i in 1:n, j in 1:n
        @constraint(modele, x[i, j] * d[i, j] <= z)
    end

    @objective(modele, Min, z)

    # Ajouter les nouvelles contraintes
    ajouter_contraintes_nouvelles!(modele, tabX, tabY, d, x, y, budget=budget)

    optimize!(modele)

    if termination_status(modele) == MOI.OPTIMAL
        y_vals = value.(y)
        return y_vals, objective_value(modele)
    else
        println("Pas de solution optimale trouvée.")
        return [], Inf
    end
end

# Méthode bonus pour minimiser le budget avec contrainte sur la somme des distances
function resoudre_p_center_bonus2(tabX, tabY, dmax)
    n = length(tabX)
    d = [distance(tabX[i], tabY[i], tabX[j], tabY[j]) for i in 1:n, j in 1:n]
    modele = Model(GLPK.Optimizer)

    @variable(modele, 0 <= x[1:n, 1:n] <= 1)
    @variable(modele, 0 <= y[1:n] <= 1)
    @variable(modele, z >= 0)

    for i in 1:n
        @constraint(modele, sum(x[i, j] for j in 1:n) == 1)
    end
    for i in 1:n, j in 1:n
        @constraint(modele, x[i, j] <= y[j])
    end
    @constraint(modele, sum(d[i, j] * x[i, j] for i in 1:n, j in 1:n) <= dmax)

    @objective(modele, Min, sum(y[j] for j in 1:n))

    # Ajouter les nouvelles contraintes
    ajouter_contraintes_nouvelles!(modele, tabX, tabY, d, x, y, dmax=dmax)

    optimize!(modele)

    if termination_status(modele) == MOI.OPTIMAL
        y_vals = value.(y)
        return y_vals, objective_value(modele)
    else
        println("Pas de solution optimale trouvée.")
        return [], Inf
    end
end

# Fonction principale pour la partie bonus
function main_bonus()
    nom_fichier = "inst_100000.flp"
    tabX, tabY, f = Float64[], Float64[], Float64[]
    n = Lit_fichier_UFLP(nom_fichier, tabX, tabY, f)
    budget = 50
    dmax = 80.0

    # Résolution avec les nouvelles contraintes
    println("Méthode bonus 1 : Minimiser le max de la somme des distances avec budget fixe")
    y_vals, z = resoudre_p_center_bonus1(tabX, tabY, budget)
    println("Valeur de z : ", z)

    println("Méthode bonus 2 : Minimiser le budget avec contrainte sur la somme des distances")
    y_vals, p = resoudre_p_center_bonus2(tabX, tabY, dmax)
    println("Budget p : ", p)
end

main_bonus()
