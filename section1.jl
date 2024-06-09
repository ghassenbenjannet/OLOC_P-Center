using Random
using Plots
using JuMP
using GLPK
include("IO_UFLP.jl") 

# Distance
function dist(x1, y1, x2, y2)
    return sqrt((x2 - x1)^2 + (y2 - y1)^2)
end

# Aléatoire
function generate_random_solution(n, p)
    S = zeros(Int, n)  
    indices = randperm(n)[1:p]  
    for idx in indices
        S[idx] = 1
    end
    return S
end

# rayon max
function calculate_max_radius(tabX, tabY, S)
    max_radius = 0.0
    for i in 1:length(tabX)
        min_dist = Inf
        for j in 1:length(tabX)
            if S[j] == 1
                distance = dist(tabX[i], tabY[i], tabX[j], tabY[j])
                if distance < min_dist
                    min_dist = distance
                end
            end
        end
        max_radius = max(max_radius, min_dist)
    end
    return max_radius
end

# une meilleure solution aléatoire
function find_better_random_solution(n, p, tabX, tabY, attempts=30)
    best_S = zeros(Int, n)
    best_radius = -1
    for _ in 1:attempts
        S = generate_random_solution(n, p)
        radius = calculate_max_radius(tabX, tabY, S)
        if radius < best_radius
            best_radius = radius
            best_S = S
        end
    end
    return best_S, best_radius
end


# Methode exacte 
function solve_p_center_exacte(tabX, tabY, p)
    n = length(tabX)
    model = Model(GLPK.Optimizer)
    
    @variable(model, x[1:n, 1:n], Bin)  # x_ij
    @variable(model, y[1:n], Bin)       # x_jj (y[j] pour simplifier)
    @variable(model, z)                 # Distance maximale à minimiser
    
    # C1: Au plus p antennes
    @constraint(model, sum(y) <= p)
    
    # Contrainte C2: Chaque point est affecté à au moins une antenne
    for i in 1:n
        @constraint(model, sum(x[i, j] for j in 1:n) == 1)
    end
    
    # Contrainte C3: Si i est affecté à j, alors j doit avoir une antenne
    for i in 1:n, j in 1:n
        @constraint(model, x[i, j] <= y[j])
    end

    # C4: Distance de i à son antenne est majorée par z
    for i in 1:n
        for j in 1:n
            @constraint(model, x[i, j] * dist(tabX[i], tabY[i], tabX[j], tabY[j]) <= z)
        end
    end
    
    # Minimiser z
    @objective(model, Min, z)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        println("Solution trouvée avec z = ", objective_value(model))
        S = [value(y[j]) for j in 1:n]
        return S, objective_value(model)
    else
        println("Pas de solution optimale habibi")
        return [], Inf
    end
end


# la relaxation linéaire
function solve_p_center_relaxed(tabX, tabY, p)
    n = length(tabX)
    model = Model(GLPK.Optimizer)
    
    @variable(model, 0 <= x[1:n, 1:n] <= 1)
    @variable(model, 0 <= y[1:n] <= 1)
    @variable(model, z)

    # Contraintes
    @constraint(model, sum(y[j] for j in 1:n) <= p)
    for i in 1:n
        @constraint(model, sum(x[i, j] for j in 1:n) == 1)
    end
    
    for i in 1:n, j in 1:n
        @constraint(model, x[i, j] <= y[j])
    end
    
    for i in 1:n
        for j in 1:n
            @constraint(model, x[i, j] * dist(tabX[i], tabY[i], tabX[j], tabY[j]) <= z)
        end
    end
    
    @objective(model, Min, z)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        y_vals = value.(y)
        return y_vals
    else
        println("Pas de solution optimale trouvée :o ")
        return []
    end
end

# arrondissement relaxee
function round_relaxed_solution(y_vals, p)
    sorted_indices = sortperm(y_vals, rev=true)
    return zeros(Int, length(y_vals)), sorted_indices[1:p]
end

# une descente stochastique sur une solution initiale
function stochastic_descent(tabX, tabY, p, initial_solution)
    current_solution = copy(initial_solution)
    best_solution = copy(initial_solution)
    best_radius = calculate_max_radius(tabX, tabY, best_solution)
    unchanged_rounds = 0
    max_unchanged_rounds = 20  # Nombre d'itérations sans amélioration avant d'arrêter

    while unchanged_rounds < max_unchanged_rounds
        # Sélectionner aléatoirement une antenne à déplacer
        antenne_idx = findall(x -> x == 1, current_solution)
        non_antenne_idx = findall(x -> x == 0, current_solution)

        if isempty(antenne_idx) || isempty(non_antenne_idx)
            break
        end

        # Faire un échange aléatoire
        swap_out = rand(antenne_idx)
        swap_in = rand(non_antenne_idx)

        current_solution[swap_out] = 0
        current_solution[swap_in] = 1

        # Évaluer la nouvelle solution
        current_radius = calculate_max_radius(tabX, tabY, current_solution)

        if current_radius < best_radius
            best_solution = copy(current_solution)
            best_radius = current_radius
            unchanged_rounds = 0  # Réinitialiser le compteur d'arrêt
        else
            unchanged_rounds += 1
        end
    end

    return best_solution, best_radius
end

# descente stochastique itérée
function iterated_stochastic_descent(tabX, tabY, p, num_iterations)
    best_global_solution = []
    best_global_radius = Inf

    for i in 1:num_iterations
        initial_solution = generate_random_solution(length(tabX), p)
        local_solution, local_radius = stochastic_descent(tabX, tabY, p, initial_solution)

        if local_radius < best_global_radius
            best_global_solution = local_solution
            best_global_radius = local_radius
        end
    end

    return best_global_solution, best_global_radius
end



# Fonction principale
function main()
    nom_fichier = "inst_10000.flp"  
    tabX, tabY, f = Float64[], Float64[], Float64[]
    n = Lit_fichier_UFLP(nom_fichier, tabX, tabY, f)
    p = 8 

    # Dessin des villes
    Dessine_UFLP(nom_fichier)
    
   
    # S, best_radius = find_better_random_solution(n, p, tabX, tabY)
    # S, best_radius = solve_p_center_exacte(tabX, tabY, p)

    # pour relaxation
    y_vals = solve_p_center_relaxed(tabX, tabY, p)
    S, indices = round_relaxed_solution(y_vals, p)
    # Marquer les positions des antennes sélectionnées
    for idx in indices
        S[idx] = 1
    end
    best_radius = calculate_max_radius(tabX, tabY, S)

   # descente:
   #num_iterations = 5
   #S, best_radius = iterated_stochastic_descent(tabX, tabY, p, num_iterations)

    #println("Meilleur rayon trouvé: ", best_radius)
    
    # Dessin de la meilleure solution
    Dessine_UFLP(nom_fichier, n, tabX, tabY, S)
end

main()
