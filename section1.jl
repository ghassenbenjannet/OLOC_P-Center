using Random
using Plots
using JuMP
using GLPK
using DataStructures  # Pour PriorityQueue
using Printf
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

# Trouver une meilleure solution aléatoire
function trouver_meilleure_solution_aleatoire(n, p, tabX, tabY, essais=30)
    meilleure_S = zeros(Int, n)
    meilleur_rayon = Inf
    for _ in 1:essais
        S = generer_solution_aleatoire(n, p)
        rayon = calculer_rayon_max(tabX, tabY, S)
        if rayon < meilleur_rayon
            meilleur_rayon = rayon
            meilleure_S = S
        end
    end
    return meilleure_S, meilleur_rayon
end

# Méthode exacte pour résoudre le problème p-center
function resoudre_p_center_exact(tabX, tabY, p)
    n = length(tabX)
    modele = Model(GLPK.Optimizer)

    # Déclaration des variables
    @variable(modele, x[1:n, 1:n], Bin)  # x_ij: 1 si le point i est affecté à l'antenne j
    @variable(modele, y[1:n], Bin)       # y_j: 1 si une antenne est placée au point j
    @variable(modele, z >= 0)            # z: distance maximale à minimiser

    # Contrainte C1: Au plus p antennes
    @constraint(modele, sum(y) <= p)

    # Contrainte C2: Chaque point est affecté à au moins une antenne
    for i in 1:n
        @constraint(modele, sum(x[i, :]) == 1)
    end

    # Contrainte C3: Si un point i est affecté à j, alors j doit avoir une antenne
    for i in 1:n, j in 1:n
        @constraint(modele, x[i, j] <= y[j])
    end

    # Contrainte C4: La distance de i à son antenne est majorée par z
    for i in 1:n
        for j in 1:n
            @constraint(modele, x[i, j] * distance(tabX[i], tabY[i], tabX[j], tabY[j]) <= z)
        end
    end

    # Objectif: Minimiser z
    @objective(modele, Min, z)

    # Résoudre le problème
    optimize!(modele)

    # Vérifier si la solution est optimale
    if termination_status(modele) == MOI.OPTIMAL
        println("Solution trouvée avec z = ", objective_value(modele))
        S = [value(y[j]) for j in 1:n]  # Récupérer la configuration des antennes
        return S, objective_value(modele)
    else
        println("Pas de solution optimale trouvée.")
        return [], Inf
    end
end

# La relaxation linéaire
function resoudre_p_center_relax(tabX, tabY, p)
    n = length(tabX)
    modele = Model(GLPK.Optimizer)

    @variable(modele, 0 <= x[1:n, 1:n] <= 1)
    @variable(modele, 0 <= y[1:n] <= 1)
    @variable(modele, z >= 0)

    # Contraintes
    @constraint(modele, sum(y[j] for j in 1:n) <= p)
    for i in 1:n
        @constraint(modele, sum(x[i, j] for j in 1:n) == 1)
    end

    for i in 1:n, j in 1:n
        @constraint(modele, x[i, j] <= y[j])
    end

    for i in 1:n
        for j in 1:n
            @constraint(modele, x[i, j] * distance(tabX[i], tabY[i], tabX[j], tabY[j]) <= z)
        end
    end

    @objective(modele, Min, z)

    optimize!(modele)

    if termination_status(modele) == MOI.OPTIMAL
        y_vals = value.(y)
        return y_vals, objective_value(modele)
    else
        println("Pas de solution optimale trouvée.")
        return [], Inf
    end
end

# Arrondi relaxé (simple arrondi)
function arrondir_solution_relax(y_vals, p, tabX, tabY)
    indices_tries = sortperm(y_vals, rev=true)
    solution_arrondie = zeros(Int, length(y_vals))
    indices = indices_tries[1:p]
    for idx in indices
        solution_arrondie[idx] = 1
    end

    # Recalculer le rayon maximal pour cette solution entière
    rayon_entier = calculer_rayon_max(tabX, tabY, solution_arrondie)

    return solution_arrondie, Int(ceil(rayon_entier))
end

# Descente stochastique améliorée sur une solution initiale
function descente_stochastique(tabX, tabY, p, solution_initiale)
    solution_actuelle = copy(solution_initiale)
    meilleure_solution = copy(solution_initiale)
    meilleur_rayon = calculer_rayon_max(tabX, tabY, meilleure_solution)
    println("Valeur initiale de z (descente stochastique) : ", meilleur_rayon)
    iterations_sans_ameli = 0
    max_iterations_sans_ameli = 100  # Nombre d'itérations sans amélioration avant d'arrêter

    while iterations_sans_ameli < max_iterations_sans_ameli
        amelioration = false

        # Parcourir chaque antenne et essayer de la déplacer
        for i in 1:length(solution_actuelle)
            if solution_actuelle[i] == 1
                for j in 1:length(solution_actuelle)
                    if solution_actuelle[j] == 0
                        solution_actuelle[i] = 0
                        solution_actuelle[j] = 1

                        # Évaluer la nouvelle solution
                        rayon_actuel = calculer_rayon_max(tabX, tabY, solution_actuelle)

                        if rayon_actuel < meilleur_rayon
                            meilleure_solution = copy(solution_actuelle)
                            meilleur_rayon = rayon_actuel
                            amelioration = true
                        else
                            # Revenir à la solution précédente
                            solution_actuelle[i] = 1
                            solution_actuelle[j] = 0
                        end
                    end
                end
            end
        end

        if amelioration
            iterations_sans_ameli = 0
        else
            iterations_sans_ameli += 1
        end
    end

    println("Valeur de z après descente stochastique : ", meilleur_rayon)
    return meilleure_solution, Int(ceil(meilleur_rayon))
end

# Descente stochastique itérée
function descente_stochastique_iteree(tabX, tabY, p, nb_iterations)
    meilleure_solution_globale = []
    meilleur_rayon_global = Inf

    for i in 1:nb_iterations
        solution_initiale = generer_solution_aleatoire(length(tabX), p)
        solution_locale, rayon_local = descente_stochastique(tabX, tabY, p, solution_initiale)

        if rayon_local < meilleur_rayon_global
            meilleure_solution_globale = solution_locale
            meilleur_rayon_global = rayon_local
        end
    end

    println("Valeur de z après descente stochastique itérée : ", meilleur_rayon_global)
    return meilleure_solution_globale, meilleur_rayon_global
end

# Définir la structure du noeud pour le Branch and Bound
struct Noeud
    fixe::Vector{Int}
    libre::Vector{Int}
    borne_inf::Float64
end

# Fonction principale branch and bound
function branch_and_bound(tabX, tabY, p)
    n = length(tabX)
    meilleure_solution = []
    meilleur_rayon = Inf

    # Utiliser une file de priorité pour gérer les nœuds partiels
    file = PriorityQueue{Noeud, Float64}()

    # Fonction pour insérer dans la file de priorité
    function inserer!(file, noeud)
        enqueue!(file, noeud => noeud.borne_inf)
    end

    # Initialisation de la file avec le nœud racine
    noeud_racine = Noeud(Int[], collect(1:n), 0.0)
    inserer!(file, noeud_racine)

    while !isempty(file)
        current_noeud = dequeue!(file)

        fixe, libre, borne_inf = current_noeud.fixe, current_noeud.libre, current_noeud.borne_inf

        # Si le nœud est une solution complète
        if length(fixe) == p
            S = zeros(Int, n)
            for idx in fixe
                S[idx] = 1
            end
            rayon = calculer_rayon_max(tabX, tabY, S)
            if rayon < meilleur_rayon
                meilleur_rayon = rayon
                meilleure_solution = S
            end
            continue
        end

        # Si le nœud est élagué
        if borne_inf >= meilleur_rayon
            continue
        end

        # Brancher le nœud
        for idx in libre
            nouveau_fixe = vcat(fixe, idx)
            nouveau_libre = setdiff(libre, [idx])

            # Calculer la borne inférieure pour le nouveau nœud
            S = zeros(Int, n)
            for id in nouveau_fixe
                S[id] = 1
            end

            _, nouvelle_borne_inf = resoudre_p_center_relax(tabX, tabY, p - length(nouveau_fixe))
            nouvelle_borne_inf += calculer_rayon_max(tabX, tabY, S)

            if nouvelle_borne_inf < meilleur_rayon
                nouveau_noeud = Noeud(nouveau_fixe, nouveau_libre, nouvelle_borne_inf)
                inserer!(file, nouveau_noeud)
            end
        end
    end

    return meilleure_solution, Int(ceil(meilleur_rayon))
end

# Fonction pour comparer les méthodes
function comparer_methodes(nom_fichier, p)
    tabX, tabY, f = Float64[], Float64[], Float64[]
    n = Lit_fichier_UFLP(nom_fichier, tabX, tabY, f)

    methodes = ["Méthode exacte", "Solution aléatoire améliorée", "Relaxation linéaire", "Descente stochastique itérée", "Branch and Bound"]
    resultats = DataFrame(Method = String[], Z = Float64[], Gap = Float64[], Temps = Float64[])

    z_optimal = 0.0
    @time begin
        _, z_optimal = resoudre_p_center_exact(tabX, tabY, p)
    end

    for methode in methodes
        println("Exécution de la méthode : ", methode)
        temps = @elapsed begin
            if methode == "Méthode exacte"
                _, z = resoudre_p_center_exact(tabX, tabY, p)
            elseif methode == "Solution aléatoire améliorée"
                _, z = trouver_meilleure_solution_aleatoire(n, p, tabX, tabY)
            elseif methode == "Relaxation linéaire"
                y_vals, _ = resoudre_p_center_relax(tabX, tabY, p)
                _, z = arrondir_solution_relax(y_vals, p, tabX, tabY)
            elseif methode == "Descente stochastique itérée"
                _, z = descente_stochastique_iteree(tabX, tabY, p, 100)
            elseif methode == "Branch and Bound"
                _, z = branch_and_bound(tabX, tabY, p)
            end
        end

        gap = (z - z_optimal) / z_optimal
        push!(resultats, (methode, z, gap, temps))
    end

    println(resultats)
    return resultats
end




# Fonction principale
function main()
    nom_fichier = "inst_300000.flp"
    tabX, tabY, f = Float64[], Float64[], Float64[]
    n = Lit_fichier_UFLP(nom_fichier, tabX, tabY, f)
    p = 3

    # Dessin des villes
    Dessine_UFLP(nom_fichier)

    println("Choisissez une méthode de résolution :")
    println("[1] : Méthode exacte")
    println("[2] : Solution aléatoire améliorée")
    println("[3] : Relaxation linéaire")
    println("[4] : Descente stochastique itérée")
    println("[5] : Branch and Bound")
    println("[6] : comparaison des méthodes")


    choix = parse(Int, readline())

    if choix == 1
        S, meilleur_rayon = resoudre_p_center_exact(tabX, tabY, p)
        Dessine_UFLP(nom_fichier, n, tabX, tabY, S)
    elseif choix == 2
        S, meilleur_rayon = trouver_meilleure_solution_aleatoire(n, p, tabX, tabY)
        Dessine_UFLP(nom_fichier, n, tabX, tabY, S)
    elseif choix == 3
        y_vals, _ = resoudre_p_center_relax(tabX, tabY, p)
        S, meilleur_rayon = arrondir_solution_relax(y_vals, p, tabX, tabY)
        println("Valeur de z après relaxation et arrondi : ", meilleur_rayon)
        Dessine_UFLP(nom_fichier, n, tabX, tabY, S)
    elseif choix == 4
        nb_iterations = 5
        S, meilleur_rayon = descente_stochastique_iteree(tabX, tabY, p, nb_iterations)
    elseif choix == 5
        S, meilleur_rayon = branch_and_bound(tabX, tabY, p)
        Dessine_UFLP(nom_fichier, n, tabX, tabY, S)
    elseif choix == 6
        comparer_methodes(nom_fichier, p)
    else
        println("Choix invalide")
        return
    end

    println("=> Meilleur rayon trouvé: ", meilleur_rayon)

end

main()
