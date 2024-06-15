module TreeModule

using JuMP
using MathOptInterface
const MOI = MathOptInterface

export AbstractTree, initialize_tree, push!, processed!, update_best_bound!, branch!, next_node, termination, isempty, sort!

abstract type AbstractNode end
abstract type AbstractBranch end

mutable struct AbstractTree
    node_counter::Int
    nodes::Vector{AbstractNode}
    processed::Vector{AbstractNode}
    best_bound::Real
    best_incumbent::Real

    function AbstractTree(node_counter::Int = 0)
        return new(node_counter, [], [], -Inf, Inf)
    end
end

function initialize_tree(root::JuMPNode{T})::AbstractTree where T<:AbstractBranch
    tree = AbstractTree()
    push!(tree, root)
    return tree
end

import Base: push!, isempty, sort!

function push!(tree::AbstractTree, node::AbstractNode)
    tree.node_counter += 1
    node.id = tree.node_counter
    Base.push!(tree.nodes, node)
    sort!(tree)
end

function push!(tree::AbstractTree, nodes::Vector{T}) where T<:AbstractNode
    for node in nodes
        push!(tree, node)
    end
end

function processed!(tree::AbstractTree, node::AbstractNode)
    Base.push!(tree.processed, node)
end

function update_best_bound!(tree::AbstractTree)
    if !isempty(tree)
        tree.best_bound = Base.minimum([node.bound for node in tree.nodes])
    end
end

function branch!(tree::AbstractTree, node::AbstractNode)
    children = branch(node)
    push!(tree, children)
end

function next_node(tree::AbstractTree)
    sort!(tree)
    node = Base.pop!(tree.nodes)
    apply_changes!(node)
    return node
end

function termination(tree::AbstractTree)
    return isempty(tree)
end

function isempty(tree::AbstractTree)
    return Base.isempty(tree.nodes)
end

function sort!(tree::AbstractTree)
    Base.sort!(tree.nodes, by=x->x.bound, rev=true)
end

end