using Graphs
using Distances
using Zygote
using Plots
using PlotThemes
using LinearAlgebra

function plot_graph(g::SimpleGraph, p::Matrix{Float64})
    plt = plot(legend = false, aspect_ratio = :equal)
    for e in edges(g)
        plot!(plt, [p[1, src(e)], p[1, dst(e)]], [p[2, src(e)], p[2, dst(e)]],
                color = :black)
    end
    scatter!(plt, p[1, :], p[2, :], markersize=5, color = :blue)
    return plt
end

function plot_graph_3d(g::SimpleGraph, p::Matrix{Float64})
    plt = plot(legend = false, aspect_ratio = :equal)
    for e in edges(g)
        plot!(plt, [p[1, src(e)], p[1, dst(e)]], [p[2, src(e)], p[2, dst(e)]], [p[3, src(e)], p[3, dst(e)]],
                color = :black)
    end
    scatter!(plt, p[1, :], p[2, :], p[3, :], markersize=5, color = :blue)
    return plt
end

function stress(
    g::SimpleGraph, 
    edge_stress_f::Function, 
    vertex_stress_f::Function, 
    combine_f::Function)

    n = nv(g)
    edge_stress = 0
    for e in edges(g)
        edge_stress += edge_stress_f(src(e), dst(e))
    end
    vertex_stress = 0
    for i in 1:n
        for j in (i+1):n
            vertex_stress += vertex_stress_f(i, j)
        end
    end

    return combine_f(edge_stress, vertex_stress)
end

function fr_stress(g::SimpleGraph, p::Matrix{Float64})
    D = pairwise(Euclidean(), p, dims=2)
    edge_stress_f = (i, j) -> D[i, j]^3
    vertex_stress_f = (i, j) -> log(D[i, j] + 10^(-6))
    combine_f = (x, y) -> x / 6 - y
    return stress(g, edge_stress_f, vertex_stress_f, combine_f)
end


function erll_stress(g::SimpleGraph, p::Matrix{Float64}, deg::Vector{Int})
    D = pairwise(Euclidean(), p, dims=2)
    edge_stress_f = (i, j) -> D[i, j]
    vertex_stress_f = (i, j) -> deg[i] * deg[j] * log(D[i, j] + 10^(-6))
    combine_f = (x, y) -> x / 2 - y
    return stress(g, edge_stress_f, vertex_stress_f, combine_f)
end

function simple_gradient_descent(g::SimpleGraph, alpha::Float64, max_steps::Int64)
    n = nv(g) # Number of vertecies
    deg = degree(g) # The degrees of the edges
    p = rand(2, n) # Initial positions

    stresses = Float64[]

    for k in 1:max_steps
        s, (grad,) = Zygote.withgradient(q -> erll_stress(g, q, deg), p)
        push!(stresses, s)

        if length(stresses) > 1 && abs(stresses[end] - stresses[end-1]) < 1e-6
            println("converged!")
            println("iteration ", k)
            plot1 = plot_graph(g, p)
            plot2 = plot(stresses, label="stress")
            final = plot(plot1, plot2, layout=2)
            savefig(final, "converged.svg")
            break
        end

        p -= alpha * grad

        if k % 30 == 0
            plot1 = plot_graph(g, p)
            plot2 = plot(stresses, label="stress")
            display(plot(plot1, plot2, layout=2))
        end
    end
end

function nesterov_gradient_descent(g::SimpleGraph, alpha::Float64, beta::Float64, max_steps::Int64)
    n = nv(g) # Number of vertecies
    deg = degree(g) # The degrees of the edges
    p = rand(2, n) # Initial positions

    stresses = Float64[]

    momentum = zeros(2, n)

    for k in 0:max_steps
        s, (grad,) = Zygote.withgradient(q -> fr_stress(g, q), p + beta * momentum)
        new_momentum = beta * momentum - alpha * grad
        push!(stresses, s)

        if length(stresses) > 50 && abs(stresses[end] - stresses[end-1]) < 1e-7
            println("converged!")
            println("iteration ", k)
            plot1 = plot_graph(g, p)
            plot2 = plot(stresses, label="stress")
            final = plot(plot1, plot2, layout=2)
            savefig(final, "converged.svg")
            break
        end

        p += new_momentum
        momentum = new_momentum

        if k % 30 == 0
            plot1 = plot_graph(g, p)
            plot2 = plot(stresses, label="stress")
            display(plot(plot1, plot2, layout=2))
        end
    end
end

function run(g)
    nesterov_gradient_descent(g, 0.001, 0.9, 1000000)
end

run(static_scale_free(14, 40, 5))
