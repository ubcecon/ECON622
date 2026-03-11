module Integrate

import FastGaussQuadrature: gausshermite
import Distributions

abstract type AbstractIntegrator end

(∫::AbstractIntegrator)(f::Function) = #sum(∫.w[i]*f(∫.x[i]) for i in eachindex(∫.x))
    sum(w*f(x) for (x,w) in zip(∫.x, ∫.w))

struct FixedNodeIntegrator{Tx,Tw} <: AbstractIntegrator
    x::Tx
    w::Tw
end

function FixedNodeIntegrator(dx::Distributions.Normal, draws)
  (x,w) = gausshermite(draws)
  x = √2*x*dx.σ .+ dx.μ
  w ./= √π
  FixedNodeIntegrator(x, w)
end

end

using LinearAlgebra

import FastDifferentiation as FD
import SimpleNonlinearSolve
using StaticArrays
import Optim
import Distributions
using Statistics

function create_flowprofits(production=(k,l,m)->k^(0.2)*l^(0.4)*m^(0.3))
    profits(k,l,m,ω,pm,py,pl) = py*exp(ω)*production(k,l,m) - pm*m - pl*l
    FD.@variables k l m ω pm py pl
    ∇profits_sym = FD.jacobian([profits(k,l,m,ω,pm,py,pl)], [m, l])
    ∇profits = FD.make_function(∇profits_sym, [l, m, k, ω, pm, py, pl])
    F(x,p) = ∇profits(SVector{7,eltype(x)}(exp(x[1]),exp(x[2]),p[1],p[2],p[3],p[4],p[5]))
    prob = SimpleNonlinearSolve.NonlinearProblem(F,SVector{2}(0.0, 0.0), SVector{5}(1.0, 0.0, 1.0, 1.0, 1.0))
    function flowprofits(k, ω, pm, py, pl)
        res = SimpleNonlinearSolve.solve(prob, SimpleNonlinearSolve.SimpleTrustRegion(), p=SVector{5}(k, ω, pm, py, pl))
        l = exp(res.u[1])
        m = exp(res.u[2])
        return profits(k, l, m, ω, pm, py, pl)
    end
end


function setupgrid(kmin, kmax, nk, Fη, ωevolve; S=10_000)
    kgrid = range(kmin, kmax, length=nk)
    ωs = zeros(S)
    for s in 2:S
        ωs[s] = ωevolve(ωs[s-1], rand(Fη))
    end
    qgrid = range(0, 1, length=nk)
    ωgrid = quantile(ωs, qgrid)
    return ( (x,y)->((x,y)) ).(kgrid, ωgrid')[:]
end


function investmentdp(profits, cost, discount, Eη, Fη, ωevolve, interpolator;
    kmin, kmax, maxiter=1000, tol=1e-6,  nk=10, pm=1.0, py=1.0, pl=1.0)
    kω = setupgrid(kmin, kmax, nk, Fη, ωevolve)
    V = [profits(k,ω, pm, py, pl)/(1-discount) for (k,ω) in kω]
    policy = [x[1] for x in kω]
    for iter in 1:maxiter
        V₀ = copy(V)
        p₀ = copy(policy)
        𝒱 = interpolator(kω,V)
        for (i, (k, ω)) in enumerate(kω)
            obj(a) = -profits(k, ω, pm, py, pl) + cost(a,k) - discount * Eη(η->𝒱( ( a, ωevolve(ω, η) ) ) )
            res = Optim.optimize(obj, max(policy[i]-1,kmin), min(policy[i]+1,kmax), Optim.Brent())
            V[i] = -res.minimum
            policy[i] = res.minimizer
        end
        println("iter: $iter, ||V-V₀||: $(norm(V-V₀)), ||policy-p₀||: $(norm(policy-p₀))")
        if norm(V-V₀) < tol
            break
        end
    end
    return(𝒱=interpolator(kω,V), kω=kω, V=V, policy=policy)
end

import Surrogates


profitfn = create_flowprofits()
" quadratic adjustment cost with depreciation of 20% "
cost(a,k) = (a-0.8*k) + 0.5*(a-0.8*exp(k))^2
discount = 0.9
ση = 0.2
Fη = Distributions.Normal(0, ση)
Eη = Integrate.FixedNodeIntegrator(Fη, 10)
ωevolve(ω, η) = 0.8*ω + η
nk = 5
kmin = 0.01
kmax = 5.0
kω = setupgrid(kmin, kmax, nk, Fη, ωevolve)
lb = [kmin , minimum(x[2] for x in kω)]
ub = [kmax , maximum(x[2] for x in kω)]
interpolator(kω, V) = Surrogates.Kriging(kω, V, lb, ub; p=[2.0, 2.0])

investmentdp(profitfn, cost, discount, Eη, Fη, ωevolve, interpolator;
    kmin=kmin, kmax=kmax, maxiter=1000, tol=1e-6, nk=10)
