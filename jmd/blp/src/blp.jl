module BLP

import NLsolve

include("integrate.jl")

@doc raw"""
    share(δ, Σ, dFν, x)

Computes shares in random coefficient logit with mean tastes `δ`, observed characteristics `x`, unobserved taste distribution `dFν`, and taste covariances `Σ`. 
Assumes there is an outside option with u=0. The outside option has share `1-sum(s)`

# Arguments

- `δ` vector of length `J`
- `Σ` `K` by `K` matrix
- `x` `J` by `K` array
- `∫` AbstractIntegrator for integrating over distribution of `ν`

# Returns

- vector of length `J` consisting of $s_1$, ..., $s_J$
"""
function share(δ, Σ, x, ∫::Integrate.AbstractIntegrator)
  J,K = size(x)
  (length(δ) == J) || error("length(δ)=$(length(δ)) != size(x,1)=$J")
  (K,K) == size(Σ) || error("size(x,2)=$K != size(Σ)=$(size(Σ))")
  function shareν(ν)
    s = δ .+ x*Σ*ν
    smax=max(0,maximum(s))
    s .-= smax
    s .= exp.(s)
    s ./= (sum(s) + exp(0-smax))
    return(s)
  end
  return(∫(shareν))
end

function delta(s, Σ, x, ∫)
  function eqns!(F,δ) 
    F .= s - share(δ,Σ,x,∫)
  end
  δ0 = log.(s) .- log(1-sum(s))
  sol=NLsolve.nlsolve(eqns!, δ0, autodiff=:forward, method=:trust_region)
  if (sol.residual_norm > 1e-4)
    @warn "Possible problem in delta(s, ...)\n".*"$sol"
  end
  return(sol.zero)
end

end