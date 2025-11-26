using LinearAlgebra
using JuMP


mutable struct Kriging{X,Y,P,T,M,B,S,R,x_means,x_stds,y_std}
  x::X
  y::Y
  p::P
  theta::T
  mu::M
  b::B
  sigma::S
  inverse_of_R::R
  x_means::x_means
  x_stds::x_stds
  y_std::y_std
end


"""
Gives the current estimate for array 'val' with respect to the Kriging object k.
"""
function (k::Kriging)(val::Vector{Float64})

  n = length(k.x)
  d = length(val)

  return k.mu +
         sum(k.b[i] *
             exp(-sum(k.theta[j] * (val[j] - k.x[i][j])^k.p for j in 1:d))
             for i in 1:n)
end

function (k::Kriging)(val::Vector{AffExpr})

  n = length(k.x)
  d = length(val)

  return k.mu +
         sum(k.b[i] *
             exp(-sum(k.theta[j] * (val[j] - k.x[i][j])^k.p for j in 1:d))
             for i in 1:n)
end

function pred(k::Kriging, x)
  x = (x .- k.x_means) ./ k.x_stds
  return k(x) .* k.y_std
end

"""
  Returns sqrt of expected mean_squared_error error at the point.
"""
function std_error_at_point(k::Kriging, val::Vector{Float64})

  n = length(k.x)
  d = length(k.x[1])
  r = zeros(eltype(k.x[1]), n, 1)
  r = [
    let
      sum = zero(eltype(k.x[1]))
      for l in 1:d
        sum = sum + k.theta[l] * norm(val[l] - k.x[i][l])^(k.p)
      end
      exp(-sum)
    end
    for i in 1:n
  ]

  one = ones(eltype(k.x[1]), n, 1)
  one_t = one'
  a = r' * k.inverse_of_R * r
  b = one_t * k.inverse_of_R * one

  mean_squared_error = k.sigma * (1 - a[1] + (1 - a[1])^2 / b[1])
  return sqrt(abs(mean_squared_error))
end


function Kriging(x::Vector{Vector{Float64}}, y::Vector{Float64}, theta::Vector{Float64}, x_means::Vector{Float64}, x_stds::Vector{Float64}, y_std::Float64)
  if length(x) != length(unique(x))
    println("There exists a repetition in the samples, cannot build Kriging.")
    return
  end

  for i in 1:length(x[1])
    if theta[i] ≤ 0.0
      throw(ArgumentError("All theta must be positive! Got: $theta."))
    end
  end

  mu, b, sigma, inverse_of_R = _calc_kriging_coeffs(x, y, theta)
  Kriging(x, y, 2, theta, mu, b, sigma, inverse_of_R, x_means, x_stds, y_std)
end


function _calc_kriging_coeffs(x::Vector{Vector{Float64}}, y::Vector{Float64}, theta::Vector{Float64})
  n = length(x)
  d = length(x[1])

  R = [
    let
      sum = zero(eltype(x[1]))
      for l in 1:d
        sum = sum + theta[l] * norm(x[i][l] - x[j][l])^2
      end
      exp(-sum)
    end
    for j in 1:n, i in 1:n
  ]

  # Estimate nugget based on maximum allowed condition number
  # This regularizes R to allow for points being close to each other without R becoming
  # singular, at the cost of slightly relaxing the interpolation condition
  # Derived from "An analytic comparison of regularization methods for Gaussian Processes"
  # by Mohammadi et al (https://arxiv.org/pdf/1602.00853.pdf)
  λ = eigen(R).values

  λmax = λ[end]
  λmin = λ[1]

  κmax = 1e8
  λdiff = λmax - κmax * λmin
  if λdiff ≥ 0
    nugget = λdiff / (κmax - 1)
  else
    nugget = 0.0
  end

  one = ones(eltype(x[1]), n)
  one_t = one'

  R = R + Diagonal(nugget * one[:, 1])
  inverse_of_R = inv(R)

  mu = (one_t * inverse_of_R * y) / (one_t * inverse_of_R * one)

  y_minus_1μ = y - one * mu

  b = inverse_of_R * y_minus_1μ

  sigma = (y_minus_1μ' * b) / n

  mu[1], b, sigma[1], inverse_of_R
end


function log_likelihood(params::Vector{Float64}, x::Vector{Vector{Float64}}, y::Vector{Float64})
  θ = params
  n = length(y)

  mu, b, σ, inverse_of_R = _calc_kriging_coeffs(x, y, θ)

  term1 = -0.5 * n * log(σ)
  term2 = 0.5 * logdet(inverse_of_R)

  return -(term1 + term2)  # return negative for Optim
end


function train_kriging(x::Vector{Vector{Float64}}, y::Vector{Float64})

  num_dims = length(x[1])
  init_params = fill(1.0, num_dims)

  lower_bounds = fill(1e-3, num_dims)
  upper_bounds = fill(100, num_dims)

  result = optimize(
    p -> log_likelihood(p, x, y),
    lower_bounds, upper_bounds,
    init_params,
    Fminbox(NelderMead())
  )
  θ_opt = result.minimizer
  return θ_opt
end