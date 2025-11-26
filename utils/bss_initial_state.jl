using DifferentialEquations
using Interpolations
using Distributions
using JuMP
using Ipopt
using JLD


mutable struct InitialStateSolver
  m::Model
  csp_s::VariableRef
  csn_s::VariableRef
  phi_p::VariableRef
  phi_n::VariableRef
  it::VariableRef
  iint::VariableRef
  Up::VariableRef
  Un::VariableRef
  theta_p::VariableRef
  theta_n::VariableRef
  csp_avg0::VariableRef
  csn_avg0::VariableRef
  delta_sei0::VariableRef
  power::VariableRef
end

function get_initial_state_solver()
  m = Model(optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0,
    "linear_solver" => "mumps",
    "max_cpu_time" => 600.0,
  ))

  @variable(m, csp_avg0)
  @variable(m, csn_avg0)
  @variable(m, delta_sei0)
  @variable(m, power)

  @variable(m, it)
  @variable(m, iint)

  @variable(m, csp_s)
  @variable(m, csn_s)

  @constraint(m, 5 * (csp_s - csp_avg0) + Rpp * it / F / Dp / ap / lp == 0)
  @constraint(m, 5 * (csn_s - csn_avg0) - Rpn * iint / F / Dn / an / lnn == 0)


  @variable(m, 0 <= theta_p <= 1)
  @variable(m, 0 <= theta_n <= 1)

  @constraint(m, theta_p * cspmax == csp_s)
  @constraint(m, theta_n * csnmax == csn_s)

  @variable(m, Up)
  @variable(m, phi_p)
  @variable(m, Un)
  @variable(m, phi_n)
  @NLconstraint(m, (cspmax - csp_s)^(0.5) * csp_s^(0.5) * sinh(0.5 * F / R / T * (phi_p - Up)) - (it / ap / F / lp / (2 * kp * ce^(0.5))) == 0)
  @NLconstraint(m, (csnmax - csn_s)^(0.5) * csn_s^(0.5) * sinh(0.5 * F / R / T * (phi_n - Un + (delta_sei0 / Kappa_sei + Rsei) * it / an / lnn)) + iint / an / F / lnn / (2 * kn * ce^(0.5)) == 0)

  @NLconstraint(m, Up == 7.49983 - 13.7758 * theta_p^0.5 + 21.7683 * theta_p - 12.6985 * theta_p^1.5 + 0.0174967 / theta_p - 0.41649 * theta_p^(-0.5) -
                         0.0161404 * exp(100 * theta_p - 97.1069) + 0.363031 * tanh(5.89493 * theta_p - 4.21921))
  @NLconstraint(m, Un == 9.99877 - 9.99961 * theta_n^0.5 - 9.98836 * theta_n + 8.2024 * theta_n^1.5 + 0.23584 / theta_n - 2.03569 * theta_n^(-0.5) -
                         1.47266 * exp(-1.14872 * theta_n + 2.13185) - 9.9989 * tanh(0.60345 * theta_n - 1.58171))

  isei = it - iint
  @NLconstraint(m, 1e4 * (-isei + an * lnn * ksei * exp(-1 * F / R / T * (phi_n - Urefs + (delta_sei0 / Kappa_sei + Rsei) * it / an / lnn))) == 0)
  @constraint(m, it * (phi_p - phi_n) == power)

  return InitialStateSolver(m, csp_s, csn_s, phi_p, phi_n, it, iint, Up, Un, theta_p, theta_n, csp_avg0, csn_avg0, delta_sei0, power)
end


function reset_solver!(solver::InitialStateSolver, u0, power)
  csp_avg0, csn_avg0, delta_sei0 = u0[1], u0[Ncp+1], u0[Ncp+Ncn+7]

  theta_p_guess = min(0.9, csp_avg0 / cspmax)
  theta_n_guess = min(0.9, csn_avg0 / csnmax)

  # initial guess
  Un_guess = 9.99877 - 9.99961 * theta_n_guess .^ 0.5 - 9.98836 * theta_n_guess + 8.2024 * theta_n_guess .^ 1.5 + 0.23584 ./ theta_n_guess - 2.03569 * theta_n_guess .^ (-0.5) -
             1.47266 * exp(-1.14872 * theta_n_guess + 2.13185) -
             9.9989 * tanh(0.60345 * theta_n_guess - 1.58171)
  Up_guess = 7.49983 - 13.7758 * theta_p_guess .^ 0.5 + 21.7683 * theta_p_guess - 12.6985 * theta_p_guess .^ 1.5 + 0.0174967 ./ theta_p_guess - 0.41649 * theta_p_guess .^ (-0.5) -
             0.0161404 * exp(100 * theta_p_guess - 97.1069) +
             0.363031 * tanh(5.89493 * theta_p_guess - 4.21921)

  pot0 = max(min(Up_guess - Un_guess, 3.3), 2.0)
  it0 = power / pot0

  set_start_value(solver.it, it0)
  set_start_value(solver.iint, it0)
  set_start_value(solver.csp_s, csp_avg0)
  set_start_value(solver.csn_s, csn_avg0)
  set_start_value(solver.Up, Up_guess)
  set_start_value(solver.Un, Un_guess)
  set_start_value(solver.phi_p, Up_guess)
  set_start_value(solver.phi_n, Un_guess)
  set_start_value(solver.theta_p, theta_p_guess)
  set_start_value(solver.theta_n, theta_n_guess)
  fix(solver.csp_avg0, csp_avg0)
  fix(solver.csn_avg0, csn_avg0)
  fix(solver.delta_sei0, delta_sei0)
  fix(solver.power, power)
end


function compute_initial_state(solver::InitialStateSolver, u0, power)
  start_time = time()
  reset_solver!(solver, u0, power)
  optimize!(solver.m)
  status = is_solved_and_feasible(solver.m)
  if !status
    println("initial-state solver failed, elapsed time:  ", time() - start_time)
  end

  csp_s0 = value(solver.csp_s)
  csn_s0 = value(solver.csn_s)
  phi_p0 = value(solver.phi_p)
  phi_n0 = value(solver.phi_n)
  pot0 = phi_p0 - phi_n0
  it0 = value(solver.it)
  iint0 = value(solver.iint)
  isei0 = it0 - iint0

  u0[Ncp] = csp_s0
  u0[Ncp+Ncn] = csn_s0
  u0[Ncp+Ncn+1] = iint0
  u0[Ncp+Ncn+2] = phi_p0
  u0[Ncp+Ncn+3] = phi_n0
  u0[Ncp+Ncn+4] = pot0
  u0[Ncp+Ncn+5] = it0
  u0[Ncp+Ncn+6] = isei0

  return u0
end