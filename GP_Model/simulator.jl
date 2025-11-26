using DelimitedFiles
using Sundials
using Interpolations
using ModelingToolkit
using BlockDiagonals
using LinearAlgebra
using SymbolicIndexingInterface: parameter_values

include("../utils/setup.jl")
include("../utils/utils.jl")


# Dynamics
function f_common(out, du, u, p, t)
  csp = u[1:Ncp]
  csn = u[(Ncp+1):(Ncp+Ncn)]
  csp = max.(1, min.(csp, cspmax - 1))
  csn = max.(1, min.(csn, csnmax - 1))
  csp_avg = csp[1]
  csp_s = csp[2]
  csn_avg = csn[1]
  csn_s = csn[2]

  iint = u[Ncp+Ncn+1]
  phi_p = u[Ncp+Ncn+2]
  phi_n = u[Ncp+Ncn+3]
  pot = u[Ncp+Ncn+4]

  it = u[Ncp+Ncn+5]
  isei = u[Ncp+Ncn+6]
  delta_sei = u[Ncp+Ncn+7]

  cf = u[Ncp+Ncn+Nsei+5]

  #C2. Additional Equaions
  #Positive electrode
  theta_p = csp_s / cspmax
  Up = 7.49983 - 13.7758 * theta_p^0.5 + 21.7683 * theta_p - 12.6985 * theta_p^1.5 + 0.0174967 / theta_p - 0.41649 * theta_p^(-0.5) - 0.0161404 * exp(100 * theta_p - 97.1069) +
       0.363031 * tanh(5.89493 * theta_p - 4.21921)
  jp = 2 * kp * ce^(0.5) * (cspmax - csp_s)^(0.5) * csp_s^(0.5) * sinh(0.5 * F / R / T * (phi_p - Up))

  #Negative electrode
  theta_n = csn_s / csnmax
  Un = 9.99877 - 9.99961 * theta_n^0.5 - 9.98836 * theta_n + 8.2024 * theta_n^1.5 + 0.23584 / theta_n - 2.03569 * theta_n^(-0.5) -
       1.47266 * exp(-1.14872 * theta_n + 2.13185) -
       9.9989 * tanh(0.60345 * theta_n - 1.58171)
  jn = 2 * kn * ce^(0.5) * (csnmax - csn_s)^(0.5) * csn_s^(0.5) * sinh(0.5 * F / R / T * (phi_n - Un + (Rsei + delta_sei / Kappa_sei) * it / an / lnn))

  #C1. Governing Equations
  #Positive electrode
  out[1] = -3 * jp / Rpp - du[1]
  out[2] = 5 * (csp_s - csp_avg) + Rpp * jp / Dp

  #Negative electrode
  out[3] = -3 * jn / Rpn - du[Ncp+1]
  out[4] = 5 * (csn_s - csn_avg) + Rpn * jn / Dn

  out[5] = (jp - (it / ap / F / lp))
  out[6] = (jn + (iint / an / F / lnn))
  out[7] = pot - phi_p + phi_n
  out[8] = p - pot * it

  #C3. SEI layer Equations
  out[9] = -iint + it - isei
  out[10] = -isei + an * lnn * ksei * exp(-1 * F / R / T * (phi_n - Urefs + it / an / lnn * (delta_sei / Kappa_sei + Rsei)))
  out[11] = isei * M_sei / F / rho_sei / an / lnn - du[Ncp+Ncn+4+3]

  #C4. Charge stored
  out[12] = isei / 3600 - du[Ncp+Ncn+Nsei+5]
end


mutable struct Simulator
  prob
  tspan

  function Simulator(u0, battery_charge)

    TIME_SEGMENT = 0:dt:DT_STATE
    tspan = (float(TIME_SEGMENT[1]), float(TIME_SEGMENT[end]))

    differential_vars = falses(Ncp + Ncn + 4 + Nsei + Ncum)
    differential_vars[1] = true
    differential_vars[3] = true
    differential_vars[11] = true
    differential_vars[12] = true

    prob = SciMLBase.DAEProblem(f_common, zero(u0), u0, tspan, battery_charge, differential_vars=differential_vars)

    new(prob, tspan)
  end
end


function simulate(simulator, u0, grid_price, battery_charge, swap_battery_idx, swap_battery_state)

  start_time = time()

  u1 = zero(u0)
  # state mutation if any battery is selected for swapping
  if !isnothing(swap_battery_idx)
    u1[swap_battery_idx, 1] = swap_battery_state[1]
    u1[swap_battery_idx, 3] = swap_battery_state[2]
    u1[swap_battery_idx, Ncp+Ncn+4+3] = swap_battery_state[3]
    u1[swap_battery_idx, Ncp+Ncn+Nsei+5] = swap_battery_state[4]
  end

  cf0 = u0[:, Ncp+Ncn+Nsei+5]
  fade = cf0 ./ Qmax
  capacity_remain = 1 .- fade

  success = true
  for k in 1:NUM_BATTERIES_IN_STATION
    if k != swap_battery_idx

      function stop_cond(u, t, integrator)
        csn_avg = u[Ncp+1]
        soc_in = csn_avg / csnmax
        min_stop = capacity_remain[k] * soc_min_stop
        max_stop = capacity_remain[k] * soc_max_stop

        if soc_in >= (min_stop + max_stop) / 2
          out = max_stop - soc_in
        else
          out = soc_in - min_stop
        end
      end
      affect!(integrator) = terminate!(integrator)
      cb = ContinuousCallback(stop_cond, affect!, rootfind=true, interp_points=100)


      prob = remake(simulator.prob, u0=copy(u0[k, :]), p=battery_charge[k])
      sol = DifferentialEquations.solve(prob, IDA(), verbose=false, callback=cb)
      csn_avg = (sol[end])[Ncp+1]
      soc_end = csn_avg / csnmax

      if sol.t[end] != simulator.tspan[end] || soc_end <= soc_min_stop || soc_end >= soc_max_stop
        success = false
        # println("soc too large or small, stop at ", sol.t[end], "  soc_end:  ", soc_end)
      else
        u1[k, :] .= sol[end]
      end
    end
  end

  profit = -grid_price * sum(battery_charge)
  # println("profit:  ", profit)
  # for k in 1:NUM_BATTERIES_IN_STATION
  #   println("u1_sim[$k]:  ", u1[k, [1, 3, 11, 12]])
  # end

  # elapsed = time() - start_time
  # println("simulation elapsed time:     ", elapsed)

  return u1, profit, success
end