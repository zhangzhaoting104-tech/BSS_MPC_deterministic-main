using DifferentialEquations
using PyPlot
using Interpolations
using Distributions
using JuMP
using GAMS

include("../utils/setup.jl")
include("../utils/utils.jl")


mutable struct OptimalControlSolver
  # Model
  m::Model

  # Parameters
  grid_price::Any
  swap_battery_times::Any
  swap_battery_states::Any
  cf_min::Any

  # Decision variables
  csp_avg_scaled::Any
  csn_avg_scaled::Any
  delta_sei_scaled::Any
  cf_scaled::Any
  bat_charge_scaled::Any

  csp_avg::Any
  csn_avg::Any
  delta_sei::Any
  cf::Any
  bat_charge::Any

  eps::Any

  profit::Any
  penalty::Any

  # Scaling Parameters
  csp_avg_scale::Any
  csn_avg_scale::Any
  delta_sei_scale::Any
  cf_scale::Any
  bat_charge_scale::Any
  csp_avg_offset::Any
  csn_avg_offset::Any
  delta_sei_offset::Any
  cf_offset::Any
  bat_charge_offset::Any

  # warm-start buffer
  last_bat_charge_scaled::Any
  last_csp_avg_scaled::Any
  last_csn_avg_scaled::Any
  last_delta_sei_scaled::Any
  last_cf_scaled::Any
  last_eps::Any
end

function get_optimal_control_solver(krigin_surrogate)

  ############
  ## solver ##
  ############
  m = Model(GAMS.Optimizer)
  set_optimizer_attribute(m, GAMS.ModelType(), "MINLP")
  set_optimizer_attribute(m, "MINLP", "dicopt")
  set_optimizer_attribute(m, "NLP", "ipopth")
  set_optimizer_attribute(m, "solver", "ipopth")
  set_optimizer_attribute(m, "linear_solver", "ma86")
  set_optimizer_attribute(m, "MIP", "cplex")
  set_optimizer_attribute(m, "HoldFixed", 1)
  set_optimizer_attribute(m, "threads", 18)
  # set_optimizer_attribute(m, "resLim", 120)
  set_optimizer_attribute(m, MOI.Silent(), true)
  set_optimizer_attribute(m, GAMS.WorkDir(), "D:\\GAMS\\GAMSWorkspace\\tmp\\original")

  ##########
  ## init ##
  ##########
  csp_avg_scale = x_stds[1]
  csn_avg_scale = x_stds[2]
  delta_sei_scale = x_stds[3]
  cf_scale = x_stds[4]
  csp_avg_offset = x_means[1]
  csn_avg_offset = x_means[2]
  delta_sei_offset = x_means[3]
  cf_offset = x_means[4]
  bat_charge_scale = x_stds[5]
  bat_charge_offset = x_means[5]

  # csp_avg_scale = state_max[1] - state_min[1]
  # csn_avg_scale = state_max[2] - state_min[2]
  # delta_sei_scale = state_max[3] - state_min[3]
  # cf_scale = state_max[4] - state_min[4]
  # csp_avg_offset = state_min[1]
  # csn_avg_offset = state_min[2]
  # delta_sei_offset = state_min[3]
  # cf_offset = state_min[4]
  # bat_charge_scale = input_max - input_min
  # bat_charge_offset = input_min


  ###########
  ## input ##
  ###########
  @variable(m, bat_charge_scaled[k in 1:NUM_BATTERIES_IN_STATION, t in 2:Nt_INPUT+1])


  ###########
  ## state ##
  ###########
  @variable(m, csp_avg_scaled[k in 1:NUM_BATTERIES_IN_STATION, t in 1:Nt_STATE])
  @variable(m, csn_avg_scaled[k in 1:NUM_BATTERIES_IN_STATION, t in 1:Nt_STATE])
  @variable(m, delta_sei_scaled[k in 1:NUM_BATTERIES_IN_STATION, t in 1:Nt_STATE])
  @variable(m, cf_scaled[k in 1:NUM_BATTERIES_IN_STATION, t in 1:Nt_STATE])


  #########################
  ## selective variables ##
  #########################
  @variable(m, eps[k in 1:NUM_BATTERIES_IN_STATION, t in 1:Nt_STATE], Bin)


  ############
  ## revert ##
  ############
  bat_charge = @expression(m, bat_charge_scaled .* bat_charge_scale .+ bat_charge_offset)
  csp_avg = @expression(m, csp_avg_scaled .* csp_avg_scale .+ csp_avg_offset)
  csn_avg = @expression(m, csn_avg_scaled .* csn_avg_scale .+ csn_avg_offset)
  delta_sei = @expression(m, delta_sei_scaled .* delta_sei_scale .+ delta_sei_offset)
  cf = @expression(m, cf_scaled .* cf_scale .+ cf_offset)


  ################
  ## Parameters ##
  ################
  @variable(m, grid_price[t in 2:Nt_INPUT+1] in Parameter(0))
  @variable(m, swap_battery_times[t in 1:Nt_STATE] in Parameter(0))
  @variable(m, swap_battery_states[t in 1:Nt_STATE, i in 1:4] in Parameter(0))
  @variable(m, cf_min[t in 1:1] in Parameter(0))


  ##############
  ## Dynamics ##
  ##############
  for t in 1:Nt_STATE-1
    for k in 1:NUM_BATTERIES_IN_STATION
      x = vcat(csp_avg[k, t+1], csn_avg[k, t+1], delta_sei[k, t+1], cf[k, t+1], bat_charge[k, t+1])
      csp_avg_pred = pred(krigin_surrogate[1], x)
      csn_avg_pred = pred(krigin_surrogate[2], x)
      delta_sei_pred = pred(krigin_surrogate[3], x)
      cf_pred = pred(krigin_surrogate[4], x)

      @constraint(m, csn_avg[k, t] / csnmax >= (replaceable_soc + 0.001) * eps[k, t])

      @constraint(m, 0 == csp_avg[k, t+1] + csp_avg_pred .- jump_state(csp_avg[k, t], swap_battery_states[t, 1], eps[k, t]))
      @constraint(m, 0 == csn_avg[k, t+1] + csn_avg_pred .- jump_state(csn_avg[k, t], swap_battery_states[t, 2], eps[k, t]))
      @constraint(m, 0 == delta_sei[k, t+1] + delta_sei_pred .- jump_state(delta_sei[k, t], swap_battery_states[t, 3], eps[k, t]))
      @constraint(m, 0 == cf[k, t+1] + cf_pred .- jump_state(cf[k, t], swap_battery_states[t, 4], eps[k, t]))
    end
    @constraint(m, sum(eps[:, t+1]) >= swap_battery_times[t+1] - 0.9)
    @constraint(m, sum(eps[:, t+1]) <= swap_battery_times[t+1] + 0.9)
  end


  ###############
  ## Objective ##
  ###############
  ############## profit ##############
  profit = -sum(sum(bat_charge[k, :] for k in 1:NUM_BATTERIES_IN_STATION) .* grid_price)

  ############## penalty ##############
  penalty = 0
  # coeffi1 = 1e3
  # for t in 1:Nt_STATE-1
  #   if t == 1
  #     penalty = penalty - sum(cf[:, t+1] .- cf[:, t]) * 1e6
  #   else
  #     penalty = penalty - sum((1 .- eps[:, t]) .* (cf[:, t+1] .- cf[:, t])) * coeffi1 * 0.9^(t - 1)
  #     penalty = penalty - sum(eps[:, t] .* (cf[:, t+1] .- swap_battery_states[t, 4])) * coeffi1 * 0.9^(t - 1)
  #   end
  # end
  coeffi2 = 1e3
  for t in 2:Nt_STATE
    penalty = penalty - sum((1 .- eps[:, t]) .* (cf[:, t] .- cf_min)) * coeffi2
  end

  ############## obj ##############
  obj = profit + penalty
  @objective(m, Max, obj)

  OptimalControlSolver(m::Model,
    grid_price, swap_battery_times, swap_battery_states, cf_min,
    csp_avg_scaled, csn_avg_scaled, delta_sei_scaled, cf_scaled, bat_charge_scaled,
    csp_avg, csn_avg, delta_sei, cf, bat_charge, eps, profit, penalty,
    csp_avg_scale, csn_avg_scale, delta_sei_scale, cf_scale, bat_charge_scale,
    csp_avg_offset, csn_avg_offset, delta_sei_offset, cf_offset, bat_charge_offset, [], [], [], [], [], [])
end


function get_scaled_params(solver::OptimalControlSolver, csp_avg0, csn_avg0, delta_sei0, cf0)

  back = 0.01
  capacity_remain = 1 .- cf0 ./ Qmax
  csp_avg_max, csn_avg_max, delta_sei_max, cf_max = cspmax - 1, maximum(capacity_remain) * (soc_max_stop - back) * csnmax, 3e-7, 1.48
  csp_avg_min, csn_avg_min, delta_sei_min, cf_min = 1, minimum(capacity_remain) * (back + soc_min_stop) * csnmax, 0.0, 0.0

  csp_avg0_scaled = (csp_avg0 .- solver.csp_avg_offset) ./ solver.csp_avg_scale
  csn_avg0_scaled = (csn_avg0 .- solver.csn_avg_offset) ./ solver.csn_avg_scale
  delta_sei0_scaled = (delta_sei0 .- solver.delta_sei_offset) ./ solver.delta_sei_scale
  cf0_scaled = (cf0 .- solver.cf_offset) ./ solver.cf_scale

  csp_avg_lb_scaled = (csp_avg_min .- solver.csp_avg_offset) ./ solver.csp_avg_scale
  csp_avg_ub_scaled = (csp_avg_max .- solver.csp_avg_offset) ./ solver.csp_avg_scale
  csn_avg_lb_scaled = (capacity_remain .* (back .+ soc_min_stop) .* csnmax .- solver.csn_avg_offset) ./ solver.csn_avg_scale
  csn_avg_ub_scaled = (capacity_remain .* (soc_max_stop .- back) .* csnmax .- solver.csn_avg_offset) ./ solver.csn_avg_scale
  csn_avg_lb_end_scaled = (capacity_remain .* soc_min .* csnmax .- solver.csn_avg_offset) ./ solver.csn_avg_scale
  csn_avg_ub_end_scaled = (capacity_remain .* soc_max .* csnmax .- solver.csn_avg_offset) ./ solver.csn_avg_scale
  delta_sei_lb_scaled = (delta_sei_min .- solver.delta_sei_offset) ./ solver.delta_sei_scale
  delta_sei_ub_scaled = (delta_sei_max .- solver.delta_sei_offset) ./ solver.delta_sei_scale
  cf_lb_scaled = (cf_min .- solver.cf_offset) ./ solver.cf_scale
  cf_ub_scaled = (cf_max .- solver.cf_offset) ./ solver.cf_scale

  bat_charge_lb_scaled = (-P_max .- solver.bat_charge_offset) ./ solver.bat_charge_scale
  bat_charge_ub_scaled = (P_max .- solver.bat_charge_offset) ./ solver.bat_charge_scale

  return (csp_avg0_scaled, csn_avg0_scaled, delta_sei0_scaled, cf0_scaled,
    csp_avg_lb_scaled, csp_avg_ub_scaled, csn_avg_lb_scaled, csn_avg_ub_scaled, csn_avg_lb_end_scaled, csn_avg_ub_end_scaled,
    delta_sei_lb_scaled, delta_sei_ub_scaled, cf_lb_scaled, cf_ub_scaled, bat_charge_lb_scaled, bat_charge_ub_scaled)
end


function set_parameters!(solver::OptimalControlSolver, grid_price, swap_battery_times, swap_battery_states, cf_min)
  set_parameter_value.(solver.grid_price, grid_price)
  set_parameter_value.(solver.swap_battery_times, swap_battery_times)
  set_parameter_value.(solver.swap_battery_states, swap_battery_states)
  set_parameter_value.(solver.cf_min, cf_min)
end


function set_initial_guess!(solver, swap_battery_times, csp_avg0_scaled, csn_avg0_scaled, delta_sei0_scaled, cf0_scaled, start_point)
  set_start_value.(solver.bat_charge_scaled, get_bat_charge_initial_guess(NUM_BATTERIES_IN_STATION, Nt_INPUT, start_point, solver.last_bat_charge_scaled))
  set_start_value.(solver.csp_avg_scaled, get_state_initial_guess(start_point, solver.last_csp_avg_scaled; state0=csp_avg0_scaled))
  set_start_value.(solver.csn_avg_scaled, get_state_initial_guess(start_point, solver.last_csn_avg_scaled; state0=csn_avg0_scaled))
  set_start_value.(solver.delta_sei_scaled, get_state_initial_guess(start_point, solver.last_delta_sei_scaled; state0=delta_sei0_scaled))
  set_start_value.(solver.cf_scaled, get_state_initial_guess(start_point, solver.last_cf_scaled; state0=cf0_scaled))
  set_start_value.(solver.eps, get_eps_initial_guess(NUM_BATTERIES_IN_STATION, Nt_STATE, swap_battery_times, start_point, solver.last_eps))
end


function set_bounds!(solver::OptimalControlSolver, bat_charge_lb_scaled, bat_charge_ub_scaled, csp_avg_lb_scaled, csp_avg_ub_scaled,
  csn_avg_lb_scaled, csn_avg_ub_scaled, delta_sei_lb_scaled, delta_sei_ub_scaled, cf_lb_scaled, cf_ub_scaled,
  csn_avg_lb_end_scaled, csn_avg_ub_end_scaled)
  for t in 2:Nt_STATE
    set_lower_bound.(solver.bat_charge_scaled[:, t], bat_charge_lb_scaled)
    set_upper_bound.(solver.bat_charge_scaled[:, t], bat_charge_ub_scaled)
    set_lower_bound.(solver.csp_avg_scaled[:, t], csp_avg_lb_scaled)
    set_upper_bound.(solver.csp_avg_scaled[:, t], csp_avg_ub_scaled)

    set_lower_bound.(solver.csn_avg_scaled[:, t], csn_avg_lb_scaled)
    set_upper_bound.(solver.csn_avg_scaled[:, t], csn_avg_ub_scaled)
    set_lower_bound.(solver.delta_sei_scaled[:, t], delta_sei_lb_scaled)
    set_upper_bound.(solver.delta_sei_scaled[:, t], delta_sei_ub_scaled)
    set_lower_bound.(solver.cf_scaled[:, t], cf_lb_scaled)
    set_upper_bound.(solver.cf_scaled[:, t], cf_ub_scaled)
  end
  set_lower_bound.(solver.csn_avg_scaled[:, Nt_STATE], csn_avg_lb_end_scaled)
  set_upper_bound.(solver.csn_avg_scaled[:, Nt_STATE], csn_avg_ub_end_scaled)
end


function set_start_point!(solver::OptimalControlSolver, csp_avg0_scaled, csn_avg0_scaled, delta_sei0_scaled, cf0_scaled)
  fix.(solver.csp_avg_scaled[:, 1], csp_avg0_scaled; force=true)
  fix.(solver.csn_avg_scaled[:, 1], csn_avg0_scaled; force=true)
  fix.(solver.delta_sei_scaled[:, 1], delta_sei0_scaled; force=true)
  fix.(solver.cf_scaled[:, 1], cf0_scaled; force=true)
  fix.(solver.eps[:, 1], 0)
end


function reset_solver!(solver::OptimalControlSolver, u0, grid_price, swap_battery_times, swap_battery_states, start_point, cf_min)
  csp_avg0, csn_avg0, delta_sei0, cf0 = u0[:, 1], u0[:, Ncp+1], u0[:, Ncp+Ncn+7], u0[:, Ncp+Ncn+Nsei+5]
  (csp_avg0_scaled, csn_avg0_scaled, delta_sei0_scaled, cf0_scaled,
    csp_avg_lb_scaled, csp_avg_ub_scaled,
    csn_avg_lb_scaled, csn_avg_ub_scaled, csn_avg_lb_end_scaled, csn_avg_ub_end_scaled,
    delta_sei_lb_scaled, delta_sei_ub_scaled, cf_lb_scaled, cf_ub_scaled,
    bat_charge_lb_scaled, bat_charge_ub_scaled) = get_scaled_params(solver, csp_avg0, csn_avg0, delta_sei0, cf0)

  set_parameters!(solver, grid_price, swap_battery_times, swap_battery_states, cf_min)
  set_initial_guess!(solver, swap_battery_times, csp_avg0_scaled, csn_avg0_scaled, delta_sei0_scaled, cf0_scaled, start_point)
  set_bounds!(solver, bat_charge_lb_scaled, bat_charge_ub_scaled, csp_avg_lb_scaled, csp_avg_ub_scaled,
    csn_avg_lb_scaled, csn_avg_ub_scaled, delta_sei_lb_scaled, delta_sei_ub_scaled, cf_lb_scaled, cf_ub_scaled,
    csn_avg_lb_end_scaled, csn_avg_ub_end_scaled)
  set_start_point!(solver, csp_avg0_scaled, csn_avg0_scaled, delta_sei0_scaled, cf0_scaled)
end


function update_warm_start_buffer!(solver::OptimalControlSolver, csp_avg, csn_avg, delta_sei, cf, bat_charge, eps)
  solver.last_csp_avg_scaled = csp_avg
  solver.last_csn_avg_scaled = csn_avg
  solver.last_delta_sei_scaled = delta_sei
  solver.last_cf_scaled = cf
  solver.last_bat_charge_scaled = bat_charge
  solver.last_eps = eps
end


function compute_ocp_solution(solver::OptimalControlSolver, u0, grid_price, swap_battery_times, swap_battery_states, start_point, cf_min)
  start_time = time()
  reset_solver!(solver, u0, grid_price, swap_battery_times, swap_battery_states, start_point, cf_min)
  optimize!(solver.m)
  status = is_solved_and_feasible(solver.m)
  while !status # reset everything to solve again
    reset_solver!(solver, u0, grid_price, swap_battery_times, swap_battery_states, 1, cf_min)
    optimize!(solver.m)
    status = is_solved_and_feasible(solver.m)
  end
  println("ocp suceeded, elapsed time:  ", time() - start_time)

  profit = value(solver.profit)
  penalty = value(solver.penalty)

  bat_charge = get_array_from_dense_axis_array(solver.bat_charge)
  csp_avg = get_array_from_dense_axis_array(solver.csp_avg)
  csn_avg = get_array_from_dense_axis_array(solver.csn_avg)
  delta_sei = get_array_from_dense_axis_array(solver.delta_sei)
  cf = get_array_from_dense_axis_array(solver.cf)
  bat_charge_scaled = get_array_from_dense_axis_array(solver.bat_charge_scaled)
  csp_avg_scaled = get_array_from_dense_axis_array(solver.csp_avg_scaled)
  csn_avg_scaled = get_array_from_dense_axis_array(solver.csn_avg_scaled)
  delta_sei_scaled = get_array_from_dense_axis_array(solver.delta_sei_scaled)
  cf_scaled = get_array_from_dense_axis_array(solver.cf_scaled)
  eps = get_array_from_dense_axis_array(solver.eps)
  update_warm_start_buffer!(solver, csp_avg_scaled, csn_avg_scaled, delta_sei_scaled, cf_scaled, bat_charge_scaled, eps)

  csn_avg_swapped = csn_avg
  cf_swapped = cf
  for t in 1:Nt_STATE
    swap_battery_idx = findall(x -> x == 1, eps[:, t])
    csn_avg_swapped[swap_battery_idx, t] .= swap_battery_states[2]
    cf_swapped[swap_battery_idx, t] .= swap_battery_states[4]
  end
  soc_swapped = csn_avg_swapped ./ csnmax

  # for t in 1:Nt_STATE-1
  #   println("eps:  ", eps[:, t])
  # end
  println("trajectory profit: ", profit)
  println("trajectory penalty:  ", penalty)

  # # plot
  # rows = floor(Int, sqrt(NUM_BATTERIES_IN_STATION))
  # cols = ceil(Int, NUM_BATTERIES_IN_STATION / rows)
  # figure()
  # for k in 1:NUM_BATTERIES_IN_STATION
  #   subplot(rows, cols, k)
  #   PyPlot.plot(soc_swapped[k, :])
  #   title("Bat $k", fontsize=8)
  #   xlabel("Time step")
  #   ylabel("soc")
  #   PyPlot.grid(true)
  # end
  # show(block=false)

  # figure()
  # for k in 1:NUM_BATTERIES_IN_STATION
  #   subplot(rows, cols, k)
  #   PyPlot.plot(cf_swapped[k, :])
  #   title("Bat $k", fontsize=8)
  #   xlabel("Time step")
  #   ylabel("cf")
  #   PyPlot.grid(true)
  # end
  # show(block=false)

  # figure()
  # for k in 1:NUM_BATTERIES_IN_STATION
  #   subplot(rows, cols, k)
  #   PyPlot.plot(bat_charge[k, :])
  #   title("Bat $k", fontsize=8)
  #   xlabel("Time step")
  #   ylabel("charge")
  #   PyPlot.grid(true)
  # end
  # show(block=false)

  # cumulative_profit = zeros(Nt_STATE)
  # for t in 1:Nt_STATE-1
  #   cumulative_profit[t+1] = cumulative_profit[t] - sum(bat_charge[:, t]) * grid_price[t]
  # end
  # figure()
  # PyPlot.plot(cumulative_profit)
  # title("cumulative profit", fontsize=8)
  # xlabel("Time step")
  # ylabel("cumulative profit")
  # PyPlot.grid(true)
  # show()

  # u1 = zero(u0)
  # u1[:, 1] .= csp_avg[:, 2]
  # u1[:, 3] .= csn_avg[:, 2]
  # u1[:, 11] .= delta_sei[:, 2]
  # u1[:, 12] .= cf[:, 2]

  # for k in 1:NUM_BATTERIES_IN_STATION
  #   println("u1_ocp[$k]:  ", u1[k, [1, 3, 11, 12]])
  # end

  return bat_charge, eps
end