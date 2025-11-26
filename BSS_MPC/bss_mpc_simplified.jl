using DifferentialEquations
using PyPlot
using Interpolations
using Distributions
using JuMP
using GAMS
using JLD

include("../utils/setup.jl")
include("../utils/utils.jl")


function OptimalControl(u0, grid_price, swap_battery_times, swap_battery_states, start_point, last_bat_charge, last_eps)

  # initial battery internal states
  soc0 = u0[:, 3] ./ csnmax
  capacity_remain0 = 1 .- u0[:, Ncp+Ncn+Nsei+5] ./ Qmax
  swap_battery_soc = swap_battery_states[:, 2] ./ csnmax

  m = Model(GAMS.Optimizer)
  set_optimizer_attribute(m, GAMS.ModelType(), "MINLP")
  set_optimizer_attribute(m, "MINLP", "dicopt")
  set_optimizer_attribute(m, "NLP", "ipopth")
  set_optimizer_attribute(m, "solver", "ipopth")
  set_optimizer_attribute(m, "linear_solver", "ma86")
  set_optimizer_attribute(m, "MIP", "cplex")
  set_optimizer_attribute(m, "HoldFixed", 1)
  set_optimizer_attribute(m, "threads", 14)
  # set_optimizer_attribute(m, "resLim", 120)
  set_optimizer_attribute(m, MOI.Silent(), true)
  set_optimizer_attribute(m, GAMS.WorkDir(), "D:\\GAMS\\GAMSWorkspace\\tmp\\simplified")


  ############
  ## inputs ##
  ############
  @variable(m, bat_charge[k in 1:NUM_BATTERIES_IN_STATION, t in 2:Nt_INPUT+1])
  set_lower_bound.(bat_charge, -P_nominal * maxC)
  set_upper_bound.(bat_charge, P_nominal * maxC)
  set_start_value.(bat_charge, 0.0)


  ############
  ## states ##
  ############
  @variable(m, soc[k in 1:NUM_BATTERIES_IN_STATION, t in 1:Nt_STATE], start = 0.5)
  fix.(soc[:, 1], soc0; force=true)
  for t in 2:Nt_STATE-1
    for k in 1:NUM_BATTERIES_IN_STATION
      set_lower_bound(soc[k, t], capacity_remain0[k] * soc_min_stop)
      set_upper_bound(soc[k, t], capacity_remain0[k] * soc_max_stop)
    end
  end
  set_lower_bound.(soc[:, end], capacity_remain0 .* soc_min)
  set_upper_bound.(soc[:, end], capacity_remain0 .* soc_max)


  #########################
  ## selective variables ##
  #########################
  @variable(m, eps[k in 1:NUM_BATTERIES_IN_STATION, t in 1:Nt_STATE], Bin)
  set_start_value.(eps, get_eps_initial_guess(NUM_BATTERIES_IN_STATION, Nt_STATE, swap_battery_times, start_point, last_eps))
  fix.(eps[:, 1], 0)


  ################ constraints ################
  for t in 1:Nt_STATE-1
    for k in 1:NUM_BATTERIES_IN_STATION
      @constraint(m, soc[k, t] >= (soc_max) * eps[k, t])
      @constraint(m, 0 == soc[k, t+1] - bat_charge[k, t+1] / state_cycle / P_nominal * DT_STATE .- jump_state(soc[k, t], swap_battery_soc[t], eps[k, t]))
    end
    @constraint(m, swap_battery_times[t+1] - 0.9 <= sum(eps[:, t+1]) <= swap_battery_times[t+1] + 0.9)
  end


  ###############
  ## Objective ##
  ###############
  ############## profit ##############
  profit = -sum(sum(bat_charge[k, :] for k in 1:NUM_BATTERIES_IN_STATION) .* grid_price)

  ############## penalty ##############
  penalty_coeffi = 1e-3
  penalty = -sum(sum(bat_charge .^ 2)) * penalty_coeffi

  ############## obj ##############
  obj = profit + penalty
  @objective(m, Max, obj)


  ############## solve and get result ##############
  start_time = time()
  optimize!(m)
  status = is_solved_and_feasible(m)
  if status
    println("ocp suceeded, elapsed time:  ", time() - start_time)
  else # return last solution if solver failed
    println("ocp failed, elapsed time:  ", time() - start_time)
    bat_charge = hcat(last_bat_charge[:, 2:end-1], zeros(NUM_BATTERIES_IN_STATION))
    eps = get_eps_initial_guess(NUM_BATTERIES_IN_STATION, Nt_STATE, swap_battery_times, start_point, last_eps)
    return bat_charge, eps
  end

  profit = value(profit)
  penalty = value(penalty)

  bat_charge = get_array_from_dense_axis_array(bat_charge)
  soc = get_array_from_dense_axis_array(soc)
  eps = get_array_from_dense_axis_array(eps)

  # for t in 1:Nt_STATE-1
  #   println("eps:  ", eps[:, t])
  # end
  println("trajectory profit: ", profit)
  println("trajectory penalty: ", penalty)

  # # plot
  # rows = floor(Int, sqrt(NUM_BATTERIES_IN_STATION))
  # cols = ceil(Int, NUM_BATTERIES_IN_STATION / rows)
  # figure()
  # for k in 1:NUM_BATTERIES_IN_STATION
  #   subplot(rows, cols, k)
  #   PyPlot.plot(soc[k, :])
  #   title("Bat $k", fontsize=8)
  #   xlabel("Time step")
  #   ylabel("soc")
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

  return bat_charge, eps
end
