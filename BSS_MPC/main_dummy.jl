using Random
using StatsBase
using Distributions
using Plots

include("../utils/bss_initial_state.jl")
include("../utils/bss_quasi_static.jl")
include("../utils/bss_simulator.jl")
include("../utils/bss_manager.jl")
include("../utils/bss_data_record.jl")


quasi_static_solver = get_quasi_static_solver()
simulator = Simulator(zeros(Nu), 0)
manager = create_bss_manager(NUM_BATTERIES_IN_STATION, NUM_BATTERIES_IN_AREA)

global u0 = get_initial_states(manager)
global profit = []
global cf_avg = []
global cf_std = []
global num_soc_violate_list = []
total_iters = 30 * 6 * 24
############################# mpc #############################
loop_start_time = time()
for hour in 1:total_iters
    start_time = time()
    println("############################# step $hour #############################")

    # get swap info
    swap_battery_times, swap_battery_states = get_approx_swap_info(manager, hour)

    # solve ocp once
    grid_price = Prices[hour:hour+Nt_INPUT-1]

    # solve for bat_charge, u0 and eps
    eps = select_batteries_with_max_cf(manager, swap_battery_times[2])
    bat_charge = zeros(NUM_BATTERIES_IN_STATION)
    for k in 1:NUM_BATTERIES_IN_STATION
        u0[k, :], bat_charge[k] = compute_quasi_static(quasi_static_solver, u0[k, :], soc_max)
    end

    # simulate
    swap_battery_idx = findall(x -> x == 1, eps)
    swap_battery_state = get_swap_state(manager, swap_battery_times[2])
    global u0, profit0, _, delta_sei_list, cf_list, num_soc_violate = simulate(simulator, u0, grid_price[1], bat_charge, swap_battery_idx, swap_battery_state; log=true)

    # update all battery
    update_bss_manager!(manager, swap_battery_idx, delta_sei_list, cf_list)

    push!(profit, profit0)
    push!(cf_avg, mean(manager.cf_list))
    push!(cf_std, std(manager.cf_list))
    push!(num_soc_violate_list, num_soc_violate)
    println("total profit: ", sum(profit))
    println("total elapsed time:  ", time() - start_time)
end
println("Finished! $total_iters iterations elapsed time:  ", time() - loop_start_time)
cumulative_profit = cumsum(profit)
record("../record/dummy.csv", cf_avg, cf_std, cumulative_profit, num_soc_violate_list)


plt = Plots.plot(
    Plots.plot(cumulative_profit,
        xlabel="Time step",
        ylabel="Cumulative Profit",
        title="Cumulative Profit over Time",
        lw=2,
        legend=false
    ),
    Plots.plot(cf_avg,
        xlabel="Time step",
        ylabel="cf value",
        title="avg cf over Time",
        lw=2,
        legend=false
    ),
    Plots.plot(cf_std,
        xlabel="Time step",
        ylabel="cf value",
        title="cf std over Time",
        lw=2,
        legend=false
    ),
    guidefontsize=7,
    titlefontsize=6,
    tickfontsize=5,
)

display(plt)
sleep(100000000)