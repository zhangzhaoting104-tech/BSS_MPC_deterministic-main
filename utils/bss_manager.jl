

mutable struct BssManager
    bat_in_station_ids::Vector{Int}
    bat_in_community_ids::Vector{Int}
    delta_sei_list::Vector{Float64}
    cf_list::Vector{Float64}
end


function create_bss_manager(num_bat_in_station, num_bat_in_area)
    bat_in_station_ids = collect(1:num_bat_in_station)
    bat_in_community_ids = collect(1+num_bat_in_station:num_bat_in_area)
    delta_sei_list = fill(1e-10, num_bat_in_area)
    cf_list = fill(1e-5, num_bat_in_area)
    return BssManager(bat_in_station_ids, bat_in_community_ids, delta_sei_list, cf_list)
end


# update battery capacity and ids
function update_bss_manager!(manager::BssManager, swap_battery_from_station_ids, delta_sei_in_station, cf_in_station)
    manager.delta_sei_list[manager.bat_in_station_ids] .= delta_sei_in_station
    manager.cf_list[manager.bat_in_station_ids] .= cf_in_station
    println("max cf:  ", maximum(manager.cf_list), " min cf:  ", minimum(manager.cf_list), " mean cf:  ", mean(manager.cf_list))

    num_swap = length(swap_battery_from_station_ids)
    swap_battery_from_community_ids = manager.bat_in_community_ids[1:num_swap]
    manager.bat_in_community_ids = manager.bat_in_community_ids[num_swap+1:end]
    manager.bat_in_community_ids = vcat(manager.bat_in_community_ids, manager.bat_in_station_ids[swap_battery_from_station_ids])
    manager.bat_in_station_ids[swap_battery_from_station_ids] .= swap_battery_from_community_ids
end


# get initial states for mpc loop
function get_initial_states(manager::BssManager)
    soc0 = 0.7

    csp_avg0 = cspmax * 1 .- csnmax .* soc0 .* lnn .* en ./ lp ./ ep
    csn_avg0 = csnmax .* soc0
    delta_sei0 = manager.delta_sei_list[manager.bat_in_station_ids]
    cf0 = manager.cf_list[manager.bat_in_station_ids]

    u0 = zeros(NUM_BATTERIES_IN_STATION, Nu)
    u0[:, 1:Ncp] .= csp_avg0
    u0[:, (Ncp+1):(Ncp+Ncn)] .= csn_avg0
    u0[:, Ncp+Ncn+4+3] .= delta_sei0
    u0[:, Ncp+Ncn+Nsei+5] .= cf0
    return u0
end


# get swap_battery_times and swap_battery_states approximation for mpc
function get_approx_swap_info(manager::BssManager, start)
    swap_battery_times = [0]
    for h in 1+start:start+horizon
        swap_num = swap_counts_data[h]
        append!(swap_battery_times, swap_num)
    end
    println("swap time:  ", swap_battery_times)

    num_horizons = length(swap_battery_times)
    swap_battery_states = zeros(num_horizons, 4)
    start_id = 1
    for t in 2:num_horizons
        if swap_battery_times[t] > 0
            swap_battery_state = get_swap_state(manager, swap_battery_times[t], start_id=start_id)
            swap_battery_states[t, :] .= vec(mean(swap_battery_state, dims=1))
        else
            swap_battery_states[t, :] .= 0
        end
        start_id = start_id + swap_battery_times[t-1]
    end
    return swap_battery_times, swap_battery_states
end


# get swap_battery_state for simulation
function get_swap_state(manager::BssManager, swap_battery_time; start_id=1)
    if swap_battery_time > 0
        swap_battery_state = zeros(swap_battery_time, 4)
        soc_swap = 0.2
        csp_avg_swap = cspmax * 1 - csnmax * soc_swap * lnn * en / lp / ep
        csn_avg_swap = csnmax * soc_swap
        swap_battery_from_community_ids = manager.bat_in_community_ids[start_id:start_id+swap_battery_time-1]
        delta_sei_swap = manager.delta_sei_list[swap_battery_from_community_ids]
        cf_swap = manager.cf_list[swap_battery_from_community_ids]
        swap_battery_state[:, 1] .= csp_avg_swap
        swap_battery_state[:, 2] .= csn_avg_swap
        swap_battery_state[:, 3] .= delta_sei_swap
        swap_battery_state[:, 4] .= cf_swap
        return swap_battery_state
    else
        return nothing
    end
end


# select batteries with max cf
function select_batteries_with_max_cf(manager::BssManager, swap_battery_time)
    cf_list = manager.cf_list[manager.bat_in_station_ids]
    top_indices = sortperm(cf_list, rev=true)[1:swap_battery_time]
    eps = zeros(NUM_BATTERIES_IN_STATION)
    eps[top_indices] .= 1
    return eps
end

# get minimum cf
function get_minimum_cf(swap_battery_times, cf_current)
    cf_list = vcat(swap_battery_times, cf_current)
    return minimum(cf_list)
end
