using AxisArrays
using JuMP
using DelimitedFiles
using CSV
using XLSX
using DataFrames
using JLD


function load_hourly_data(Demand_File)
  XLSX.openxlsx(Demand_File) do xf
    sheet = xf["Sheet1"]
    raw_data = XLSX.getdata(sheet)
    df = DataFrame(raw_data[3:end, :], Symbol.(raw_data[2, :]); makeunique=true)

    hourly_counts = df[:, 5:28]
    non_empty_rows = [!all(ismissing, row) for row in eachrow(hourly_counts)]
    hourly_counts_clean = hourly_counts[non_empty_rows, :]
    swap_counts_data = vec(permutedims(Matrix(hourly_counts_clean)))

    return swap_counts_data
  end
end


function get_array_from_dense_axis_array(dense_axis_array)
  return Array(value.(dense_axis_array))
end


function jump_state(prev, new, eps)
  return eps .* new .+ (1 .- eps) .* prev
end


function add_Big_M_constraints(m, x_next, x_prev, x_new, eps, M)
  @constraint(m, x_next <= x_prev + M * eps)
  @constraint(m, x_next >= x_prev - M * eps)
  @constraint(m, x_next <= x_new + M * (1 - eps))
  @constraint(m, x_next >= x_new - M * (1 - eps))
end


function get_eps_initial_guess(NUM_BATTERIES_IN_STATION, Nt_STATE, swap_battery_times, start_point, last_eps)
  eps = zeros(NUM_BATTERIES_IN_STATION, Nt_STATE)
  if start_point == 1
    for t in 2:Nt_STATE
      num_swap = swap_battery_times[t]
      battery_indices = randperm(NUM_BATTERIES_IN_STATION)[1:num_swap]
      eps[battery_indices, t] .= 1
    end
  else
    num_swap = swap_battery_times[end]
    battery_indices = randperm(NUM_BATTERIES_IN_STATION)[1:num_swap]
    new_eps = zeros(NUM_BATTERIES_IN_STATION, 1)
    new_eps[battery_indices] .= 1
    eps = hcat(last_eps[:, 2:end], new_eps)
  end
  return eps
end


function get_state_initial_guess(start_point, last_state; state0=[])
  if start_point == 1
    return state0
  else
    return hcat(last_state[:, 2:end], last_state[:, end])
  end
end


function get_bat_charge_initial_guess(NUM_BATTERIES_IN_STATION, Nt_INPUT, start_point, last_bat_charge)
  if start_point == 1
    return zeros(NUM_BATTERIES_IN_STATION, Nt_INPUT)
  else
    return hcat(last_bat_charge[:, 2:end], zeros(NUM_BATTERIES_IN_STATION, 1))
  end
end
