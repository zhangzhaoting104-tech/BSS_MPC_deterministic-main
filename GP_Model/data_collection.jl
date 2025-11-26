
using CSV, DataFrames, Random
using Distributions

include("../utils/setup.jl")
include("../utils/bss_initial_state.jl")
include("simulator.jl")


NUM_BATTERIES_IN_STATION = 1

function collect_dataset(simulator::Simulator, initial_state_solver, u0::Vector{Float64};
  Nmax::Int=10000, filename::String="./battery_data.csv")

  data = DataFrame(
    csp_avg=Float64[],
    csn_avg=Float64[],
    delta_sei=Float64[],
    cf=Float64[],
    bat_charge=Float64[],
    csp_avg_next=Float64[],
    csn_avg_next=Float64[],
    delta_sei_next=Float64[],
    cf_next=Float64[]
  )
  u_cur = transpose(copy(u0))

  i = 1
  while i < Nmax
    # sampling control input
    bat_charge = rand(truncated(Normal(0, P_max), -P_nominal * maxC, P_nominal * maxC))
    u_cur = compute_initial_state(initial_state_solver, u_cur, bat_charge)

    # simulate one step
    u_next, _, success = simulate(simulator, u_cur, 0.0, [bat_charge], nothing, nothing)

    if success # soc should within 0.1~0.9

      # collect until retired
      cf = u_next[end]
      fade = cf / Qmax
      capacity_remain = 1 - fade
      if capacity_remain < soc_retire
        println("Battery retired at step $i, capacity_remain = $capacity_remain")
        break
      end

      # store the data
      row = vcat(u_cur[1], u_cur[3], u_cur[11], u_cur[12], bat_charge, u_next[1], u_next[3], u_next[11], u_next[12])
      push!(data, collect(row'))

      # update states
      u_cur = copy(u_next)

      if i % 100 == 0
        println("Step $i, capacity_remain = $capacity_remain")
      end
      i = i + 1
    end
  end

  Nx = 4
  names = ["x$i" for i in 1:Nx]
  push!(names, "u")
  append!(names, ["x_next$i" for i in 1:Nx])

  rename!(data, names)
  CSV.write(filename, data)
  println("✅ Data written to $filename")
end


simulator = Simulator(zeros(Nu), 0)
initial_state_solver = get_initial_state_solver()

# assumed initial states
dist = truncated(Normal(0.5, 0.1), 0.2, 0.8)
soc0 = rand(dist)
csp_avg0 = cspmax * 1 - csnmax * soc0 * lnn * en / lp / ep
csn_avg0 = csnmax * soc0
delta_sei0 = 1e-10

u0 = zeros(Nu)
u0[1:Ncp] .= csp_avg0
u0[(Ncp+1):(Ncp+Ncn)] .= csn_avg0
u0[Ncp+Ncn+4+3] = delta_sei0
u0[Ncp+Ncn+Nsei+5] = 0
collect_dataset(simulator, initial_state_solver, u0)
