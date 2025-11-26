using CSV
using DataFrames
using Plots
using JLD

include("simulator.jl")
include("Kriging.jl")
include("../simplified/setup.jl")
include("../simplified/bss_quasi_static.jl")


NUM_BATTERIES_IN_STATION = 1

function predict(krigin_surrogate, x)
  y_csp_avg = pred(krigin_surrogate[1], x)
  y_csn_avg = pred(krigin_surrogate[2], x)
  y_delta_sei = pred(krigin_surrogate[3], x)
  y_cf = pred(krigin_surrogate[4], x)
  y_pred = [y_csp_avg, y_csn_avg, y_delta_sei, y_cf] .+ x[1:4]
  y_soc = y_pred[2] / csnmax
  pred_success = true
  if y_soc <= soc_min_stop || y_soc >= soc_max_stop
    pred_success = false
  end
  return y_pred, pred_success
end


function simulate(simulator, krigin_surrogate, u0, horizon)
  u0 = transpose(copy(u0))
  x0 = [u0[1], u0[3], u0[11], u0[12], 0.0]

  Y_traj_true = zeros(horizon, 4)
  Y_traj_pred = zeros(horizon, 4)

  h = 1
  while h <= horizon
    bat_charge = rand(Uniform(-P_max, P_max))
    u0 = getinitial(u0, bat_charge)
    x0[5] = bat_charge

    # simulate one step
    u1, _, sim_success = simulate(simulator, u0, 0.0, [bat_charge], nothing, nothing)
    # predict
    x1, pred_success = predict(krigin_surrogate, x0)

    if sim_success && pred_success
      if h % 10 == 0
        println("Step $h")
      end

      Y_traj_true[h, :] = [u1[1], u1[3], u1[11], u1[12]]
      Y_traj_pred[h, :] = x1
      h = h + 1
      u0 = u1
      x0[1:4] = x1
      # x0[3:4] = [u1[11], u1[12]]
    end
  end

  return Y_traj_true, Y_traj_pred
end


horizon = 300
simulator = Simulator(zeros(Nu), 0)

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
u0[Ncp+Ncn+Nsei+5] = 0.0


# load params
data = load("gp_model_params.jld")

θ_csp_avg = data["θ_csp_avg"]
θ_csn_avg = data["θ_csn_avg"]
θ_delta_sei = data["θ_delta_sei"]
θ_cf = data["θ_cf"]
train_indices_csp_avg = data["train_indices_csp_avg"]
train_indices_csn_avg = data["train_indices_csn_avg"]
train_indices_delta_sei = data["train_indices_delta_sei"]
train_indices_cf = data["train_indices_cf"]
state_means = data["state_means"]
state_stds = data["state_stds"]
input_means = data["input_means"]
input_stds = data["input_stds"]


# load data
df = CSV.read("battery_data.csv", DataFrame)
x = Matrix(df[:, [:x1, :x2, :x3, :x4, :u]])
y = Matrix(df[:, [:x_next1, :x_next2, :x_next3, :x_next4]])
total_data_length = size(x, 1)

# Normalization
x_means = hcat(state_means, input_means)
x_stds = hcat(state_stds, input_stds)
x = (x .- x_means) ./ x_stds
y = (y .- state_means) ./ state_stds - x[:, 1:4]
y_csp_avg = y[:, 1]
y_csn_avg = y[:, 2]
y_delta_sei = y[:, 3]
y_cf = y[:, 4]


# build manual GPR
x_train_1 = [collect(row) for row in eachrow(x[train_indices_csp_avg, :])]
y_train_1 = y_csp_avg[train_indices_csp_avg]
x_train_2 = [collect(row) for row in eachrow(x[train_indices_csn_avg, :])]
y_train_2 = y_csn_avg[train_indices_csn_avg]
x_train_3 = [collect(row) for row in eachrow(x[train_indices_delta_sei, :])]
y_train_3 = y_delta_sei[train_indices_delta_sei]
x_train_4 = [collect(row) for row in eachrow(x[train_indices_cf, :])]
y_train_4 = y_cf[train_indices_cf]
krigin_surrogate = [
  Kriging(x_train_1, y_train_1, θ_csp_avg, vec(x_means), vec(x_stds), state_stds[1]),
  Kriging(x_train_2, y_train_2, θ_csn_avg, vec(x_means), vec(x_stds), state_stds[2]),
  Kriging(x_train_3, y_train_3, θ_delta_sei, vec(x_means), vec(x_stds), state_stds[3]),
  Kriging(x_train_4, y_train_4, θ_cf, vec(x_means), vec(x_stds), state_stds[4])
]

Y_traj_true, Y_traj_pred = simulate(simulator, krigin_surrogate, u0, horizon)


labels = ["csp_avg", "csn_avg", "delta_sei", "cf"]
p = Plots.plot(layout=(2, 2), size=(1000, 800))
for i in 1:4
  plot!(
    p[i],
    1:horizon, Y_traj_true[:, i],
    label="True $(labels[i])",
    lw=2,
    xlabel="Sample Index",
    ylabel="Value",
    title="Prediction vs True for $(labels[i])"
  )
  plot!(
    p[i],
    Y_traj_pred[:, i],
    label="Predicted $(labels[i])",
    lw=2,
    ls=:dash
  )
end
display(p)
sleep(100)