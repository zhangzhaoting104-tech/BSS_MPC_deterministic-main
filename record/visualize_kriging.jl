

using Statistics
using CSV
using DataFrames
using Plots, Measures
using JLD
using LaTeXStrings

include("../GP_model/Kriging.jl")


data = load("../GP_Model/gp_model_params.jld")

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
df = CSV.read("../GP_Model/battery_data.csv", DataFrame)
state_input = Matrix(df[:, [:x_next1, :x_next2, :x_next3, :x_next4, :u]])
last_state = Matrix(df[:, [:x1, :x2, :x3, :x4]])
x = copy(state_input)
y = copy(last_state)
total_data_length = size(x, 1)

# Normalization
state_max = copy(maximum(vcat(reshape(y[1, :], 1, 4), x[:, 1:4]), dims=1))
state_min = copy(minimum(vcat(reshape(y[1, :], 1, 4), x[:, 1:4]), dims=1))
input_max = copy(maximum(x[:, 5], dims=1))
input_min = copy(minimum(x[:, 5], dims=1))

# Standardization
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
  Kriging(x_train_1, y_train_1, θ_csp_avg, vec(x_means), vec(x_stds), x_stds[1]),
  Kriging(x_train_2, y_train_2, θ_csn_avg, vec(x_means), vec(x_stds), x_stds[2]),
  Kriging(x_train_3, y_train_3, θ_delta_sei, vec(x_means), vec(x_stds), x_stds[3]),
  Kriging(x_train_4, y_train_4, θ_cf, vec(x_means), vec(x_stds), x_stds[4])
]


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
err = zero(y[50:end, :])
for i in 1:4
    x_test = [collect(row) for row in eachrow(x[50:end, :])]
    y_pred = zero(y_cf[50:end, :])
    for j in 1:total_data_length-50+1
        y_pred[j] = pred(krigin_surrogate[i], x_test[j] .* vec(x_stds) .+ vec(x_means))
    end
    err[:, i] = abs.((y_pred .+ state_input[50:end, i]) .- last_state[50:end, i]) ./ last_state[50:end, i]
end

labels = [L"\textbf{C^{avg}_{p}}", L"\textbf{C^{avg}_{n}}", L"\textbf{δ_{\mathrm{SEI}}}", L"\textbf{c_f}"]
p_err = plot(layout=(2, 2), size=(900, 800), margin=5mm)
for i in 1:4
    plot!(
        p_err[i],
        1:total_data_length-50+1, err[:, i],
        lw=2,
        legend=false,
        xlabel=L"\textbf{Sample~Index}",
        ylabel=L"\textbf{Relative~Error~for~}" * labels[i],
        titlefont=font(10),
        guidefont=font(10),
        tickfont=font(10),
        gridalpha=0.7, 
        gridcolor=:black, 
    )
end
gui()
sleep(500)
