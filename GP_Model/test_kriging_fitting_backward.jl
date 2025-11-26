using QuasiMonteCarlo
using Distributions
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using Plots
using Optim
using JLD

include("Kriging.jl")


df = CSV.read("battery_data.csv", DataFrame)

state_input = Matrix(df[:, [:x_next1, :x_next2, :x_next3, :x_next4, :u]])
last_state = Matrix(df[:, [:x1, :x2, :x3, :x4]])
total_data_length = size(state_input, 1)
x = copy(state_input)
y = copy(last_state)


# Normalization
state_means = Matrix(mean(vcat(x[:, 1:4], Matrix(y[end, :]')), dims=1))
state_stds = Matrix(std(vcat(x[:, 1:4], Matrix(y[end, :]')), dims=1))
input_means = mean(x[:, 5], dims=1)
input_stds = std(x[:, 5], dims=1)
x_means = hcat(state_means, input_means)
x_stds = hcat(state_stds, input_stds)
x[:, 1:4] = (x[:, 1:4] .- state_means) ./ state_stds
x[:, 5] = (x[:, 5] .- input_means) ./ input_stds
y = (y .- state_means) ./ state_stds - x[:, 1:4]
y_csp_avg = y[:, 1]
y_csn_avg = y[:, 2]
y_delta_sei = y[:, 3]
y_cf = y[:, 4]


max_iter = 100
################################# csp_avg #################################
tol = 5e-2
global train_indices_csp_avg = []
global θ_csp_avg = []
global num_obs_per_iter_csp_avg = 10
global num_obs_added = 1
samples = QuasiMonteCarlo.sample(num_obs_per_iter_csp_avg, 0, 1, SobolSample())
global train_indices_csp_avg = unique(vec(floor.(Int, total_data_length * samples)) .+ 1)
global test_indices_csp_avg = setdiff(1:total_data_length, train_indices_csp_avg)
i = 0
while i < max_iter
    start_time = time()

    # train
    x_train = x[train_indices_csp_avg, :]
    x_train = [collect(row) for row in eachrow(x_train)]
    y_train = y_csp_avg[train_indices_csp_avg]
    global θ_csp_avg = train_kriging(x_train, y_train)
    krigin_surrogate = Kriging(x_train, y_train, θ_csp_avg, vec(x_means), vec(x_stds), x_stds[1])

    # test
    x_test = x[test_indices_csp_avg, :]
    x_test = [collect(row) for row in eachrow(x_test)]
    y_test = y_csp_avg[test_indices_csp_avg, :]
    y_pred = zero(y_test)
    for i in 1:length(test_indices_csp_avg)
        y_pred[i] = krigin_surrogate(x_test[i])
    end
    max_re = maximum(abs.(y_test .- y_pred))
    println("num_obs = $(length(train_indices_csp_avg)), max_re = ", max_re)
    println("fitting elapsed time:  ", time() - start_time)

    if max_re < tol
        break
    else
        re = abs.(y_test .- y_pred)
        selected_in_candidate = sortperm(re, rev=true, dims=1)[1:min(num_obs_added, length(re))]
        new_selected_indices = test_indices_csp_avg[selected_in_candidate]

        global train_indices_csp_avg = vcat(train_indices_csp_avg, new_selected_indices)
        global test_indices_csp_avg = setdiff(test_indices_csp_avg, new_selected_indices)
    end
end


################################# csn_avg #################################
tol = 5e-2
global train_indices_csn_avg = []
global θ_csn_avg = []
global num_obs_per_iter_csn_avg = 10
global num_obs_added = 1
samples = QuasiMonteCarlo.sample(num_obs_per_iter_csn_avg, 0, 1, SobolSample())
global train_indices_csn_avg = unique(vec(floor.(Int, total_data_length * samples)) .+ 1)
global test_indices_csn_avg = setdiff(1:total_data_length, train_indices_csn_avg)
i = 0
while i < max_iter
    start_time = time()

    # train
    x_train = x[train_indices_csn_avg, :]
    x_train = [collect(row) for row in eachrow(x_train)]
    y_train = y_csn_avg[train_indices_csn_avg]
    global θ_csn_avg = train_kriging(x_train, y_train)
    krigin_surrogate = Kriging(x_train, y_train, θ_csn_avg, vec(x_means), vec(x_stds), x_stds[2])

    # test
    x_test = x[test_indices_csn_avg, :]
    x_test = [collect(row) for row in eachrow(x_test)]
    y_test = y_csn_avg[test_indices_csn_avg, :]
    y_pred = zero(y_test)
    for i in 1:length(test_indices_csn_avg)
        y_pred[i] = krigin_surrogate(x_test[i])
    end
    max_re = maximum(abs.(y_test .- y_pred))
    println("num_obs = $(length(train_indices_csn_avg)), max_re = ", max_re)
    println("fitting elapsed time:  ", time() - start_time)

    if max_re < tol
        break
    else
        re = abs.(y_test .- y_pred)
        selected_in_candidate = sortperm(re, rev=true, dims=1)[1:min(num_obs_added, length(re))]
        new_selected_indices = test_indices_csn_avg[selected_in_candidate]

        global train_indices_csn_avg = vcat(train_indices_csn_avg, new_selected_indices)
        global test_indices_csn_avg = setdiff(test_indices_csn_avg, new_selected_indices)
    end
end


################################# delta_sei #################################
tol = 5e-4
global train_indices_delta_sei = []
global θ_delta_sei = []
global num_obs_per_iter_delta_sei = 3
global num_obs_added = 1
samples = QuasiMonteCarlo.sample(num_obs_per_iter_delta_sei, 0, 1, SobolSample())
global train_indices_delta_sei = unique(vec(floor.(Int, total_data_length * samples)) .+ 1)
global test_indices_delta_sei = setdiff(1:total_data_length, train_indices_delta_sei)
i = 0
while i < max_iter
    start_time = time()

    # train
    x_train = x[train_indices_delta_sei, :]
    x_train = [collect(row) for row in eachrow(x_train)]
    y_train = y_delta_sei[train_indices_delta_sei]
    global θ_delta_sei = train_kriging(x_train, y_train)
    krigin_surrogate = Kriging(x_train, y_train, θ_delta_sei, vec(x_means), vec(x_stds), x_stds[3])

    # test
    x_test = x[test_indices_delta_sei, :]
    x_test = [collect(row) for row in eachrow(x_test)]
    y_test = y_delta_sei[test_indices_delta_sei, :]
    y_pred = zero(y_test)
    for i in 1:length(test_indices_delta_sei)
        y_pred[i] = krigin_surrogate(x_test[i])
    end
    max_re = maximum(abs.(y_test .- y_pred))
    println("num_obs = $(length(train_indices_delta_sei)), max_re = ", max_re)
    println("fitting elapsed time:  ", time() - start_time)

    if max_re < tol
        break
    else
        re = abs.(y_test .- y_pred)
        selected_in_candidate = sortperm(re, rev=true, dims=1)[1:min(num_obs_added, length(re))]
        new_selected_indices = test_indices_delta_sei[selected_in_candidate]

        global train_indices_delta_sei = vcat(train_indices_delta_sei, new_selected_indices)
        global test_indices_delta_sei = setdiff(test_indices_delta_sei, new_selected_indices)
    end
end


################################# cf #################################
tol = 5e-4
global train_indices_cf = []
global θ_cf = []
global num_obs_per_iter_cf = 3
global num_obs_added = 1
samples = QuasiMonteCarlo.sample(num_obs_per_iter_cf, 0, 1, SobolSample())
global train_indices_cf = unique(vec(floor.(Int, total_data_length * samples)) .+ 1)
global test_indices_cf = setdiff(1:total_data_length, train_indices_cf)
i = 0
while i < max_iter
    start_time = time()

    # train
    x_train = x[train_indices_cf, :]
    x_train = [collect(row) for row in eachrow(x_train)]
    y_train = y_cf[train_indices_cf]
    global θ_cf = train_kriging(x_train, y_train)
    krigin_surrogate = Kriging(x_train, y_train, θ_cf, vec(x_means), vec(x_stds), x_stds[4])

    # test
    x_test = x[test_indices_cf, :]
    x_test = [collect(row) for row in eachrow(x_test)]
    y_test = y_cf[test_indices_cf, :]
    y_pred = zero(y_test)
    for i in 1:length(test_indices_cf)
        y_pred[i] = krigin_surrogate(x_test[i])
    end
    max_re = maximum(abs.(y_test .- y_pred))
    println("num_obs = $(length(train_indices_cf)), max_re = ", max_re)
    println("fitting elapsed time:  ", time() - start_time)

    if max_re < tol
        break
    else
        re = abs.(y_test .- y_pred)
        selected_in_candidate = sortperm(re, rev=true, dims=1)[1:min(num_obs_added, length(re))]
        new_selected_indices = test_indices_cf[selected_in_candidate]

        global train_indices_cf = vcat(train_indices_cf, new_selected_indices)
        global test_indices_cf = setdiff(test_indices_cf, new_selected_indices)
    end
end


save("gp_model_params.jld", Dict(
    "θ_csp_avg" => θ_csp_avg,
    "θ_csn_avg" => θ_csn_avg,
    "θ_delta_sei" => θ_delta_sei,
    "θ_cf" => θ_cf,
    "train_indices_csp_avg" => train_indices_csp_avg,
    "train_indices_csn_avg" => train_indices_csn_avg,
    "train_indices_delta_sei" => train_indices_delta_sei,
    "train_indices_cf" => train_indices_cf,
    "state_means" => state_means,
    "state_stds" => state_stds,
    "input_means" => input_means,
    "input_stds" => input_stds
))


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

labels = ["csp_avg", "csn_avg", "delta_sei", "cf"]
p_err = plot(layout=(2, 2), size=(1000, 800))
for i in 1:4
    plot!(
        p_err[i],
        1:total_data_length-50+1, err[:, i],
        label="Relative Error for $(labels[i])",
        lw=2,
        xlabel="Sample Index",
        ylabel="Relative Error",
        title="Relative Error for $(labels[i])"
    )
end
gui()
sleep(500)