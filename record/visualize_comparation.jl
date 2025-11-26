using CSV
using DataFrames
using Plots, Measures


df_dummy = CSV.read("./dummy.csv", DataFrame)
df_simplified = CSV.read("./simplified.csv", DataFrame)
df_original_low_cf = CSV.read("./original_low_cf.csv", DataFrame)
df_original_low_cost = CSV.read("./original_low_cost.csv", DataFrame)

cf_dummy = Matrix(df_dummy[:, [:cf_avg, :cf_std]])
cumulative_profit_dummy = Matrix(df_dummy[:, [:cumulative_profit]])
cf_simplified = Matrix(df_simplified[:, [:cf_avg, :cf_std]])
cumulative_profit_simplified = Matrix(df_simplified[:, [:cumulative_profit]])
cf_original_low_cf = Matrix(df_original_low_cf[:, [:cf_avg, :cf_std]])
cumulative_profit_original_low_cf = Matrix(df_original_low_cf[:, [:cumulative_profit]])
cf_original_low_cost = Matrix(df_original_low_cost[:, [:cf_avg, :cf_std]])
cumulative_profit_original_low_cost = Matrix(df_original_low_cost[:, [:cumulative_profit]])


p1 = plot(1:length(cf_dummy[:, 1]), cf_dummy[:, 1], label="Rule Based", linewidth=1, title="cf avg", xlabel="Hour Index", ylabel="cf avg", left_margin=10mm, bottom_margin=10mm, margin=5mm)
plot!(1:length(cf_simplified[:, 1]), cf_simplified[:, 1], label="Low Fidelity", linewidth=2)
plot!(1:length(cf_original_low_cf[:, 1]), cf_original_low_cf[:, 1], label="BSS-MPC (low-cf)", linewidth=1)
plot!(1:length(cf_original_low_cost[:, 1]), cf_original_low_cost[:, 1], label="BSS-MPC (low-cost)", linewidth=1)

p2 = plot(1:length(cf_dummy[:, 2]), cf_dummy[:, 2], label="Rule Based", linewidth=1, title="cf std", xlabel="Hour Index", ylabel="cf std", margin=5mm)
plot!(1:length(cf_simplified[:, 2]), cf_simplified[:, 2], label="Low Fidelity", linewidth=2)
plot!(1:length(cf_original_low_cf[:, 2]), cf_original_low_cf[:, 2], label="BSS-MPC (low-cf)", linewidth=1)
plot!(1:length(cf_original_low_cost[:, 2]), cf_original_low_cost[:, 2], label="BSS-MPC (low-cost)", linewidth=1)

p3 = plot(1:length(cumulative_profit_dummy), cumulative_profit_dummy, label="Rule Based", linewidth=1, title="cumulative cost", xlabel="Hour Index", ylabel="cumulative profit", margin=5mm)
plot!(1:length(cumulative_profit_simplified), cumulative_profit_simplified, label="Low Fidelity", linewidth=2)
plot!(1:length(cumulative_profit_original_low_cf), cumulative_profit_original_low_cf, label="BSS-MPC (low-cf)", linewidth=1)
plot!(1:length(cumulative_profit_original_low_cost), cumulative_profit_original_low_cost, label="BSS-MPC (low-cost)", linewidth=1)


plot(p1, p2, p3,
    layout=(1, 3),
    size=(1400, 400),
    guidefontsize=8,
    titlefontsize=10,
    tickfontsize=8,
    legend=:best,
    grid=true,
    minorgrid=false,
    minorticks=false,
    gridalpha=0.7,
    gridcolor="gray"
)
gui()
sleep(100)