using CSV
using DataFrames
using Plots


df_dummy = CSV.read("./dummy.csv", DataFrame)
df_simplified = CSV.read("./simplified.csv", DataFrame)
df_original = CSV.read("./original_low_cf.csv", DataFrame)

soc_num = vec(Matrix(df_dummy[:, [:num_swap]]))
soc_violate_dummy = vec(Matrix(df_dummy[:, [:num_soc_violate]]))
soc_violate_simplified = vec(Matrix(df_simplified[:, [:num_soc_violate]]))
soc_violate_original = vec(Matrix(df_original[:, [:num_soc_violate]]))

x = 1:24
bar(
    x, soc_num[1:24], labels="Swap Demand",
    xlabel="Hour of the Day", ylabel="Number of Violations", title="SOC Constraint Violations per Hour", legend=:topleft, lw=1, bar_width=0.7,
    guidefontsize=8, titlefontsize=8, tickfontsize=8
)
bar!(x, soc_violate_dummy[1:24], labels="Rule Based")
bar!(x, soc_violate_simplified[1:24], labels="Low Fidelity")
bar!(x, soc_violate_original[1:24], labels="BSS-MPC")


gui()
sleep(100)