using CSV
using DataFrames
using Plots


df = CSV.read("battery_data.csv", DataFrame)

x_cur = Matrix(df[:, [:x1, :x2, :x3, :x4]])
u = Matrix(df[:, [:u]])
x_next = Matrix(df[:, [:x_next1, :x_next2, :x_next3, :x_next4]])


csp_avg = x_next[:, 1]-x_cur[:, 1]
csn_avg = x_next[:, 2]-x_cur[:, 2]
delta_sei = x_next[:, 3]-x_cur[:, 3]
cf = x_next[:, 4]-x_cur[:, 4]

n = 1:length(csp_avg)
step = 50
indices = 1:step:length(csp_avg)

p1 = plot(indices, csp_avg[indices], label="csp_avg", xlabel="Sample Index", ylabel="Value", title="CSP AVG", linewidth=2)
p2 = plot(indices, csn_avg[indices], label="csn_avg", xlabel="Sample Index", ylabel="Value", title="CSN AVG", linewidth=2)
p3 = plot(indices, delta_sei[indices], label="delta_sei", xlabel="Sample Index", ylabel="Value", title="Delta SEI", linewidth=2)
p4 = plot(indices, cf[indices], label="cf", xlabel="Sample Index", ylabel="Value", title="CF", linewidth=2)
plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))

gui()
sleep(100)
