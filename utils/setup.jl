using DelimitedFiles
using CSV
using XLSX
using DataFrames
using JLD

include("utils.jl")
include("D:/vscode codes/BSS_MPC_deterministic-main/GP_Model/Kriging.jl")


#=
A. Data
=#
Price_File = "D:/vscode codes/BSS_MPC_deterministic-main/data/rtm_cal.csv"
Demand_File = "D:/vscode codes/BSS_MPC_deterministic-main/data/长泰国际金融大厦.xlsx"

# Prices = CSV.read(Price_File, DataFrame, header=false)[:, 1]
# println(Prices[1:10])

Prices = readdlm("D:/vscode codes/BSS_MPC_deterministic-main/data/slow_Price.csv", ',', Float64)
Prices = vcat(Prices'...)

swap_counts_data = load_hourly_data(Demand_File)
swap_counts_data = vcat(swap_counts_data, swap_counts_data)
swap_counts_cycle = 1 * 60 * 60 # counts swap number every 1 hour
# println(size(swap_counts_data))
# println(swap_counts_data[1:30])


#=
B. Parameters
1: cathode, 2: anode
Dp,Dn: Solid phase diffusivity (m2/s),
kp, kn: Rate constant for lithium intercalation reaction (m2.5/(mol0.5s))
cspmax,csnmax: Maximum solid phase concentration at positive (mol/m3),
lp,ln : Region thickness (m),
ap,an: Particel surface area to volume (m2/m3)
Rpp ,Rpn: particle radius at positive (m),
ce: Electrolyte Concentration (mol/m3)
M[sei]: Molecular weight of SEI (Kg/mol)
Kappa[sei]: SEI ionic conductivity (S/m)
rho[sei]: SEI density (Kg/m3)
ksei: rate constant of side reaction (C m/s mol)
F : Faraday's constant (C/mol) , R : Ideal gas constant (J/K/mol), T : Temperature (K) 
=#

F = 96487
R = 8.3143
T = 298.15
M_sei = 0.073
Kappa_sei = 5e-6
rho_sei = 2.1e3
ksei = 1.5e-12                            #1e-8
Urefs = 0.4
Rsei = 0.01


area = 0.3108
cspmax = 10350
csnmax = 29480
lp = 6.521e-5
lnn = 2.885e-5
Rpp = 1.637e-7
Rpn = 3.596e-6
ce = 1042
ep = 1 - 0.52
en = 1 - 0.619
ap = 3 * ep / Rpp
an = 3 * en / Rpn
Sp = area * lp * ap
Sn = area * lnn * an
kp = 1.127e-7 / F
kn = 8.696e-7 / F
Dn = 8.256e-14
Dp = 1.736e-14
TC = 2.3 / 0.3108
Qmax = TC
P_nominal = TC * 3.1

V_max = 3.65
V_min = 2.0
maxC = 10
soc_retire = 0.8
soc_min = 0.2 # ensure that there is enough soc at the next horizon
soc_max = 0.8
soc_min_stop = 0.1
soc_max_stop = 0.9

replaceable_soc = 0.7 # soc at which can be swapped
P_max = P_nominal * maxC / 12


#=
C. Number of node points
=#
N1 = 20
N2 = 20
Ncp = 2 # number of concentration at the positive side, csp_avg, csp_s
Ncn = 2 # number of concentration at the negative side, csn_avg, csn_s
Nsei = 3 # it, isei, delta_sei
Ncum = 1 # cf
Nu = Ncp + Ncn + 4 + Nsei + Ncum


#=
D. battery statistics
=#
# number of battery
NUM_BATTERIES_IN_AREA = 200
NUM_BATTERIES_IN_STATION = 21
# capacity list
cf_list = zeros(NUM_BATTERIES_IN_AREA) # recording battery's capacity fade, initial value is 0


#=
E. mpc settings
# state: battery states, DT_STATE = 60(min)
# input: buy_from_grid, DT_INPUT = 60(min)
# HORIZON = 1(h)
=#
dt = 2.0 # simulator
horizon = 24 # hour
state_cycle = 60 * 60
input_cycle = 60 * 60 # price changes every 60 min
DT_STATE = state_cycle
DT_INPUT = input_cycle
HORIZON = horizon * swap_counts_cycle

TIME_STATE = 0:DT_STATE:HORIZON
Nt_STATE = round(Int, HORIZON ./ DT_STATE) + 1
Nt_INPUT = round(Int, HORIZON ./ DT_INPUT)

# expected revenue
expectedrevenue = 10 * P_nominal

# epsilon penalty
penalty_deviation = 100000
penalty_selection_count = 1000
penalty_repeated_selection = 1000

value_loss_coeffi = 1000
soc_shortage_discount_coeffi = 1000

# soc change penalty
delta_soc_penalty_coeffi = 10

# # === 在 setup.jl 的 E 部分添加或修改以下参数 ===
# w1 = 1e2  # 电池退化惩罚权重 (对应 PDF 中的 w1) 
# w2 = 1e1  # 使用均衡惩罚权重 (对应 PDF 中的 w2) 
# PI_Degradation = 5000.0  # 电池退化惩罚因子 (对应 PDF 中的 Pi) [cite: 108]

# 确保其他物理量与PDF一致
# V_max, V_min 等已定义 [cite: 50, 134]
#=
F. surrogate settings
=#
# load params
data = load("D:/vscode codes/BSS_MPC_deterministic-main/GP_Model/gp_model_params.jld")

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
df = CSV.read("D:/vscode codes/BSS_MPC_deterministic-main/GP_Model/battery_data.csv", DataFrame)
x = Matrix(df[:, [:x_next1, :x_next2, :x_next3, :x_next4, :u]])
y = Matrix(df[:, [:x1, :x2, :x3, :x4]])
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