using DelimitedFiles
using Sundials
using JuMP
using Ipopt  
Sei = true
Cum = true


# A. Number of node points
N1 = 20
N2 = 20
Ncp = 2 #number of concentration at the positive side
Ncn = 2 #number of concentration at the positive side
Nsei = 3
if !Sei
    Nsei = 0
end
Ncum = 0
if Cum
    Ncum += 2
    if Sei
        Ncum += 2
    end
end


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


area = 1  #0.3108
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


differential_vars = trues(Ncp + Ncn + 4 + Nsei + Ncum)
differential_vars[Ncp] = false
differential_vars[Ncp+Ncn] = false
differential_vars[Ncp+Ncn+1] = false
differential_vars[Ncp+Ncn+2] = false
differential_vars[Ncp+Ncn+3] = false
differential_vars[Ncp+Ncn+4] = false
if Sei
    differential_vars[Ncp+Ncn+5] = false
    differential_vars[Ncp+Ncn+6] = false
end

nyears = 2
Totaltime = nyears * 365 * 24 * 60 * 60                 # seconds
TotalHours = round(Int, Totaltime / 3600)         # hours
dt_FR = 2                                       # time step for FR
Nt_FR_hour = round(Int, 3600 / dt_FR)
Nt_FR_year = round(Int, 365 * 24 * 3600 / dt_FR)
Nt_FR = round(Int, Totaltime / dt_FR) + 1           # number of FR time step
TIME_FR = 0:dt_FR:Totaltime


Files = ["D:/vscode codes/BSS_MPC_deterministic-main/data/01_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/02_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/03_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/04_2017_Dynamic.csv",
    "D:/vscode codes/BSS_MPC_deterministic-main/data/05_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/06_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/07_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/08_2017_Dynamic.csv",
    "D:/vscode codes/BSS_MPC_deterministic-main/data/09_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/10_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/11_2017_Dynamic.csv", "D:/vscode codes/BSS_MPC_deterministic-main/data/12_2017_Dynamic.csv"]
function load_signals(files)
    signal = []
    for i in 1:length(files)
        filename = files[i]
        originalsignal = readdlm(filename, ',', Float64)[1:end-1, :]

        if i == 1
            signal = vcat(originalsignal...)
        else
            signal = [signal; vcat(originalsignal...)]
        end

        if i == length(files)
            signal = [signal; originalsignal[end, end]]
        end
    end
    return signal
end

signal = load_signals(Files[1:12])
signal = min.(max.(signal, -1), 1)
signal = repeat(signal; outer=[nyears])
signal = signal[1:Nt_FR]
FR_price = readdlm("D:/vscode codes/BSS_MPC_deterministic-main/data/FR_Incentive.csv", ',', Float64)
FR_price = vcat(FR_price'...)
FR_price = repeat(FR_price; outer=[nyears])

grid_price = readdlm("D:/vscode codes/BSS_MPC_deterministic-main/data/slow_Price.csv", ',', Float64)
grid_price = vcat(grid_price'...)
grid_price = repeat(grid_price; outer=[nyears])


# 修改 f_common 函数，移除点运算符
function f_common(out, du, u, p, t)
    input_power = p[1]  # 重命名避免冲突

    csp = u[1:Ncp]
    csn = u[Ncp+1:Ncp+Ncn]
    csp = max.(1, min.(csp, cspmax - 1))
    csn = max.(1, min.(csn, csnmax - 1))
    csp_avg = csp[1]
    csp_s = csp[2]
    csn_avg = csn[1]
    csn_s = csn[2]

    iint = u[Ncp+Ncn+1]
    phi_p = u[Ncp+Ncn+2]
    phi_n = u[Ncp+Ncn+3]
    pot = u[Ncp+Ncn+4]
    if Sei
        it = u[Ncp+Ncn+5]
        isei = u[Ncp+Ncn+6]
        delta_sei = u[Ncp+Ncn+7]
    else
        it = iint
        delta_sei = 0
    end
    if Cum
        cm = u[Ncp+Ncn+5+Nsei]
        cp = u[Ncp+Ncn+6+Nsei]
        if Sei
            Q = u[Ncp+Ncn+7+Nsei]
            cf = u[Ncp+Ncn+8+Nsei]
        end
    end

    ff = 1
    #C2. Additional Equations
    #Positive electrode
    theta_p = csp[Ncp] / cspmax
    
    # 修复：移除点运算符
    Up = 7.49983 - 13.7758 * theta_p^0.5 + 21.7683 * theta_p - 12.6985 * theta_p^1.5 + 0.0174967 / theta_p - 0.41649 * theta_p^(-0.5) - 
         0.0161404 * exp(100 * theta_p - 97.1069) + 0.363031 * tanh(5.89493 * theta_p - 4.21921)
    
    jp = 2 * kp * ce^0.5 * (cspmax - csp[Ncp])^0.5 * csp[Ncp]^0.5 * sinh(0.5 * F / R / T * (phi_p - Up))
    out[Ncp+Ncn+1] = (jp - (it / ap / F / lp))

    #Negative electrode
    theta_n = csn[Ncn] / csnmax
    
    # 修复：移除点运算符
    Un = 9.99877 - 9.99961 * theta_n^0.5 - 9.98836 * theta_n + 8.2024 * theta_n^1.5 + 0.23584 / theta_n - 2.03569 * theta_n^(-0.5) -
         1.47266 * exp(-1.14872 * theta_n + 2.13185) - 9.9989 * tanh(0.60345 * theta_n - 1.58171)
    
    jn = 2 * kn * ce^0.5 * (csnmax - csn[Ncn])^0.5 * csn[Ncn]^0.5 * sinh(0.5 * F / R / T * (phi_n - Un + (Rsei + delta_sei / Kappa_sei) * it / an / lnn))
    out[Ncp+Ncn+2] = (jn + (iint / an / F / lnn))
    out[Ncp+Ncn+3] = pot - phi_p + phi_n

    #C1. Governing Equations
    out[1] = -3 * jp / Rpp - du[1]
    out[2] = 5 * (csp_s - csp_avg) + Rpp * jp / Dp

    out[Ncp+1] = -3 * jn / Rpn - du[Ncp+1]
    out[Ncp+2] = 5 * (csn_s - csn_avg) + Rpn * jn / Dn

    if Sei
        out[Ncp+Ncn+4] = -iint + it - isei
        out[Ncp+Ncn+5] = -isei + an * lnn * ksei * exp(-1 * F / R / T * (phi_n - Urefs + it / an / lnn * (delta_sei / Kappa_sei + Rsei)))
        out[Ncp+Ncn+6] = isei * M_sei / F / rho_sei / an / lnn - du[Ncp+Ncn+7]
    end

    if Cum
        out[Ncp+Ncn+4+Nsei] = iint / 3600 - du[Ncp+Ncn+5+Nsei]
        out[Ncp+Ncn+5+Nsei] = it * pot / 3600 - du[Ncp+Ncn+6+Nsei]
        if Sei
            out[Ncp+Ncn+6+Nsei] = it / 3600 - du[Ncp+Ncn+7+Nsei]
            out[Ncp+Ncn+7+Nsei] = isei / 3600 - du[Ncp+Ncn+8+Nsei]
        end
    end
end

# 完全重写 getinitial 函数，避免所有可能的冲突
function getinitial(csp_avg, csn_avg, delta_sei, input_val, mode)
    println("DEBUG getinitial: input_val=$input_val, mode=$mode")
    
    try
        # 方法1：使用重命名的内部函数
        return getinitial_internal(csp_avg, csn_avg, delta_sei, input_val, mode)
    catch e
        println("WARNING: getinitial 失败: ", e)
        println("使用默认值...")
        # 返回合理的默认值
        pot_default = 3.3
        it_default = input_val / pot_default
        return csp_avg, csn_avg, it_default, 3.5, 0.2, pot_default, it_default, 0.0
    end
end

# 内部函数，避免命名冲突
function getinitial_internal(csp_avg, csn_avg, delta_sei, input_val, mode)
    # 使用完全限定的函数名
    m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    
    if !Sei
        delta_sei = 0
    end

    theta_p_guess = min(0.9, csp_avg / cspmax)
    theta_n_guess = min(0.9, csn_avg / csnmax)

    # 计算初始猜测 - 没有点运算符
    Un_guess_val = 9.99877 - 9.99961 * theta_n_guess^0.5 - 9.98836 * theta_n_guess + 8.2024 * theta_n_guess^1.5 + 
                   0.23584 / theta_n_guess - 2.03569 * theta_n_guess^(-0.5) -
                   1.47266 * exp(-1.14872 * theta_n_guess + 2.13185) -
                   9.9989 * tanh(0.60345 * theta_n_guess - 1.58171)
    
    Up_guess_val = 7.49983 - 13.7758 * theta_p_guess^0.5 + 21.7683 * theta_p_guess - 12.6985 * theta_p_guess^1.5 + 
                   0.0174967 / theta_p_guess - 0.41649 * theta_p_guess^(-0.5) -
                   0.0161404 * exp(100 * theta_p_guess - 97.1069) +
                   0.363031 * tanh(5.89493 * theta_p_guess - 4.21921)

    if mode == 1
        it0_val = input_val
    elseif mode == 2
        it0_val = TC / 2
    elseif mode == 3
        pot0_val = max(min(Up_guess_val - Un_guess_val, 3.3), 2.0)
        it0_val = input_val / pot0_val
    end
    
    # 定义所有变量
    @variable(m, it_var, start = it0_val)
    @variable(m, iint_var, start = it0_val)
    @variable(m, csp_s_var, start = csp_avg)
    @variable(m, csn_s_var, start = csn_avg)

    @constraint(m, 5 * (csp_s_var - csp_avg) + Rpp * it_var / F / Dp / ap / lp == 0)
    @constraint(m, 5 * (csn_s_var - csn_avg) - Rpn * iint_var / F / Dn / an / lnn == 0)

    @variable(m, 0 <= theta_p_var <= 1, start = theta_p_guess)
    @variable(m, 0 <= theta_n_var <= 1, start = theta_n_guess)

    @constraint(m, theta_p_var * cspmax == csp_s_var)
    @constraint(m, theta_n_var * csnmax == csn_s_var)

    @variable(m, Up_var, start = Up_guess_val)
    @variable(m, phi_p_var, start = Up_guess_val)
    @variable(m, Un_var, start = Un_guess_val)
    @variable(m, phi_n_var, start = Un_guess_val)
    
    # 简化约束表达式
    @NLconstraint(m, (cspmax - csp_s_var)^0.5 * csp_s_var^0.5 * sinh(0.5 * F / R / T * (phi_p_var - Up_var)) == 
                   it_var / ap / F / lp / (2 * kp * ce^0.5))
    
    @NLconstraint(m, (csnmax - csn_s_var)^0.5 * csn_s_var^0.5 * sinh(0.5 * F / R / T * (phi_n_var - Un_var + 
                   (delta_sei / Kappa_sei + Rsei) * it_var / an / lnn)) == 
                   -iint_var / an / F / lnn / (2 * kn * ce^0.5))

    # OCV 约束 - 分解为更小的部分避免长表达式
    @NLconstraint(m, Up_var == 
                   7.49983 - 13.7758 * theta_p_var^0.5 + 21.7683 * theta_p_var - 12.6985 * theta_p_var^1.5 + 
                   0.0174967 / theta_p_var - 0.41649 * theta_p_var^(-0.5) -
                   0.0161404 * exp(100 * theta_p_var - 97.1069) + 
                   0.363031 * tanh(5.89493 * theta_p_var - 4.21921))
    
    @NLconstraint(m, Un_var == 
                   9.99877 - 9.99961 * theta_n_var^0.5 - 9.98836 * theta_n_var + 8.2024 * theta_n_var^1.5 + 
                   0.23584 / theta_n_var - 2.03569 * theta_n_var^(-0.5) - 
                   1.47266 * exp(-1.14872 * theta_n_var + 2.13185) - 
                   9.9989 * tanh(0.60345 * theta_n_var - 1.58171))

    if Sei
        @variable(m, isei_var, start = 0.0)
        @NLconstraint(m, isei_var == an * lnn * ksei * exp(-F / R / T * 
                          (phi_n_var - Urefs + (delta_sei / Kappa_sei + Rsei) * it_var / an / lnn)))
    else
        isei_var = 0.0
    end
    
    @constraint(m, -iint_var + it_var - isei_var == 0)
    
    if mode == 1
        @constraint(m, it_var == input_val)
    elseif mode == 2
        @constraint(m, phi_p_var - phi_n_var == input_val)
    elseif mode == 3
        @NLconstraint(m, it_var * (phi_p_var - phi_n_var) == input_val)
    end
    
    optimize!(m)
    
    # 使用 JuMP.value 获取结果
    iint_result = JuMP.value(iint_var)
    csp_s_result = JuMP.value(csp_s_var)
    csn_s_result = JuMP.value(csn_s_var)
    phi_p_result = JuMP.value(phi_p_var)
    phi_n_result = JuMP.value(phi_n_var)
    pot_result = phi_p_result - phi_n_result
    it_result = JuMP.value(it_var)
    isei_result = Sei ? JuMP.value(isei_var) : 0.0
    
    return csp_s_result, csn_s_result, iint_result, phi_p_result, phi_n_result, pot_result, it_result, isei_result
end


V_max = 3.65
V_min = 2.0
maxC = 10
soc_retire = 0.8
soc_min = 0.2
soc_max = 0.8
soc_min_stop = 0.1
soc_max_stop = 0.9

soc = 0.5
csp_avg0 = cspmax * 1 - csnmax * soc * lnn * en / lp / ep            #49503.111
csn_avg0 = csnmax * soc
delta_sei0 = 1e-10
u0 = zeros(Ncp + Ncn + 4 + Nsei + Ncum)
u0[1:Ncp] .= csp_avg0
u0[(Ncp+1):(Ncp+Ncn)] .= csn_avg0
if Sei
    u0[Ncp+Ncn+7] = delta_sei0              #delta_sei
end
if Cum
    u0[Ncp+Ncn+Nsei+5] = 0                  #cm
    u0[Ncp+Ncn+Nsei+6] = 0                  #cp
    if Sei
        u0[Ncp+Ncn+Nsei+7] = 0              #Q
        u0[Ncp+Ncn+Nsei+8] = 0              #cf
    end
end





function MPC(u0, method)
    println("MPC start")
    start_time = time() # tic() TODO
    i_start = 1
    i_end = 1
    FR_band_list = zeros(TotalHours)
    buy_from_grid = zeros(TotalHours)
    capacity_remain_list = soc_retire * ones(TotalHours)
    csp_avg_list = zeros(Nt_FR_year + 1)
    csn_avg_list = zeros(Nt_FR_year + 1)
    delta_sei_list = zeros(Nt_FR_year + 1)
    cf_list = zeros(Nt_FR_year + 1)
    pot_list = zeros(Nt_FR_year + 1)
    it_list = zeros(Nt_FR_year + 1)
    isei_list = zeros(Nt_FR_year + 1)
    delta_sei_list = zeros(Nt_FR_year + 1)
    Q_list = zeros(Nt_FR_year + 1)
    P_FR = zeros(Nt_FR_year + 1)
    waste_list = zeros(Nt_FR_year + 1)
    retire_time = Totaltime
    retire_index = Nt_FR

    TIME_FR_segment = Int64[]
    P_FR_segment = Float64[]
    TIME_FR_list = []
    solt = []
    solu = []
    year = 1
    hours_moving = 1
    for i_start in 1:Nt_FR_hour*hours_moving:(Nt_FR-Nt_FR_hour)

        i_end = i_start + Nt_FR_hour * hours_moving
        i_start_hour = Int(floor(TIME_FR[i_start] / 3600)) + 1
        i_end_hour = i_start_hour + hours_moving - 1

        if method == "FP_MPC"
            filename = string("Result/FP_MPC/Min", soc_min, "MPC", "Horizon", nHours_Horizon, "year", year, ".jld")
        elseif method == "sim_MPC"
            filename = string("Result/sim_MPC/Min", soc_min, "MPC", "Horizon", nHours_Horizon, "year", year, ".jld")
        elseif method == "modified_sim_MPC"
            filename = string("Result/modified_sim_MPC/Min", soc_min, "MPC", "Horizon", nHours_Horizon, "year", year, ".jld")
        elseif method == "MPC_flexible"
            filename = string("Result/flex_MPC/Min", soc_min, "MPC", "Horizon", nHours_Horizon, "year", year, ".jld")
        elseif method == "Heuristic"
            filename = string("Result/Heuristic/Min", soc_min, "C", C, ".jld")
        end

        if TIME_FR[i_start] >= year * 3600 * 24 * 365
            save(filename, "csp_avg_list", csp_avg_list, "csn_avg_list", csn_avg_list, "delta_sei_list", delta_sei_list, "cf_list", cf_list, "pot_list", pot_list, "it_list", it_list, "isei_list", isei_list, "Q_list", Q_list, "P_FR", P_FR)
            year = year + 1
            csp_avg_list[:] = 0
            csn_avg_list[:] = 0
            delta_sei_list[:] = 0
            cf_list[:] = 0
            pot_list[:] = 0
            it_list[:] = 0
            isei_list[:] = 0
            delta_sei_list[:] = 0
            Q_list[:] = 0
            P_FR[:] = 0
            waste_list[:] = 0
        end

        du0 = zeros(Ncp + Ncn + 4 + Nsei + Ncum)
        cf0 = u0[Ncp+Ncn+Nsei+8]
        fade = cf0 / Qmax
        capacity_remain = 1 - fade
        csn_avg = u0[Ncp+1]
        soc = csn_avg / csnmax
        if capacity_remain <= soc_retire
            retire_time = TIME_FR[i_start]
            retire_index = i_start
            save(filename, "retire_time", retire_time, "FR_band_list", FR_band_list, "buy_from_grid", buy_from_grid, "capacity_remain_list", capacity_remain_list, "csp_avg_list", csp_avg_list, "csn_avg_list", csn_avg_list, "delta_sei_list", delta_sei_list, "cf_list", cf_list, "pot_list", pot_list, "it_list", it_list, "isei_list", isei_list, "Q_list", Q_list, "P_FR", P_FR, "waste_list", waste_list)
            break
        end

        soc_stop = true
        if method == "MPC_flexible"
            FR_band, grid_band, waste = OptimalControl(u0, TIME_FR[i_start], soc_min, soc_max)
        elseif method == "Heuristic"
            FR_band = FR_band0 * capacity_remain
            power_next = P_nominal * soc + FR_band * mean(signal[i_start:i_end])
            println("mean signal   ", mean(signal[i_start:i_end]), "    power_next    ", power_next, " soc  ", soc, "   ", P_nominal * soc, "    ", FR_band * mean(signal[i_start:i_end]))
            if power_next <= P_nominal * capacity_remain * soc_min
                grid_band = P_nominal * capacity_remain * soc_min - power_next
            elseif power_next >= P_nominal * capacity_remain * soc_max
                grid_band = P_nominal * capacity_remain * soc_max - power_next
            else
                grid_band = 0
            end
            println("FR_band    ", FR_band, "   grid_band  ", grid_band)
        else
            FR_band, grid_band = OptimalControl(u0, TIME_FR[i_start], soc_min, soc_max)
        end

        trial = 0
        while soc_stop # Numerical simulation, if soc does not meet the constraints, adjust the control amount until the constraints are met
            trial += 1
            TIME_FR_segment = TIME_FR[i_start:i_end]
            if method == "MPC_flexible"
                P_FR_segment = signal[i_start:i_end] * FR_band .+ grid_band .- waste
            else
                P_FR_segment = signal[i_start:i_end] * FR_band .+ grid_band
            end

            band_too_large = false
            level = soc
            for k = 1:(i_end-i_start)
                level += P_FR_segment[k] * dt_FR / 3600 / P_nominal
                if level <= soc_min_stop || level >= soc_max_stop
                    band_too_large = true
                    break
                end
            end
            if band_too_large
                if FR_band >= P_nominal
                    FR_band = FR_band - P_nominal / 2
                else
                    FR_band = FR_band / 2
                end
                power_next = P_nominal * soc + FR_band * mean(signal[i_start:i_end])
                if power_next <= P_nominal * capacity_remain * soc_min
                    grid_band = P_nominal * capacity_remain * soc_min - power_next
                elseif power_next >= P_nominal * capacity_remain * soc_max
                    grid_band = P_nominal * capacity_remain * soc_max - power_next
                else
                    grid_band = 0
                end
                if method == "MPC_flexible"
                    if grid_band >= 0
                        waste[1:end] = 0
                    else
                        waste[1:end] = -grid_band
                        grid_band = 0
                    end
                end
                println("band_too_large, update FR_band to  ", FR_band, "   grid_band to  ", grid_band)
                continue
            end

            tspan = (float(TIME_FR_segment[1]), float(TIME_FR_segment[end]))
            println("P_FR_segment   ", P_FR_segment[1], "   ", P_FR_segment[end])
            println("TIME_FR_segment (day)  ", TIME_FR_segment[1] / 3600 / 24, "    ", TIME_FR_segment[end] / 3600 / 24)
            itp = interpolate((TIME_FR_segment,), P_FR_segment, Gridded(Linear()))


            function stop_cond(u, t, integrator)
                csn_avg = u[Ncp+1]
                soc_in = csn_avg / csnmax
                min_stop = capacity_remain * soc_min_stop
                max_stop = capacity_remain * soc_max_stop

                if t <= 50
                    return (min_stop + max_stop) / 2
                end

                if soc_in >= (min_stop + max_stop) / 2
                    return (max_stop - soc_in)
                else
                    return (soc_in - min_stop)
                end
            end
            affect!(integrator) = terminate!(integrator)
            cb = ContinuousCallback(stop_cond, affect!, rootfind=true, interp_points=100)


            function f_FR(out, du, u, param, t)
    # 确保 t 是 Float64 类型
    t_float = Float64(t)
    
    # 检查 t 是否在插值器范围内
    if t_float < TIME_FR_segment[1] || t_float > TIME_FR_segment[end]
        # 如果超出范围，使用边界值
        if t_float < TIME_FR_segment[1]
            p_to_battery = P_FR_segment[1]
        else
            p_to_battery = P_FR_segment[end]
        end
    else
        # 安全地调用插值器
        try
            p_to_battery = itp(t_float)  # 使用圆括号而不是方括号
        catch e
            println("插值器调用错误 at t = $t_float: $e")
            # 使用最近的值
            idx = argmin(abs.(TIME_FR_segment .- t_float))
            p_to_battery = P_FR_segment[idx]
        end
    end
    
    if Sei
        it = u[Ncp+Ncn+5]
    else
        iint = u[Ncp+Ncn+1]
        it = iint
    end
    phi_p = u[Ncp+Ncn+2]
    phi_n = u[Ncp+Ncn+3]
    p = [p_to_battery]
    f_common(out, du, u, p, t)
    out[end] = (phi_p - phi_n) * it - p_to_battery
end

            ### modify algerbric variables
            csp_avg0 = u0[1]
            csn_avg0 = u0[Ncp+1]
            if Sei
                delta_sei0 = u0[Ncp+Ncn+7]
            else
                delta_sei0 = 0
            end
            println("csp_avg0, csn_avg0, delta_sei0, P_FR_segment[1] ", csp_avg0, "     ", csn_avg0, "     ", delta_sei0, "     ", "     ", P_FR_segment[1])
            csp_s0, csn_s0, iint0, phi_p0, phi_n0, pot0, it0, isei0 = getinitial(csp_avg0, csn_avg0, delta_sei0, P_FR_segment[1], 3)
            println("csp_s0, csn_s0, iint0, phi_p0, phi_n0, pot0, it0, isei0  ", csp_s0, "     ", csn_s0, "     ", iint0, "     ", phi_p0, "     ", phi_n0, "     ", pot0, "     ", it0, "     ", isei0)
            u0[Ncp] = csp_s0
            u0[Ncp+Ncn] = csn_s0
            u0[Ncp+Ncn+1] = iint0                  #iint
            u0[Ncp+Ncn+2] = phi_p0                 #phi_p
            u0[Ncp+Ncn+3] = phi_n0                 #phi_n
            u0[Ncp+Ncn+4] = pot0                   #pot
            if Sei
                u0[Ncp+Ncn+5] = it0                     #it
                u0[Ncp+Ncn+6] = isei0                   #isei
            end

            #println("u02   ", u0)
            # 修改微分方程求解调用
        prob = DAEProblem(f_FR, du0, u0, tspan, differential_vars=differential_vars)
    
        # 添加自动初始化选项
        sol = DifferentialEquations.solve(prob, IDA(), 
        callback=cb,
        initializealg = DiffEqBase.BrownFullBasicInit(),  # 添加自动初始化
        dtmax = 10.0,
        abstol = 1e-6,
        reltol = 1e-6,
        maxiters = 10000
        )
    
        println("微分方程求解成功，最终时间: $(sol.t[end])")

            csn_avg = (sol[end])[Ncp+1]
            soc_end = csn_avg / csnmax

            if sol.t[end] != tspan[end] || soc_end <= 0.1 || soc_end >= 0.9
                println("soc too large or small, stop at   ", sol.t[end])
                if FR_band >= P_nominal
                    FR_band = FR_band - P_nominal / 2
                else
                    FR_band = FR_band / 2
                end

                power_next = P_nominal * soc + FR_band * mean(signal[i_start:i_end])
                println("mean signal   ", mean(signal[i_start:i_end]), "    power_next    ", power_next, " soc  ", soc, "   ", P_nominal * soc, "    ", FR_band * mean(signal[i_start:i_end]))
                if power_next <= P_nominal * capacity_remain * soc_min
                    grid_band = P_nominal * capacity_remain * soc_min - power_next
                elseif power_next >= P_nominal * capacity_remain * soc_max
                    grid_band = P_nominal * capacity_remain * soc_max - power_next
                else
                    grid_band = 0
                end
                if method == "MPC_flexible"
                    if grid_band >= 0
                        waste[1:end] .= 0
                    else
                        waste[1:end] .= -grid_band
                        grid_band = 0
                    end
                end
            else
                states = sol(TIME_FR_segment)
                FR_band_list[i_start_hour:i_end_hour] .= FR_band
                buy_from_grid[i_start_hour:i_end_hour] .= grid_band
                capacity_remain_list[i_start_hour:i_end_hour] .= capacity_remain

                i_start_this = i_start - Nt_FR_year * (year - 1)
                i_end_this = i_end - Nt_FR_year * (year - 1)
                csp_avg_list[i_start_this:i_end_this] = states[1, :]
                csn_avg_list[i_start_this:i_end_this] = states[Ncp+1, :]
                delta_sei_list[i_start_this:i_end_this] = states[Ncp+Ncn+7, :]
                cf_list[i_start_this:i_end_this] = states[Ncp+Ncn+8+Nsei, :]

                pot_list[i_start_this:i_end_this] = states[Ncp+Ncn+4, :]
                it_list[i_start_this:i_end_this] = states[Ncp+Ncn+5, :]
                isei_list[i_start_this:i_end_this] = states[Ncp+Ncn+6, :]
                Q_list[i_start_this:i_end_this] = states[Ncp+Ncn+7+Nsei, :]
                P_FR[i_start_this:i_end_this] = P_FR_segment
                if method == "MPC_flexible"
                    waste_list[i_start_this:i_end_this] = waste
                end
                u0 = sol[end]
                soc_stop = false
            end
        end
        cf = u0[Ncp+Ncn+Nsei+8]
        fade_this = (cf - cf0) / Qmax
        fade = cf / Qmax
        capacity_remain = 1 - fade
        csn_avg = u0[Ncp+1]
        soc = csn_avg / csnmax
        println("   fade this    ", fade_this, "    capacity_remain    ", capacity_remain, "    soc     ", soc)
    end

    println("elapsed time:     ", toc())


    Area = 1e6 / P_nominal
    battery_price_per_kwh = 200
    battery_price = P_nominal * Area / 1000 * battery_price_per_kwh
    revenue = 0
    for i in 1:TotalHours
        price = FR_price[i]
        FR_band = FR_band_list[i]
        revenue += price * FR_band * Area / 1000
    end

    cost = 0
    for i in 1:TotalHours
        price = grid_price[i]
        grid_band = buy_from_grid[i]
        if grid_band > 0
            cost += price * grid_band * Area / 1000
        end
    end

    profit = revenue - cost - battery_price
    println("retire_time:  ", retire_time)
    println("revenue,   operational cost, battery_price,  profit:    ", revenue, "   ", cost, "   ", battery_price, "    ", profit)
    println("charging times(h):  ", length(find(buy_from_grid .> 0)))
    println("discharging times(h):  ", length(find(buy_from_grid .< 0)))
end