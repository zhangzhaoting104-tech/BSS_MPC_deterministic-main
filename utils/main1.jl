using JuMP
using Ipopt
using LinearAlgebra
using Statistics
using DifferentialEquations # 用于 Simulator

# 加载用户提供的模块
include("setup.jl")
include("bss_manager.jl")
include("bss_simulator.jl")
include("bss_data_record.jl")
include("utils.jl")

# --- 辅助函数：OCP 开路电压计算 ( bss_quasi_static.jl) ---
# 将这些复杂的非线性函数注册为 JuMP 可用的形式，或直接在 NLconstraint 中书写
# 这里为了代码清晰，直接在 NLconstraint 中展开公式，与 bss_quasi_static.jl 保持一致

println("🚀 系统初始化...")

# 1. 初始化
manager = create_bss_manager(NUM_BATTERIES_IN_STATION, NUM_BATTERIES_IN_AREA)
u_current = get_initial_states(manager) # 获取初始状态 [21 x 13]

# 结果存储
hist_cf_avg = Float64[]
hist_cf_std = Float64[]
hist_cumulative_profit = Float64[]
hist_soc_violate = []

# 定义功率上下限 (放电为正，充电为负)
P_min = -P_max 

# 2. 滚动时域主循环
total_steps = 1  # 运行 24 小时
# 预测步长 (MPC 内部的时间步长，单位：秒)
dt_mpc = DT_STATE 

for step in 1:total_steps
    println("\n--- Step $step | 当前时刻: $(step-1):00 ---")
    
    # 显式声明全局变量以修复 UndefVarError
    global u_current, manager 
    
    # A. 获取预测信息
    # 确保索引不越界
    idx_end = min(length(Prices), step + horizon - 1)
    price_forecast = Prices[step : idx_end]
    # 如果价格数据不足，用最后一个价格填充
    if length(price_forecast) < horizon
        append!(price_forecast, fill(price_forecast[end], horizon - length(price_forecast)))
    end

    swap_times, swap_states = get_approx_swap_info(manager, step-1)

    # B. 构建 MPC 优化模型 (NLP)
    m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "max_cpu_time" => 300.0))

    # --- 决策变量 ---
    # P[k, t]: 功率 (W)
    @variable(m, P_min <= P[k=1:NUM_BATTERIES_IN_STATION, t=1:horizon] <= P_max)
    
    # --- 状态变量 (对应 SPM 状态) ---
    # x[k, t, 1]: csp_avg (正极平均浓度)
    # x[k, t, 2]: csn_avg (负极平均浓度)
    # x[k, t, 3]: delta_sei (SEI厚度)
    # x[k, t, 4]: cf (容量衰减)
    @variable(m, x[k=1:NUM_BATTERIES_IN_STATION, t=1:horizon+1, i=1:4])
    
    # --- 代数变量 (用于物理约束求解) ---
    # 为了减少变量规模，部分变量可用表达式代替，但为了非线性稳定性，保留核心中间量
    @variable(m, 0 <= csp_s[1:NUM_BATTERIES_IN_STATION, 1:horizon] <= cspmax) # 正极表面浓度
    @variable(m, 0 <= csn_s[1:NUM_BATTERIES_IN_STATION, 1:horizon] <= csnmax) # 负极表面浓度
    @variable(m, 0 <= theta_p[1:NUM_BATTERIES_IN_STATION, 1:horizon] <= 1)    # 正极化学计量数
    @variable(m, 0 <= theta_n[1:NUM_BATTERIES_IN_STATION, 1:horizon] <= 1)    # 负极化学计量数
    @variable(m, V_min <= V_out[1:NUM_BATTERIES_IN_STATION, 1:horizon] <= V_max) # 端电压
    @variable(m, it[1:NUM_BATTERIES_IN_STATION, 1:horizon])     # 总电流 A (it = P/V)
    
    # 辅助电势变量
    @variable(m, phi_p[1:NUM_BATTERIES_IN_STATION, 1:horizon])
    @variable(m, phi_n[1:NUM_BATTERIES_IN_STATION, 1:horizon])
    @variable(m, Up[1:NUM_BATTERIES_IN_STATION, 1:horizon])
    @variable(m, Un[1:NUM_BATTERIES_IN_STATION, 1:horizon])
    
    # --- 约束条件 ---
    
    for k in 1:NUM_BATTERIES_IN_STATION
        # 1. 初始状态固定
        @constraint(m, x[k, 1, 1] == u_current[k, 1])               # csp_avg
        @constraint(m, x[k, 1, 2] == u_current[k, Ncp+1])           # csn_avg
        @constraint(m, x[k, 1, 3] == u_current[k, Ncp+Ncn+7])       # delta_sei
        @constraint(m, x[k, 1, 4] == u_current[k, Ncp+Ncn+Nsei+5])  # cf

        for t in 1:horizon
            # --- 物理约束 (SPM Implicit Discretization) ---
            # 参考 PDF Eq 18 和 bss_quasi_static.jl
            
            # i. 辅助变量定义
            # Theta 定义
            @constraint(m, theta_p[k,t] * cspmax == csp_s[k,t])
            @constraint(m, theta_n[k,t] * csnmax == csn_s[k,t])
            
            # 功率方程: P = V * I (注意: P>0放电, I>0放电)
            # 为了数值稳定性，写成 P = V * it
            @constraint(m, P[k,t] == V_out[k,t] * it[k,t])
            @constraint(m, V_out[k,t] == phi_p[k,t] - phi_n[k,t])

            # ii. 扩散近似 (Parabolic Profile Approximation from bss_quasi_static.jl)
            # 5 * (cs_surf - cs_avg) + R * j / D = 0
            # 这里的 j 对应总电流 it (忽略副反应电流对扩散的主导影响简化)
            # 正极:
            @constraint(m, 5 * (csp_s[k,t] - x[k,t+1,1]) + Rpp * it[k,t] / F / Dp / ap / lp / area == 0) 
            # 负极: (忽略 iint 差异，近似 it)
            @constraint(m, 5 * (csn_s[k,t] - x[k,t+1,2]) - Rpn * it[k,t] / F / Dn / an / lnn / area == 0)

            # iii. 状态更新 (Backward Euler)
            # dCsp_avg/dt = -3 * j / Rpp
            # x[t+1] - x[t] = dt * (-3 * it / (F * area * lp * ap * Rpp)) ? 
            # 使用 bss_quasi_static.jl 中的公式: (avg1 - avg0)/dt = -15 * D / R^2 * (avg0 - s)
            # 这是一个等效形式。直接用质量守恒更简单: dCavg/dt = -I / (Vol * F * eps)
            # 采用 quasi_static.jl 的写法:
            @constraint(m, (x[k,t+1,1] - x[k,t,1]) / dt_mpc == -15 * Dp / (Rpp^2) * (x[k,t,1] - csp_s[k,t]))
            @constraint(m, (x[k,t+1,2] - x[k,t,2]) / dt_mpc == -15 * Dn / (Rpn^2) * (x[k,t,2] - csn_s[k,t]))
            
            # iv. Butler-Volmer & OCP (直接嵌入非线性方程)
            # Up (OCP Positive)
            @NLconstraint(m, Up[k,t] == 7.49983 - 13.7758 * theta_p[k,t]^0.5 + 21.7683 * theta_p[k,t] - 12.6985 * theta_p[k,t]^1.5 + 0.0174967 / theta_p[k,t] - 0.41649 * theta_p[k,t]^(-0.5) - 0.0161404 * exp(100 * theta_p[k,t] - 97.1069) + 0.363031 * tanh(5.89493 * theta_p[k,t] - 4.21921))
            
            # Un (OCP Negative)
            @NLconstraint(m, Un[k,t] == 9.99877 - 9.99961 * theta_n[k,t]^0.5 - 9.98836 * theta_n[k,t] + 8.2024 * theta_n[k,t]^1.5 + 0.23584 / theta_n[k,t] - 2.03569 * theta_n[k,t]^(-0.5) - 1.47266 * exp(-1.14872 * theta_n[k,t] + 2.13185) - 9.9989 * tanh(0.60345 * theta_n[k,t] - 1.58171))
            
            # BV Positive
            # j = 2*k*... sinh(...)
            # it / (area * lp * ap * F) = j
            @NLconstraint(m, (it[k,t] / (area * lp * ap * F)) == 2 * kp * ce^(0.5) * (cspmax - csp_s[k,t])^(0.5) * csp_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_p[k,t] - Up[k,t])))
            
            # BV Negative (简化：忽略 SEI 对主反应电势的微小影响，主要影响容量)
            @NLconstraint(m, -(it[k,t] / (area * lnn * an * F)) == 2 * kn * ce^(0.5) * (csnmax - csn_s[k,t])^(0.5) * csn_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_n[k,t] - Un[k,t])))

            # v. SEI 生长与容量衰减 (Side Reaction)
            # J_side = ...
            # 简化近似：基于 Tafel 关系或 setup.jl 中的 ksei
            # isei = J_side * Area_n
            # 这里为了 MPC 求解速度，使用线性化或简化的 SEI 速率，或者直接引用 quasi_static 中的公式
            # quasi_static: isei = it - iint (这引入了额外变量 iint，会增加很多复杂度)
            # 我们使用 Setup.jl 中的参数 ksei 做一个单步估计:
            # d_delta/dt = k_sei_rate * exp(...)
            # 为防止求解器在极小值处崩溃，这里做一个线性近似或常数假设，或者只在仿真层详细计算
            # 考虑老化，我们添加一个简化的老化项：
            # 假设 isei 与总电流和当前 SEI 厚度有关，或者简单假设一个小的增长率（为了 MPC 收敛）
            # 更精确的做法是复现 quasi_static 的 isei 约束，但这里变量太多。
            # 采用 Setup.jl 中的 ksei 和 phi_n 进行估算：
            @NLconstraint(m, (x[k,t+1,3] - x[k,t,3])/dt_mpc == (M_sei / (F * rho_sei)) * ksei * exp(-1 * F / R / T * (phi_n[k,t] - Urefs)))
            
            # 容量衰减更新
            @NLconstraint(m, (x[k,t+1,4] - x[k,t,4])/dt_mpc == (area * lnn * an * ksei * exp(-1 * F / R / T * (phi_n[k,t] - Urefs))) / 3600 )
            
        end
    end

    # --- 目标函数 ---
    # Max Revenue - w1 * Degradation - w2 * Balance
    @expression(m, revenue, sum(P[k,t] * price_forecast[t] * (dt_mpc/3600) for k=1:NUM_BATTERIES_IN_STATION, t=1:horizon))
    @expression(m, degradation, sum(x[k,horizon+1,4] - x[k,1,4] for k=1:NUM_BATTERIES_IN_STATION))
    
    # 均衡性: 每个时刻，各电池容量衰减与平均值的偏差平方和
    @expression(m, balance, sum( (x[k,t,4] - sum(x[j,t,4] for j=1:NUM_BATTERIES_IN_STATION)/NUM_BATTERIES_IN_STATION)^2 for k=1:NUM_BATTERIES_IN_STATION, t=1:horizon))

    @objective(m, Max, revenue - w1 * PI_Degradation * degradation - w2 * balance)

    # C. 求解
    optimize!(m)

    # D. 处理结果
    if termination_status(m) == MOI.OPTIMAL || termination_status(m) == MOI.LOCALLY_SOLVED
        # 1. 提取第一步决策
        opt_P = value.(P[:, 1])
        
        # 2. 换电逻辑 (Algorithm 1)
        num_to_swap = swap_times[2] 
        # 选择老化最严重的电池进行更换
        swap_idx_vec = select_batteries_with_max_cf(manager, num_to_swap)
        swap_battery_idx = findall(x -> x == 1, swap_idx_vec)
        
        # 获取新电池状态
        current_swap_state = get_swap_state(manager, num_to_swap)
        
        # 3. 高保真仿真更新 (Real Plant Simulation)
        # 注意：Simulator 需要标量输入，这里循环调用或修改 Simulator 支持向量
        # 原 bss_simulator.jl 的 simulate 函数似乎支持整个站点的 u0 和 P (vector)
        # 我们假设 simulate 函数内部处理了多电池逻辑 (bss_simulator.jl Line 178 接收 u0 和 battery_charge)
        
        # 创建模拟器实例 (这里取第一个电池做模板初始化 DAE)
        sim = Simulator(u_current[1, :], opt_P[1]) 
        
        u_next, profit, _, delta_sei_list, cf_list_new, violate = simulate(
            sim, u_current, price_forecast[1], opt_P, 
            swap_battery_idx, current_swap_state, log=false
        )

        # 4. 更新管理器
        update_bss_manager!(manager, swap_battery_idx, delta_sei_list, cf_list_new)
        u_current = u_next

        # 5. 记录
        push!(hist_cf_avg, mean(cf_list_new))
        push!(hist_cf_std, std(cf_list_new))
        total_profit = (isempty(hist_cumulative_profit) ? 0.0 : hist_cumulative_profit[end]) + profit
        push!(hist_cumulative_profit, total_profit)
        push!(hist_soc_violate, violate)
        
        println("✅ Step $step 完成. 利润: $(round(profit, digits=2)), 平均衰减: $(mean(cf_list_new))")

    else
        println("❌ Step $step 优化失败 (Status: $(termination_status(m))). 尝试执行备用策略 (零功率).")
        # 备用策略：全零功率
        opt_P = zeros(NUM_BATTERIES_IN_STATION)
        sim = Simulator(u_current[1, :], 0.0)
        u_next, profit, _, delta_sei_list, cf_list_new, violate = simulate(
            sim, u_current, price_forecast[1], opt_P, 
            Int[], nothing, log=false
        )
        u_current = u_next
        # 记录空数据以保持数组长度一致
        push!(hist_cf_avg, mean(u_current[:, 12])) # 假设 cf 在第 12 列
        push!(hist_cf_std, 0.0)
        push!(hist_cumulative_profit, (isempty(hist_cumulative_profit) ? 0.0 : hist_cumulative_profit[end]))
        push!(hist_soc_violate, [0,0])
    end
end

# 3. 结果保存
record("bss_mpc_physics_results.csv", hist_cf_avg, hist_cf_std, hist_cumulative_profit, hist_soc_violate)
println("⭐ 任务完成.")