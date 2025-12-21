# bss_simulator.jl

using DelimitedFiles
using Sundials
using Interpolations
using ModelingToolkit
using BlockDiagonals
using LinearAlgebra
using SymbolicIndexingInterface: parameter_values

include("setup.jl")
include("utils.jl")

function f_common(out, du, u, p, t)
    # 1. 解包状态变量
    csp = u[1:Ncp]
    csn = u[(Ncp+1):(Ncp+Ncn)]
    
    csp_avg = csp[1]
    csp_s = csp[2]       # 原始变量，可能越界
    csn_avg = csn[1]
    csn_s = csn[2]       # 原始变量，可能越界

    iint = u[Ncp+Ncn+1]
    phi_p = u[Ncp+Ncn+2]
    phi_n = u[Ncp+Ncn+3]
    pot = u[Ncp+Ncn+4]

    it = u[Ncp+Ncn+5]
    isei = u[Ncp+Ncn+6]
    delta_sei = u[Ncp+Ncn+7]

    cf = u[Ncp+Ncn+Nsei+5]

    # ==========================================
    # 🛡️ 数值安全保护 (Numerical Safeguards)
    # ==========================================
    # 用于计算 OCV 和 J (电流密度) 的中间变量
    # 即使求解器猜测了离谱的值（如负数），这里强制将其拉回安全范围进行计算
    # 这样可以防止 sqrt() 报错，同时让求解器通过残差方程感知到错误
    
    csp_s_safe = clamp(csp_s, 1.0, cspmax - 1.0)
    csn_s_safe = clamp(csn_s, 1.0, csnmax - 1.0)
    
    # 确保 theta 在 (0, 1) 之间，防止 OCV 公式中的 log 或负指数爆炸
    theta_p = clamp(csp_s_safe / cspmax, 1e-5, 0.99999)
    theta_n = clamp(csn_s_safe / csnmax, 1e-5, 0.99999)

    # ==========================================
    # 2. 辅助方程 (使用安全变量计算)
    # ==========================================
    
    # Positive electrode OCV
    Up = 7.49983 - 13.7758 * theta_p^0.5 + 21.7683 * theta_p - 12.6985 * theta_p^1.5 + 
         0.0174967 / theta_p - 0.41649 * theta_p^(-0.5) - 
         0.0161404 * exp(100 * theta_p - 97.1069) + 
         0.363031 * tanh(5.89493 * theta_p - 4.21921)
    
    # Butler-Volmer (Positive) - 使用 safe 变量防止 DomainError
    # (cspmax - csp_s_safe) 保证非负
    jp = 2 * kp * ce^(0.5) * (cspmax - csp_s_safe)^(0.5) * csp_s_safe^(0.5) * sinh(0.5 * F / R / T * (phi_p - Up))

    # Negative electrode OCV
    Un = 9.99877 - 9.99961 * theta_n^0.5 - 9.98836 * theta_n + 8.2024 * theta_n^1.5 + 
         0.23584 / theta_n - 2.03569 * theta_n^(-0.5) -
         1.47266 * exp(-1.14872 * theta_n + 2.13185) -
         9.9989 * tanh(0.60345 * theta_n - 1.58171)
         
    # Butler-Volmer (Negative)
    jn = 2 * kn * ce^(0.5) * (csnmax - csn_s_safe)^(0.5) * csn_s_safe^(0.5) * sinh(0.5 * F / R / T * (phi_n - Un + (Rsei + delta_sei / Kappa_sei) * it / an / lnn))

    # ==========================================
    # 3. 控制方程 (Residual Equations)
    # ==========================================
    # 注意：这里必须使用【原始变量】(csp_s, csn_s)
    # 这样当 csp_s 越界时，残差会变大，驱动求解器回到正确范围
    
    # Positive electrode
    out[1] = -3 * jp / Rpp - du[1]               # d(csp_avg)/dt
    out[2] = 5 * (csp_s - csp_avg) + Rpp * jp / Dp # Algebraic: surface concentration

    # Negative electrode
    out[3] = -3 * jn / Rpn - du[Ncp+1]           # d(csn_avg)/dt
    out[4] = 5 * (csn_s - csn_avg) + Rpn * jn / Dn # Algebraic

    out[5] = (jp - (it / ap / F / lp))
    out[6] = (jn + (iint / an / F / lnn))
    out[7] = pot - phi_p + phi_n
    out[8] = p - pot * it  # Power balance: p is input power

    # SEI layer Equations
    out[9] = -iint + it - isei
    # Tafel equation for SEI
    out[10] = -isei + an * lnn * ksei * exp(-1 * F / R / T * (phi_n - Urefs + it / an / lnn * (delta_sei / Kappa_sei + Rsei)))
    out[11] = isei * M_sei / F / rho_sei / an / lnn - du[Ncp+Ncn+4+3]

    # Charge stored / Capacity fade
    out[12] = isei / 3600 - du[Ncp+Ncn+Nsei+5]
end

mutable struct Simulator
  prob
  tspan

  function Simulator(u0_single, battery_charge_scalar)
    TIME_SEGMENT = 0:dt:DT_STATE
    tspan = (float(TIME_SEGMENT[1]), float(TIME_SEGMENT[end]))

    differential_vars = falses(Ncp + Ncn + 4 + Nsei + Ncum)
    differential_vars[1] = true   # csp_avg
    differential_vars[3] = true   # csn_avg
    differential_vars[11] = true  # delta_sei
    differential_vars[12] = true  # cf

    du0 = zeros(length(u0_single))
    
    # 初始化 DAE Problem
    # 注意：battery_charge_scalar 必须是 parameter (p)
    prob = DAEProblem(
      f_common,
      du0,
      u0_single,
      tspan,
      battery_charge_scalar;
      differential_vars = differential_vars
    )

    new(prob, tspan)
  end
end

function simulate(
    simulator::Simulator,
    init_solver::InitialStateSolver,
    u0::Matrix,
    grid_price::Float64,
    battery_charge::Vector,
    swap_battery_idx::Vector,
    swap_battery_state::Matrix;
    log::Bool = false
)
    swap_set = Set(swap_battery_idx)
    start_time = time()
    
    u1 = zero(u0)
    cf0 = u0[:, Ncp+Ncn+Nsei+5]
    fade = cf0 ./ Qmax
    capacity_remain = 1 .- fade

    for k in 1:NUM_BATTERIES_IN_STATION
        # 0. 默认继承上一时刻状态
        u1[k, :] = u0[k, :]

        # 1. 如果是刚换进来的电池，只做初始化平衡，不进行动态演化
        if k in swap_set
            # 使用 0 电流平衡
            u1[k, :] .= safe_initial_state(init_solver, u0[k, :], 0.0)
            continue
        end

        # 2. 正常电池：先计算一致性初值
        # 使用当前指令功率 battery_charge[k] 计算代数变量的初值
        u0_consistent = safe_initial_state(init_solver, u0[k, :], battery_charge[k])

        # 定义停止条件 (SOC 保护)
        function stop_cond(u, t, integrator)
            csn_avg = u[Ncp+1]
            soc_in = csn_avg / csnmax
            min_stop = capacity_remain[k] * soc_min_stop
            max_stop = capacity_remain[k] * soc_max_stop

            if soc_in >= (min_stop + max_stop) / 2
                return max_stop - soc_in
            else
                return soc_in - min_stop
            end
        end
        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(stop_cond, affect!; rootfind = true)

        # 3. 构建并求解 DAE
        # 关键：remake 时传入当前的功率 p
        prob = remake(
            simulator.prob,
            u0 = copy(u0_consistent),
            p  = Float64(battery_charge[k])
        )

        # 增加 maxiters 防止死循环，放宽一点公差
        sol = solve(
            prob,
            IDA();
            initializealg = DiffEqBase.BrownFullBasicInit(),
            reltol = 1e-3,  # 稍微放宽，提高稳定性
            abstol = 1e-3,
            maxiters = 5000,
            callback = cb,
            verbose = false
        )

        # 4. 处理结果
        if sol.retcode == :Success || sol.retcode == :Terminated
            u1[k, :] = sol.u[end]
            
            # 如果是因为功率过大导致电压截止，这里做一个简单的线性缩放修正实际功率
            # (可选优化：根据 t_end 修正 battery_charge)
            if sol.t[end] < simulator.tspan[end] && abs(battery_charge[k]) > 1e-3
                 # 实际上这里只是记录状态，功率修正通常在 profit 计算时体现
            end
        else
            if log
                println("⚠️ Battery $k DAE failed ($(sol.retcode)). Using Quasi-Static fallback.")
            end
            
            # 失败回退：使用准静态投影
            u_proj, flag = quasi_static_project(init_solver, u0_consistent, Float64(battery_charge[k]))
            
            if flag == :success
                u1[k, :] = u_proj
            else
                # 彻底失败：保持静置
                u1[k, :] = compute_initial_state(init_solver, u0[k, :], 0.0)
                battery_charge[k] = 0.0 # 强制功率为0
            end
        end
    end

    # ================================
    # 后处理：计算收益和统计
    # ================================
    delta_sei_list = copy(u1[:, 11])
    cf_list = copy(u1[:, 12])

    # 简单的退化成本估算
    cost = (expectedrevenue / (1 - soc_retire)) * sum((u1[:, 12] - u0[:, 12])) / Qmax

    soc_cost_coeffi = 20.0
    num_soc_violate = [0, length(swap_battery_idx)]

    for i in 1:length(swap_battery_idx)
        idx = swap_battery_idx[i]
        # 检查被换走的电池 SOC 是否合格
        # 注意：这里检查的是 u1 (演化后) 还是 u0 (换电前)? 
        # 逻辑上应该是检查换电前的状态是否满足需求，但在 MPC 框架下通常假设满足
        soc = u0[idx, 3] / csnmax # 使用换电前的状态
        
        if soc < replaceable_soc
            cost += soc_cost_coeffi * (replaceable_soc - soc)
            num_soc_violate[1] += 1
        end

        # 执行换电：状态覆盖
        u1[idx, 1] = swap_battery_state[i, 1] # csp_avg
        u1[idx, 2] = swap_battery_state[i, 1] # csp_s (reset to eq)
        u1[idx, 3] = swap_battery_state[i, 2] # csn_avg
        u1[idx, 4] = swap_battery_state[i, 2] # csn_s (reset to eq)
        u1[idx, 11] = swap_battery_state[i, 3] # delta_sei
        u1[idx, 12] = swap_battery_state[i, 4] # cf
        
        # 将代数变量重置为 0 (静置状态)，等待下一轮 InitialStateSolver 计算
        u1[idx, 5:10] .= 0.0 
    end

    profit = -grid_price * sum(battery_charge) - cost

    if log
        println("Simulation step done. Profit: $profit")
    end

    return u1, profit, battery_charge, delta_sei_list, cf_list, num_soc_violate
end
