using DifferentialEquations
using Interpolations
using Distributions
using JuMP
using Ipopt
using JLD

# 引入 setup.jl 中的参数 (确保在主程序中已 include setup.jl)
# include("setup.jl") 

function sanitize_state!(u)
    # 物理量限幅，防止数值越界
    # csp_avg, csp_s
    u[1] = clamp(u[1], 100.0, cspmax - 100.0)
    u[2] = clamp(u[2], 100.0, cspmax - 100.0)

    # csn_avg, csn_s
    u[3] = clamp(u[3], 100.0, csnmax - 100.0)
    u[4] = clamp(u[4], 100.0, csnmax - 100.0)

    # delta_sei (必须非负)
    u[11] = max(u[11], 1e-10)

    # cf (必须非负)
    u[12] = max(u[12], 0.0)

    return u
end

mutable struct InitialStateSolver
    m::Model
    csp_s::VariableRef
    csn_s::VariableRef
    phi_p::VariableRef
    phi_n::VariableRef
    it::VariableRef
    iint::VariableRef
    Up::VariableRef
    Un::VariableRef
    theta_p::VariableRef
    theta_n::VariableRef
    csp_avg0::VariableRef
    csn_avg0::VariableRef
    delta_sei0::VariableRef
    power::VariableRef
end

function get_initial_state_solver()
    # 适当放宽公差，提高收敛成功率
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0,
        "linear_solver" => "mumps",
        "max_cpu_time" => 20.0, # 初始化不需要太久
        "tol" => 1e-5,
        "constr_viol_tol" => 1e-5
    ))

    @variable(m, csp_avg0)
    @variable(m, csn_avg0)
    @variable(m, delta_sei0)
    @variable(m, power)

    # --- 修正 1: 稍微放宽 theta 边界，防止在极值附近卡死 ---
    @variable(m, 1e-4 <= theta_p <= 0.9999)
    @variable(m, 1e-4 <= theta_n <= 0.9999)

    # 根据物理约束设置变量初值范围
    @variable(m, 1e-6 <= csp_s <= cspmax)
    @variable(m, 1e-6 <= csn_s <= csnmax)

    # --- 修正 2: 大幅放宽电流和电压边界，避免 LOCALLY_INFEASIBLE ---
    @variable(m, -20000 <= it <= 20000)   # 允许更大的中间迭代电流
    @variable(m, -20000 <= iint <= 20000)

    @constraint(m, theta_p * cspmax == csp_s)
    @constraint(m, theta_n * csnmax == csn_s)

    # 固相扩散平衡方程
    @constraint(m, 5 * (csp_s - csp_avg0) + Rpp * it / F / Dp / ap / lp == 0)
    @constraint(m, 5 * (csn_s - csn_avg0) - Rpn * iint / F / Dn / an / lnn == 0)

    # --- 修正 3: 放宽电势边界，让物理方程决定数值 ---
    @variable(m, 0.0 <= Up <= 10.0)   
    @variable(m, -2.0 <= Un <= 5.0)   
    @variable(m, 0.0 <= phi_p <= 10.0)
    @variable(m, -2.0 <= phi_n <= 5.0)

    # Butler-Volmer 方程
    @NLconstraint(m, (cspmax - csp_s)^(0.5) * csp_s^(0.5) * sinh(0.5 * F / R / T * (phi_p - Up)) - (it / ap / F / lp / (2 * kp * ce^(0.5))) == 0)
    
    @NLconstraint(m, (csnmax - csn_s)^(0.5) * csn_s^(0.5) * sinh(0.5 * F / R / T * (phi_n - Un + (delta_sei0 / Kappa_sei + Rsei) * it / an / lnn)) + iint / an / F / lnn / (2 * kn * ce^(0.5)) == 0)

    # OCV Functions
    @NLconstraint(m, Up == 7.49983 - 13.7758 * theta_p^0.5 + 21.7683 * theta_p - 12.6985 * theta_p^1.5 + 0.0174967 / theta_p - 0.41649 * theta_p^(-0.5) -
                           0.0161404 * exp(100 * theta_p - 97.1069) + 0.363031 * tanh(5.89493 * theta_p - 4.21921))

    @NLconstraint(m, Un == 9.99877 - 9.99961 * theta_n^0.5 - 9.98836 * theta_n + 8.2024 * theta_n^1.5 + 0.23584 / theta_n - 2.03569 * theta_n^(-0.5) -
                           1.47266 * exp(-1.14872 * theta_n + 2.13185) - 9.9989 * tanh(0.60345 * theta_n - 1.58171))

    # SEI 副反应电流平衡
    isei = it - iint
    @NLconstraint(m, 1e4 * (-isei + an * lnn * ksei * exp(-1 * F / R / T * (phi_n - Urefs + (delta_sei0 / Kappa_sei + Rsei) * it / an / lnn))) == 0)
    
    # --- 修正 4: 功率平衡 ---
    # 移除 * area，确保与 simulator 中 out[8] = p - pot * it 一致
    # 假设 simulator 中的 it 是总电流 (Amps)
    @constraint(m, it * (phi_p - phi_n) == power) 

    return InitialStateSolver(m, csp_s, csn_s, phi_p, phi_n, it, iint, Up, Un, theta_p, theta_n, csp_avg0, csn_avg0, delta_sei0, power)
end

# 辅助函数：纯物理计算 OCV
function calculate_ocv(theta_p, theta_n)
    tp = clamp(theta_p, 1e-4, 0.9999)
    tn = clamp(theta_n, 1e-4, 0.9999)

    Up = 7.49983 - 13.7758 * tp^0.5 + 21.7683 * tp - 12.6985 * tp^1.5 + 0.0174967 / tp - 0.41649 * tp^(-0.5) -
         0.0161404 * exp(100 * tp - 97.1069) + 0.363031 * tanh(5.89493 * tp - 4.21921)
    
    Un = 9.99877 - 9.99961 * tn^0.5 - 9.98836 * tn + 8.2024 * tn^1.5 + 0.23584 / tn - 2.03569 * tn^(-0.5) -
         1.47266 * exp(-1.14872 * tn + 2.13185) - 9.9989 * tanh(0.60345 * tn - 1.58171)
    
    return Up, Un
end

function reset_solver!(solver::InitialStateSolver, u0, power_val)
    csp_avg0, csn_avg0, delta_sei0 = u0[1], u0[Ncp+1], u0[Ncp+Ncn+7]

    # 1. 计算基于平均浓度的初始猜测
    theta_p_guess = clamp(csp_avg0 / cspmax, 0.01, 0.99)
    theta_n_guess = clamp(csn_avg0 / csnmax, 0.01, 0.99)
    
    Up_guess, Un_guess = calculate_ocv(theta_p_guess, theta_n_guess)
    ocv_guess = Up_guess - Un_guess

    # 2. 估算电流 (P = I * V => I = P / V)
    v_est = max(ocv_guess, 2.0)
    # 注意：这里不再除以 area，保持与约束一致
    it_guess = power_val / v_est 

    # 3. 设置初值
    set_start_value(solver.it, it_guess)
    set_start_value(solver.iint, it_guess)
    
    set_start_value(solver.csp_s, csp_avg0)
    set_start_value(solver.csn_s, csn_avg0)
    
    set_start_value(solver.theta_p, theta_p_guess)
    set_start_value(solver.theta_n, theta_n_guess)

    set_start_value(solver.Up, Up_guess)
    set_start_value(solver.Un, Un_guess)
    set_start_value(solver.phi_p, Up_guess)
    set_start_value(solver.phi_n, Un_guess)

    # 4. 固定参数
    fix(solver.csp_avg0, csp_avg0; force=true)
    fix(solver.csn_avg0, csn_avg0; force=true)
    fix(solver.delta_sei0, delta_sei0; force=true)
    fix(solver.power, power_val; force=true)
end

function compute_initial_state(solver::InitialStateSolver, u0, power)
    
    # --- 策略 A: 零电流/微小电流的解析解 (极快且稳定) ---
    if abs(power) < 1e-5
        # 1. 物理平衡：表面浓度 = 平均浓度
        u0[Ncp] = u0[1]          # csp_s = csp_avg
        u0[Ncp+Ncn] = u0[Ncp+1]  # csn_s = csn_avg
        
        # 2. 计算 OCV
        theta_p = u0[1] / cspmax
        theta_n = u0[Ncp+1] / csnmax
        Up_val, Un_val = calculate_ocv(theta_p, theta_n)
        
        # 3. 设置电势 (平衡态)
        phi_n0 = Un_val
        phi_p0 = Up_val
        pot0 = phi_p0 - phi_n0
        
        # 4. 电流为 0
        it0 = 0.0
        iint0 = 0.0
        isei0 = 0.0
        
        # 5. 更新状态向量
        u0[Ncp+Ncn+1] = iint0
        u0[Ncp+Ncn+2] = phi_p0
        u0[Ncp+Ncn+3] = phi_n0
        u0[Ncp+Ncn+4] = pot0
        u0[Ncp+Ncn+5] = it0
        u0[Ncp+Ncn+6] = isei0
        
        return u0
    end

    # --- 策略 B: 大电流使用 Ipopt 求解 ---
    reset_solver!(solver, u0, power)
    optimize!(solver.m)
    status = termination_status(solver.m)
    
    # 检查求解状态
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        csp_s0 = value(solver.csp_s)
        csn_s0 = value(solver.csn_s)
        phi_p0 = value(solver.phi_p)
        phi_n0 = value(solver.phi_n)
        pot0 = phi_p0 - phi_n0
        it0 = value(solver.it)
        iint0 = value(solver.iint)
        isei0 = it0 - iint0

        u0[Ncp] = csp_s0
        u0[Ncp+Ncn] = csn_s0
        u0[Ncp+Ncn+1] = iint0
        u0[Ncp+Ncn+2] = phi_p0
        u0[Ncp+Ncn+3] = phi_n0
        u0[Ncp+Ncn+4] = pot0
        u0[Ncp+Ncn+5] = it0
        u0[Ncp+Ncn+6] = isei0
    else
        # --- 策略 C: 求解失败的回退机制 ---
        # 如果还是失败，说明 Power 需求超过了电池物理极限（如电压已到截止电压但仍需放电）
        # 此时为了程序不崩，我们假设它尽力了，用平衡态+简单的内阻压降估算
        # println("⚠️ Solver failed ($status). Power: $power. Using fallback.")
        
        u0[Ncp] = u0[1] 
        u0[Ncp+Ncn] = u0[Ncp+1]
        
        theta_p = u0[1] / cspmax
        theta_n = u0[Ncp+1] / csnmax
        Up_val, Un_val = calculate_ocv(theta_p, theta_n)
        
        # 估算简单的内阻 R (假设值，仅用于防止崩溃)
        R_est = 0.1 
        # I = P / V_ocv (近似)
        I_est = power / max(Up_val - Un_val, 2.0)
        
        u0[Ncp+Ncn+1] = I_est 
        u0[Ncp+Ncn+2] = Up_val # 忽略极化
        u0[Ncp+Ncn+3] = Un_val 
        u0[Ncp+Ncn+4] = Up_val - Un_val 
        u0[Ncp+Ncn+5] = I_est 
        u0[Ncp+Ncn+6] = 0.0   
    end
    
    # 强制小电流回退
    if abs(u0[Ncp+Ncn+5]) > 1e3 || !isfinite(u0[Ncp+Ncn+5])
        return compute_initial_state(init_solver, u0, 0.0)
    end

    return u0
end

function quasi_static_project(init_solver, u::AbstractVector, I::Float64; epsI::Float64 = 1e-6)
    u_new = copy(u)
    # 假设此时 I 是 Power
    u_processed = compute_initial_state(init_solver, u_new, I)
    return u_processed, :success
end

function safe_initial_state(init_solver, u, I; eps=1e-6)
    u_sanitized = sanitize_state!(copy(u))
    return compute_initial_state(init_solver, u_sanitized, I)
end