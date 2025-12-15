using JuMP
using Ipopt

# 引入参数
include("setup.jl")

function build_mpc_model(u0_current, horizon_steps, price_forecast, demand_forecast)
    K = NUM_BATTERIES_IN_STATION
    dt = DT_STATE 
    
    # 稍微放宽电压边界，避免求解器卡在边界上导致 Infeasible
    V_min_soft = V_min - 0.05
    V_max_soft = V_max + 0.05
    
    # --- 辅助函数：计算 OCV (用于初始化猜测，防止数值爆炸) ---
    # 公式来源参考 bss_simulator.jl 中的定义
    function get_Up_value(theta)
        # 限制 theta 防止 log/pow 报错
        t = clamp(theta, 0.005, 0.995)
        return 7.49983 - 13.7758 * t^0.5 + 21.7683 * t - 12.6985 * t^1.5 + 
               0.0174967 / t - 0.41649 * t^(-0.5) - 
               0.0161404 * exp(100 * t - 97.1069) + 
               0.363031 * tanh(5.89493 * t - 4.21921)
    end

    function get_Un_value(theta)
        t = clamp(theta, 0.005, 0.995)
        return 9.99877 - 9.99961 * t^0.5 - 9.98836 * t + 8.2024 * t^1.5 + 
               0.23584 / t - 2.03569 * t^(-0.5) - 
               1.47266 * exp(-1.14872 * t + 2.13185) - 
               9.9989 * tanh(0.60345 * t - 1.58171)
    end

    # --- 求解器配置 ---
    # 关键：开启 nlp_scaling_method 以自动处理量级差异
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 5, 
        "max_cpu_time" => 120.0,      # 增加时间限制
        "max_iter" => 5000,           # 增加迭代次数
        "tol" => 1e-4,                # 适当放宽收敛容差
        "dual_inf_tol" => 1.0,        # 允许初始对偶误差较大
        "constr_viol_tol" => 1e-4,
        "accept_after_max_steps" => 10,
        "mu_strategy" => "adaptive",
        "nlp_scaling_method" => "gradient-based" # 【重要】基于梯度的自动缩放
        
    ))

    # --- 变量定义 ---
    @variable(model, P_nominal * -1 <= power[1:K, 1:horizon_steps] <= P_nominal)
    
    @variable(model, 100 <= csp_avg[1:K, 1:horizon_steps+1] <= cspmax)
    @variable(model, 100 <= csn_avg[1:K, 1:horizon_steps+1] <= csnmax)
    @variable(model, 1e-8 <= delta_sei[1:K, 1:horizon_steps+1] <= 1)
    @variable(model, 0 <= cf[1:K, 1:horizon_steps+1])

    # 代数变量
    @variable(model, 100 <= csp_s[1:K, 1:horizon_steps] <= cspmax)
    @variable(model, 100 <= csn_s[1:K, 1:horizon_steps] <= csnmax)
    
    # 避免 theta 接近 0 或 1 导致 log/pow 错误
    @variable(model, 0.005 <= theta_p[1:K, 1:horizon_steps] <= 0.995) 
    @variable(model, 0.005 <= theta_n[1:K, 1:horizon_steps] <= 0.995)
    
    @variable(model, 1.5 <= phi_p[1:K, 1:horizon_steps] <= 5.5)
    @variable(model, 0.0 <= phi_n[1:K, 1:horizon_steps] <= 3.5)
    @variable(model, 1.5 <= Up[1:K, 1:horizon_steps] <= 5.5)
    @variable(model, 0.0 <= Un[1:K, 1:horizon_steps] <= 3.5)
    
    # 限制电流范围，防止极大值
    @variable(model, -200 <= it[1:K, 1:horizon_steps] <= 200)
    @variable(model, -200 <= iint[1:K, 1:horizon_steps] <= 200)

    # --- 初始状态固定与 Warm Start ---
    for k in 1:K
        # 1. 固定 t=1 的状态变量
        fix(csp_avg[k, 1], u0_current[k, 1]; force=true)
        fix(csn_avg[k, 1], u0_current[k, Ncp+1]; force=true)
        fix(delta_sei[k, 1], u0_current[k, Ncp+Ncn+7]; force=true)
        fix(cf[k, 1], u0_current[k, Ncp+Ncn+Nsei+5]; force=true)
        
        # 2. 计算智能初值 (Warm Start)
        current_csp = u0_current[k, 1]
        current_csn = u0_current[k, Ncp+1]
        
        # 计算初始 SOC 对应的 OCV
        theta_p_init = current_csp / cspmax
        theta_n_init = current_csn / csnmax
        Up_init = get_Up_value(theta_p_init)
        Un_init = get_Un_value(theta_n_init)
        
        for t in 1:horizon_steps
            set_start_value(power[k, t], 0.0)
            
            # 浓度初始化为当前时刻值
            set_start_value(csp_avg[k, t+1], current_csp)
            set_start_value(csn_avg[k, t+1], current_csn)
            set_start_value(csp_s[k, t], current_csp)
            set_start_value(csn_s[k, t], current_csn)
            
            set_start_value(theta_p[k, t], theta_p_init)
            set_start_value(theta_n[k, t], theta_n_init)
            
            # 【核心修复】将电压初值设为计算出的 OCV
            # 这样初始时刻的过电势 (overpotential) 接近 0，sinh 项就不会爆炸
            set_start_value(Up[k, t], Up_init)
            set_start_value(Un[k, t], Un_init)
            set_start_value(phi_p[k, t], Up_init) 
            set_start_value(phi_n[k, t], Un_init)
            
            set_start_value(it[k, t], 0.0)
            set_start_value(iint[k, t], 0.0)
        end
    end

    # --- 约束构建 ---
    for t in 1:horizon_steps
        for k in 1:K
            # 1. 物理约束 (固相扩散) - 【进行归一化处理】
            # 原式：5 * (csp_s - csp_avg) + ... = 0
            # 修改：两边除以 cspmax，使数值量级在 1 左右
            @constraint(model, (5 * (csp_s[k,t] - csp_avg[k,t]) + Rpp * it[k,t] / F / Dp / ap / lp) / cspmax == 0)
            @constraint(model, (5 * (csn_s[k,t] - csn_avg[k,t]) - Rpn * iint[k,t] / F / Dn / an / lnn) / csnmax == 0)

            @constraint(model, theta_p[k,t] * cspmax == csp_s[k,t])
            @constraint(model, theta_n[k,t] * csnmax == csn_s[k,t])

            # 2. Butler-Volmer 动力学方程
            # 这部分保持原样，但由于 Warm Start 的存在，phi 和 U 的初值接近，不会发散
            @NLconstraint(model, (cspmax - csp_s[k,t])^(0.5) * csp_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_p[k,t] - Up[k,t])) - (it[k,t] / ap / F / lp / (2 * kp * ce^(0.5))) == 0)
            @NLconstraint(model, (csnmax - csn_s[k,t])^(0.5) * csn_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_n[k,t] - Un[k,t] + (delta_sei[k,t] / Kappa_sei + Rsei) * it[k,t] / an / lnn)) + iint[k,t] / an / F / lnn / (2 * kn * ce^(0.5)) == 0)

            # 3. OCV 方程
            @NLconstraint(model, Up[k,t] == 7.49983 - 13.7758 * theta_p[k,t]^0.5 + 21.7683 * theta_p[k,t] - 12.6985 * theta_p[k,t]^1.5 + 0.0174967 / theta_p[k,t] - 0.41649 * theta_p[k,t]^(-0.5) - 0.0161404 * exp(100 * theta_p[k,t] - 97.1069) + 0.363031 * tanh(5.89493 * theta_p[k,t] - 4.21921))
            @NLconstraint(model, Un[k,t] == 9.99877 - 9.99961 * theta_n[k,t]^0.5 - 9.98836 * theta_n[k,t] + 8.2024 * theta_n[k,t]^1.5 + 0.23584 / theta_n[k,t] - 2.03569 * theta_n[k,t]^(-0.5) - 1.47266 * exp(-1.14872 * theta_n[k,t] + 2.13185) - 9.9989 * tanh(0.60345 * theta_n[k,t] - 1.58171))

            # 4. SEI 副反应
            @NLconstraint(model, (it[k,t] - iint[k,t]) == an * lnn * ksei * exp(-1 * F / R / T * (phi_n[k,t] - Urefs + (delta_sei[k,t] / Kappa_sei + Rsei) * it[k,t] / an / lnn)))

            # 5. 功率平衡
            @constraint(model, it[k,t] * (phi_p[k,t] - phi_n[k,t]) * area == power[k,t])

            # 6. 状态更新 (Differential Equations) - 【进行归一化处理】
            # 将浓度变化的约束除以 cspmax/csnmax
            @constraint(model, (csp_avg[k,t+1] - csp_avg[k,t]) / dt / cspmax == (-15 * Dp / Rpp / Rpp * (csp_avg[k,t] - csp_s[k,t])) / cspmax)
            @constraint(model, (csn_avg[k,t+1] - csn_avg[k,t]) / dt / csnmax == (-15 * Dn / Rpn / Rpn * (csn_avg[k,t] - csn_s[k,t])) / csnmax)
            
            # SEI 厚度变化很小，可以乘一个系数放大残差，或者保持原样 (这里保持原样，因为 Ipopt 的 auto-scaling 会处理)
            @NLconstraint(model, (delta_sei[k,t+1] - delta_sei[k,t]) / dt == (it[k,t] - iint[k,t]) * M_sei / F / rho_sei / an / lnn)
            @constraint(model, (cf[k,t+1] - cf[k,t]) / dt == (it[k,t] - iint[k,t]) / 3600)

            # 7. 约束放宽
            @constraint(model, V_min_soft <= phi_p[k,t] - phi_n[k,t] <= V_max_soft)
            @constraint(model, soc_min <= csn_avg[k,t+1] / csnmax <= soc_max)
        end
    end

    # 目标函数
    # 增加惩罚项权重，引导求解器找到可行解
    revenue = @expression(model, sum(-1 * power[k,t] * price_forecast[t] for k in 1:K, t in 1:horizon_steps))
    soc_tracking = @expression(model, sum((csn_avg[k, t] / csnmax - 0.5)^2 for k in 1:K, t in 1:horizon_steps))
    
    @objective(model, Max, revenue - 0.1 * soc_tracking)

    return model, power
end