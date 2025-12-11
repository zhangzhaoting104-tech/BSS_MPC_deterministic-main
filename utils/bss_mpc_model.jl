using JuMP
using Ipopt

# 引入参数
include("setup.jl")

function build_mpc_model(u0_current, horizon_steps, price_forecast, demand_forecast)
    K = NUM_BATTERIES_IN_STATION
    dt = DT_STATE 
    
    # 稍微放宽电压边界
    V_min_soft = V_min - 0.1
    V_max_soft = V_max + 0.1
    
    # --- OCV 辅助函数 ---
    function get_Up_value(theta)
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
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 5, 
        "max_cpu_time" => 60.0,
        "max_iter" => 3000,
        "tol" => 1e-4,              
        "dual_inf_tol" => 1e-2,       # 进一步放宽对偶可行性容差
        "constr_viol_tol" => 1e-4,    # 维持约束残差容忍度
        "acceptable_tol" => 1e-3,     # 设定可接受的最终容差
        "accept_after_max_steps" => 10,
        "mu_strategy" => "adaptive",
        "nlp_scaling_method" => "gradient-based" 
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
    
    @variable(model, 0.001 <= theta_p[1:K, 1:horizon_steps] <= 0.999) 
    @variable(model, 0.001 <= theta_n[1:K, 1:horizon_steps] <= 0.999)
    
    @variable(model, 1.5 <= phi_p[1:K, 1:horizon_steps] <= 5.5)
    @variable(model, 0.0 <= phi_n[1:K, 1:horizon_steps] <= 3.5)
    @variable(model, 1.5 <= Up[1:K, 1:horizon_steps] <= 5.5)
    @variable(model, 0.0 <= Un[1:K, 1:horizon_steps] <= 3.5)
    
    @variable(model, -200 <= it[1:K, 1:horizon_steps] <= 200)
    @variable(model, -200 <= iint[1:K, 1:horizon_steps] <= 200)

    # 【新增】松弛变量 (Slack Variables)
    # 松弛变量必须是非负的
    @variable(model, slack_bv_p[1:K, 1:horizon_steps] >= 0)
    @variable(model, slack_bv_n[1:K, 1:horizon_steps] >= 0)
    @variable(model, slack_power[1:K, 1:horizon_steps] >= 0)
    
    # --- 初始状态处理 (Relaxed Initialization) ---
    for k in 1:K
        # 清洗输入数据，防止超出物理极限
        safe_csp = clamp(u0_current[k, 1], 105.0, cspmax - 5.0)
        safe_csn = clamp(u0_current[k, Ncp+1], 105.0, csnmax - 5.0)
        safe_sei = max(u0_current[k, Ncp+Ncn+7], 1e-8)
        safe_cf = max(u0_current[k, Ncp+Ncn+Nsei+5], 0.0)

        # 初始状态的软约束 (±0.5% tolerance)
        margin = 0.005 
        
        @constraint(model, safe_csp * (1 - margin) <= csp_avg[k, 1] <= safe_csp * (1 + margin))
        @constraint(model, safe_csn * (1 - margin) <= csn_avg[k, 1] <= safe_csn * (1 + margin))
        
        # 对于 SEI 和 CF，可以直接固定，精度影响小
        fix(delta_sei[k, 1], safe_sei; force=true)
        fix(cf[k, 1], safe_cf; force=true)
        
        # Warm Start
        theta_p_init = safe_csp / cspmax
        theta_n_init = safe_csn / csnmax
        Up_init = get_Up_value(theta_p_init)
        Un_init = get_Un_value(theta_n_init)
        
        for t in 1:horizon_steps
            set_start_value(power[k, t], 0.0)
            set_start_value(csp_avg[k, t+1], safe_csp)
            set_start_value(csn_avg[k, t+1], safe_csn)
            set_start_value(csp_s[k, t], safe_csp)
            set_start_value(csn_s[k, t], safe_csn)
            set_start_value(theta_p[k, t], theta_p_init)
            set_start_value(theta_n[k, t], theta_n_init)
            set_start_value(Up[k, t], Up_init)
            set_start_value(Un[k, t], Un_init)
            set_start_value(phi_p[k, t], Up_init) 
            set_start_value(phi_n[k, t], Un_init)
            set_start_value(it[k, t], 0.0)
            
            # 松弛变量初值设为 0
            set_start_value(slack_bv_p[k, t], 0.0)
            set_start_value(slack_bv_n[k, t], 0.0)
            set_start_value(slack_power[k, t], 0.0)
        end
    end

    # --- 约束构建 ---
    for t in 1:horizon_steps
        for k in 1:K
            # 1. 物理约束 (固相扩散) - 保持硬约束 (归一化)
            @constraint(model, (5 * (csp_s[k,t] - csp_avg[k,t]) + Rpp * it[k,t] / F / Dp / ap / lp) / cspmax == 0)
            @constraint(model, (5 * (csn_s[k,t] - csn_avg[k,t]) - Rpn * iint[k,t] / F / Dn / an / lnn) / csnmax == 0)

            @constraint(model, theta_p[k,t] * cspmax == csp_s[k,t])
            @constraint(model, theta_n[k,t] * csnmax == csn_s[k,t])
            
            # 2. Butler-Volmer 动力学方程 - 【转化为软约束】
            # 使用 slack_bv_p 吸收残差
            @NLconstraint(model, (cspmax - csp_s[k,t])^(0.5) * csp_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_p[k,t] - Up[k,t])) - (it[k,t] / ap / F / lp / (2 * kp * ce^(0.5))) <= slack_bv_p[k,t])
            @NLconstraint(model, (cspmax - csp_s[k,t])^(0.5) * csp_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_p[k,t] - Up[k,t])) - (it[k,t] / ap / F / lp / (2 * kp * ce^(0.5))) >= -slack_bv_p[k,t])

            # 使用 slack_bv_n 吸收残差
            @NLconstraint(model, (csnmax - csn_s[k,t])^(0.5) * csn_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_n[k,t] - Un[k,t] + (delta_sei[k,t] / Kappa_sei + Rsei) * iint[k,t] / an / lnn)) + iint[k,t] / an / F / lnn / (2 * kn * ce^(0.5)) <= slack_bv_n[k,t])
            @NLconstraint(model, (csnmax - csn_s[k,t])^(0.5) * csn_s[k,t]^(0.5) * sinh(0.5 * F / R / T * (phi_n[k,t] - Un[k,t] + (delta_sei[k,t] / Kappa_sei + Rsei) * iint[k,t] / an / lnn)) + iint[k,t] / an / F / lnn / (2 * kn * ce^(0.5)) >= -slack_bv_n[k,t])

            # 3. OCV 和 SEI
            @NLconstraint(model, Up[k,t] == 7.49983 - 13.7758 * theta_p[k,t]^0.5 + 21.7683 * theta_p[k,t] - 12.6985 * theta_p[k,t]^1.5 + 0.0174967 / theta_p[k,t] - 0.41649 * theta_p[k,t]^(-0.5) - 0.0161404 * exp(100 * theta_p[k,t] - 97.1069) + 0.363031 * tanh(5.89493 * theta_p[k,t] - 4.21921))
            @NLconstraint(model, Un[k,t] == 9.99877 - 9.99961 * theta_n[k,t]^0.5 - 9.98836 * theta_n[k,t] + 8.2024 * theta_n[k,t]^1.5 + 0.23584 / theta_n[k,t] - 2.03569 * theta_n[k,t]^(-0.5) - 1.47266 * exp(-1.14872 * theta_n[k,t] + 2.13185) - 9.9989 * tanh(0.60345 * theta_n[k,t] - 1.58171))
            @NLconstraint(model, (it[k,t] - iint[k,t]) == an * lnn * ksei * exp(-1 * F / R / T * (phi_n[k,t] - Urefs + (delta_sei[k,t] / Kappa_sei + Rsei) * iint[k,t] / an / lnn))) # 注意这里是 iint

            # 4. 功率平衡 - 【转化为软约束】
            # 使用 slack_power 吸收残差
            @constraint(model, it[k,t] * (phi_p[k,t] - phi_n[k,t]) * area - power[k,t] <= slack_power[k,t])
            @constraint(model, it[k,t] * (phi_p[k,t] - phi_n[k,t]) * area - power[k,t] >= -slack_power[k,t])

            # 5. 状态更新 (归一化) - 保持硬约束
            @constraint(model, (csp_avg[k,t+1] - csp_avg[k,t]) / dt / cspmax == (-15 * Dp / Rpp / Rpp * (csp_avg[k,t] - csp_s[k,t])) / cspmax)
            @constraint(model, (csn_avg[k,t+1] - csn_avg[k,t]) / dt / csnmax == (-15 * Dn / Rpn / Rpn * (csn_avg[k,t] - csn_s[k,t])) / csnmax)
            
            @NLconstraint(model, (delta_sei[k,t+1] - delta_sei[k,t]) / dt == (it[k,t] - iint[k,t]) * M_sei / F / rho_sei / an / lnn)
            @constraint(model, (cf[k,t+1] - cf[k,t]) / dt == (it[k,t] - iint[k,t]) / 3600)

            # 6. 运行约束
            @constraint(model, V_min_soft <= phi_p[k,t] - phi_n[k,t] <= V_max_soft)
            @constraint(model, soc_min <= csn_avg[k,t+1] / csnmax <= soc_max)
        end
    end

    # 目标函数
    # 【新增】对松弛变量施加高额惩罚（1e6），强制求解器最小化松弛，以保证约束接近满足
    # 惩罚系数 $10^6$ 远大于收益和 SOC 跟踪项，确保可行性优先
    P_slack = 1e6 
    slack_penalty = @expression(model, P_slack * (sum(slack_bv_p) + sum(slack_bv_n) + sum(slack_power)))
    
    revenue = @expression(model, sum(-1 * power[k,t] * price_forecast[t] for k in 1:K, t in 1:horizon_steps))
    soc_tracking = @expression(model, sum((csn_avg[k, t] / csnmax - 0.5)^2 for k in 1:K, t in 1:horizon_steps))
    
    # 目标函数 = 最大化 (收益) - 最小化 (SOC 偏差) - 最小化 (松弛变量)
    @objective(model, Max, revenue - 0.01 * soc_tracking - slack_penalty)

    return model, power
end