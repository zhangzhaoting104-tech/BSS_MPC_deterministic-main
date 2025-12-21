using JuMP
using Ipopt

# 引入参数设置
include("setup.jl")

function build_mpc_model(u0_current, horizon_steps, price_forecast, demand_forecast)
    K = NUM_BATTERIES_IN_STATION
    dt = DT_STATE 
    
    # 1. 调整求解器参数：增加容差，允许轻微违背物理方程以获得解
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0,              # 稍微减少输出，但保留关键信息
        "max_cpu_time" => 20.0,          # MPC 需要快速，20s 没解出来通常就是卡死
        "tol" => 1e-3,                   # 放宽收敛容差
        "dual_inf_tol" => 1e-1,          # 放宽对偶容差（应对 Scaling 问题）
        "constr_viol_tol" => 1e-3,       # 允许 0.001 的约束违背
        "nlp_scaling_method" => "gradient-based",
        "mu_strategy" => "adaptive"      # 自适应更新策略通常更稳健
    ))

    # ==========================================
    # 2. 变量定义 (放宽非物理硬边界)
    # ==========================================
    # 移除 power 的硬边界，改用软惩罚或仅在约束中限制，防止初始化失败
    @variable(model, -P_nominal * 1.5 <= power[1:K, 1:horizon_steps] <= P_nominal * 1.5)
    
    # 放宽电流边界，允许瞬间大电流（数值上）以便寻找可行点
    @variable(model, -500 <= it[1:K, 1:horizon_steps] <= 500)
    @variable(model, -500 <= iint[1:K, 1:horizon_steps] <= 500)
    @variable(model, 0 <= isei[1:K, 1:horizon_steps] <= 20)

    # 浓度
    @variable(model, 10 <= csp_avg[1:K, 1:horizon_steps+1] <= cspmax + 1000)
    @variable(model, 10 <= csn_avg[1:K, 1:horizon_steps+1] <= csnmax + 1000)
    @variable(model, 10 <= csp_s[1:K, 1:horizon_steps] <= cspmax + 1000)
    @variable(model, 10 <= csn_s[1:K, 1:horizon_steps] <= csnmax + 1000)

    # 这里的边界要宽，避免 OCV 计算时卡在边界上
    @variable(model, 0.001 <= theta_p[1:K, 1:horizon_steps] <= 0.999)
    @variable(model, 0.001 <= theta_n[1:K, 1:horizon_steps] <= 0.999)

    # 电势：大幅放宽，让物理方程决定数值，而不是人为卡死
    @variable(model, 0.0 <= Up[1:K, 1:horizon_steps] <= 6.0)
    @variable(model, -1.0 <= Un[1:K, 1:horizon_steps] <= 3.0)
    @variable(model, 0.0 <= phi_p[1:K, 1:horizon_steps] <= 6.0)
    @variable(model, -1.0 <= phi_n[1:K, 1:horizon_steps] <= 3.0)

    # 其他变量保持不变
    @variable(model, 1e-10 <= delta_sei[1:K, 1:horizon_steps+1] <= 1e-3)
    @variable(model, 0 <= cf[1:K, 1:horizon_steps+1] <= Qmax)
    @variable(model, s_soc_min[1:K, 1:horizon_steps+1] >= 0)
    # 1. 增加需求缺口松弛变量 (Demand Slack)
    @variable(model, s_demand[1:horizon_steps] >= 0)
    
    # ==========================================
    # 3. 初始状态固定 & Warm Start (保留你之前添加的代码)
    # ==========================================
    # ... [请保留之前给出的 Warm Start 代码块] ...
    for k in 1:K
        # 提取当前时刻的物理量
        c_p_init = u0_current[k, 1]
        c_n_init = u0_current[k, 3]
        sei_init = u0_current[k, 11]
        
        # 计算对应的 Theta 和 OCV (使用与约束一致的公式估算)
        theta_p_init = clamp(c_p_init / cspmax, 0.01, 0.99)
        theta_n_init = clamp(c_n_init / csnmax, 0.01, 0.99)
        
        # 简化的 OCV 估算 (仅用于初值，不需要像约束那么极其精确，但要在数量级上正确)
        # 这里手动写出核心项，或者如果有 calculate_ocv 函数可直接调用
        Up_init = 4.0 # 锂电池正极典型值
        Un_init = 0.5 # 锂电池负极典型值
        
        for t in 1:horizon_steps
            # 1. 浓度与 SOC 初始化
            set_start_value(csp_avg[k, t+1], c_p_init)
            set_start_value(csn_avg[k, t+1], c_n_init)
            set_start_value(csp_s[k, t], c_p_init)
            set_start_value(csn_s[k, t], c_n_init)
            set_start_value(theta_p[k, t], theta_p_init)
            set_start_value(theta_n[k, t], theta_n_init)
            
            # 2. 电势初始化 (非常重要，防止 sinh 爆炸)
            set_start_value(Up[k, t], Up_init)
            set_start_value(Un[k, t], Un_init)
            set_start_value(phi_p[k, t], Up_init) # 假设开路，固相电势约等于 OCV
            set_start_value(phi_n[k, t], Un_init)
            
            # 3. 电流与功率初始化 (假设初始为静置状态)
            set_start_value(it[k, t], 0.0)
            set_start_value(iint[k, t], 0.0)
            set_start_value(isei[k, t], 1e-10) # 避免完全为0导致某些梯度奇异
            set_start_value(power[k, t], 0.0)
            
            # 4. 健康状态初始化
            set_start_value(delta_sei[k, t+1], sei_init)
        end
    end
    # 必须保留！否则 OCV 函数会报错。
    # ----------------------------------------------------
     for k in 1:K
        fix(csp_avg[k, 1], u0_current[k, 1]; force=true)
        fix(csn_avg[k, 1], u0_current[k, 3]; force=true)
        fix(delta_sei[k, 1], u0_current[k, 11]; force=true)
        fix(cf[k, 1], u0_current[k, 12]; force=true)
        
        # 简易 Warm Start (重要)
        c_p_init = u0_current[k, 1]
        c_n_init = u0_current[k, 3]
        theta_p_init = clamp(c_p_init / cspmax, 0.01, 0.99)
        theta_n_init = clamp(c_n_init / csnmax, 0.01, 0.99)
        
        for t in 1:horizon_steps
            set_start_value(csp_avg[k, t+1], c_p_init)
            set_start_value(csn_avg[k, t+1], c_n_init)
            set_start_value(theta_p[k, t], theta_p_init)
            set_start_value(theta_n[k, t], theta_n_init)
            set_start_value(Up[k, t], 4.0) 
            set_start_value(Un[k, t], 0.5)
            set_start_value(phi_p[k, t], 4.0)
            set_start_value(phi_n[k, t], 0.5)
            set_start_value(it[k, t], 0.0)
        end
    end

    # ==========================================
    # 4. 物理约束 (关键：手动 Scaling)
    # ==========================================
    # 比例因子：用于将所有约束的残差归一化到 1.0 附近
    S_conc = 1.0 / cspmax         # 浓度约 1e4 -> 1.0
    S_curr = 1.0 / 100.0          # 电流约 1e2 -> 1.0
    S_pot  = 1.0                  # 电势约 1e0 -> 1.0 (无需大改)
    S_BV   = 1e-4                 # Butler-Volmer 左边项很大，需要大幅缩小

    for t in 1:horizon_steps
        for k in 1:K
            # A. 几何
            @constraint(model, theta_p[k,t] == csp_s[k,t] / cspmax)
            @constraint(model, theta_n[k,t] == csn_s[k,t] / csnmax)

            # B. 固相扩散 (Scaling: 除以 cspmax)
            # 原式: 5*(cs - cavg) + ... = 0
            @constraint(model, (5 * (csp_s[k,t] - csp_avg[k,t]) + (Rpp * it[k,t]) / (F * Dp * ap * lp * area)) * S_conc == 0)
            @constraint(model, (5 * (csn_s[k,t] - csn_avg[k,t]) - (Rpn * iint[k,t]) / (F * Dn * an * lnn * area)) * S_conc == 0)

            # C. OCV (无需 Scaling，本身在 1-5V 之间)
            @NLconstraint(model, Up[k,t] == 7.49983 - 13.7758 * theta_p[k,t]^0.5 + 21.7683 * theta_p[k,t] - 
                                12.6985 * theta_p[k,t]^1.5 + 0.0174967 / theta_p[k,t] - 0.41649 * theta_p[k,t]^(-0.5) -
                                0.0161404 * exp(100 * theta_p[k,t] - 97.1069) + 0.363031 * tanh(5.89493 * theta_p[k,t] - 4.21921))

            @NLconstraint(model, Un[k,t] == 9.99877 - 9.99961 * theta_n[k,t]^0.5 - 9.98836 * theta_n[k,t] + 
                                8.2024 * theta_n[k,t]^1.5 + 0.23584 / theta_n[k,t] - 2.03569 * theta_n[k,t]^(-0.5) -
                                1.47266 * exp(-1.14872 * theta_n[k,t] + 2.13185) - 9.9989 * tanh(0.60345 * theta_n[k,t] - 1.58171))

            # D. Butler-Volmer (关键：Scaling)
            # 原始项大概在 1e4 到 1e7 之间。我们乘上 S_BV (1e-4) 将其拉回 1.0 附近。
            # 系数 calculation: 1/(area*ap*F*lp*2*kp*ce^0.5) 大约是 1e4。IT 大约是 100。乘积是 1e6。
            # 因此这里除以 1e5 或者 1e6 是合适的。这里使用 S_BV = 1e-5。
            
            S_BV_p = 1e-5
            @NLconstraint(model, ( (cspmax - csp_s[k,t])^0.5 * csp_s[k,t]^0.5 * sinh(0.5 * F / R / T * (phi_p[k,t] - Up[k,t])) 
                                - (it[k,t] / (area * ap * F * lp * (2 * kp * ce^0.5))) ) * S_BV_p == 0)

            S_BV_n = 1e-5
            @NLconstraint(model, ( (csnmax - csn_s[k,t])^0.5 * csn_s[k,t]^0.5 * sinh(0.5 * F / R / T * (phi_n[k,t] - Un[k,t] + (delta_sei[k,t] / Kappa_sei + Rsei) * it[k,t] / (area * an * lnn))) 
                                + iint[k,t] / (area * an * F * lnn * (2 * kn * ce^0.5)) ) * S_BV_n == 0)

            # E. SEI 副反应 (Scaling)
            @constraint(model, (it[k,t] - iint[k,t] - isei[k,t]) * S_curr == 0)
            
            # Tafel equation scaling
            @NLconstraint(model, (isei[k,t] - (area * an * lnn * ksei * exp(-1.0 * F / R / T * (phi_n[k,t] - Urefs + (delta_sei[k,t] / Kappa_sei + Rsei) * it[k,t] / (area * an * lnn)))) ) * 1e5 == 0) # ksei 很小，所以这里可能需要乘大数或者除小数

            # F. 状态演化 (Scaling: 除以 cspmax)
            @constraint(model, (csp_avg[k,t+1] - csp_avg[k,t] - dt * (-15 * Dp / Rpp^2 * (csp_avg[k,t] - csp_s[k,t]))) * S_conc == 0)
            @constraint(model, (csn_avg[k,t+1] - csn_avg[k,t] - dt * (-15 * Dn / Rpn^2 * (csn_avg[k,t] - csn_s[k,t]))) * S_conc == 0)
            
            # SEI 增长 (很慢，系数小)
            @constraint(model, (delta_sei[k,t+1] - delta_sei[k,t] - dt * (isei[k,t] * M_sei / (F * rho_sei * an * lnn * area))) * 1e6 == 0)
            
            # 容量衰减
            @constraint(model, (cf[k,t+1] - cf[k,t] - dt * (isei[k,t] / 3600.0)) * 1e4 == 0)

            # G. 功率
            # power ~ 3000, it ~ 100, pot ~ 3. 
            @constraint(model, (power[k,t] - it[k,t] * (phi_p[k,t] - phi_n[k,t])) * 1e-3 == 0)

            # H. SOC
            @constraint(model, csn_avg[k,t+1] / csnmax >= soc_min - s_soc_min[k,t+1])
            @constraint(model, csn_avg[k,t+1] / csnmax <= soc_max)
            @constraint(model, power[k,t] <= P_nominal) # 只加软一点的约束
            @constraint(model, power[k,t] >= -P_nominal)
        end
            #I. 换电需求
            # 2. 核心约束：站内总能量必须足以支撑未来的换电需求
            # 我们用总电量来近似：所有电池的总 SOC 必须大于 需求数 * 0.7
            # 加上松弛变量 s_demand，如果达不到就扣分
             @constraint(model, sum(csn_avg[k, t] / csnmax for k in 1:K) >= 
                          demand_forecast[t] * replaceable_soc - s_demand[t])
        
    end
    # ==========================================
    # 5. 目标函数 (大幅缩放)
    # ==========================================
    # 将目标函数值域控制在 1e-1 到 1e3 之间
    obj_scale = 1e-2 

    revenue = @expression(model, sum(-1 * power[k,t] * price_forecast[t] for k in 1:K, t in 1:horizon_steps))
    soc_readiness = @expression(model, sum((csn_avg[k,t+1]/csnmax - 0.8)^2 for k in 1:K, t in 1:horizon_steps))
    
    # 减小退化惩罚的权重，原先的 5000 太大了，容易淹没主目标
    degradation_cost = @expression(model, sum((cf[k,t+1] - cf[k,t]) for k in 1:K, t in 1:horizon_steps)) 
    
    slack_penalty = @expression(model, sum(s_soc_min[k,t] for k in 1:K, t in 2:horizon_steps+1))

    # 调整后的权重
    w_soc = 10.0
    w_deg = 5e5   # 降低权重，先求通再求精
    w_slack = 1e10
    w_demand =2e4

    @objective(model, Max, (revenue - w_soc * soc_readiness - w_deg * degradation_cost - w_slack * slack_penalty- w_demand * sum(s_demand)) * obj_scale)

    return model, power
end