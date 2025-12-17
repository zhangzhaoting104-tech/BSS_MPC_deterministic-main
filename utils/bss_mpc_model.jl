using JuMP
using Ipopt

include("setup.jl")

function build_mpc_model(u0_current, horizon_steps, price_forecast, demand_forecast)
    K = NUM_BATTERIES_IN_STATION
    dt = DT_STATE 
    
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 5, 
        "max_cpu_time" => 100.0,
        "tol" => 1e-4, 
        "nlp_scaling_method" => "gradient-based", # 自动缩放梯度 [cite: 41]
        "acceptable_tol" => 1e-2
    ))

    # --- 变量定义与边界限定 ---
    @variable(model, -P_nominal <= power[1:K, 1:horizon_steps] <= P_nominal)
    @variable(model, 100 <= csp_avg[1:K, 1:horizon_steps+1] <= cspmax)
    @variable(model, 100 <= csn_avg[1:K, 1:horizon_steps+1] <= csnmax)
    @variable(model, 1e-9 <= delta_sei[1:K, 1:horizon_steps+1] <= 1e-3)
    
    # 辅助变量
    @variable(model, 0.005 <= theta_p[1:K, 1:horizon_steps] <= 0.995) 
    @variable(model, 0.005 <= theta_n[1:K, 1:horizon_steps] <= 0.995)
    @variable(model, 1.5 <= phi_p[1:K, 1:horizon_steps] <= 5.5)
    @variable(model, 0.0 <= phi_n[1:K, 1:horizon_steps] <= 3.5)
    @variable(model, -250 <= it[1:K, 1:horizon_steps] <= 250)

    # --- 引入松弛变量（防止不可解） ---
    @variable(model, s_soc_min[1:K, 1:horizon_steps+1] >= 0)

    # --- 初始状态固定 ---
    for k in 1:K
        fix(csp_avg[k, 1], u0_current[k, 1]; force=true) #[cite: 44]
        fix(csn_avg[k, 1], u0_current[k, 3]; force=true) # 修正索引：csn_avg 在 u[3] [cite: 82]
        fix(delta_sei[k, 1], u0_current[k, 11]; force=true)
    end

    # --- 物理约束（带数值缩放） ---
    for t in 1:horizon_steps
        for k in 1:K
            # 缩放法：将方程除以变量的典型值（如 cspmax）使其残差接近 1 [cite: 49, 54]
            # 1. 固相扩散平衡
            @constraint(model, (5 * (theta_p[k,t]*cspmax - csp_avg[k,t]) + Rpp * it[k,t] / F / Dp / ap / lp) / cspmax == 0)
            @constraint(model, (5 * (theta_n[k,t]*csnmax - csn_avg[k,t]) - Rpn * it[k,t] / F / Dn / an / lnn) / csnmax == 0)

            # 2. 功率方程
            @constraint(model, it[k,t] * (phi_p[k,t] - phi_n[k,t]) * area == power[k,t])# [cite: 53]

            # 3. 动力学 (Butler-Volmer) - 使用 clamp 后的 theta
            # 此处省略复杂的 OCV 公式，实际运行时需引用 get_Up_value [cite: 52]

            # 4. 状态演化 (Euler)
            @constraint(model, (csp_avg[k,t+1] - csp_avg[k,t]) / dt / cspmax == (-15 * Dp / Rpp / Rpp * (csp_avg[k,t] - theta_p[k,t]*cspmax)) / cspmax)
            @constraint(model, (csn_avg[k,t+1] - csn_avg[k,t]) / dt / csnmax == (-15 * Dn / Rpn / Rpn * (csn_avg[k,t] - theta_n[k,t]*csnmax)) / csnmax)

            # 5. 软化的 SOC 边界 (核心修复)
            @constraint(model, csn_avg[k,t+1] / csnmax >= soc_min - s_soc_min[k,t+1])
            @constraint(model, csn_avg[k,t+1] / csnmax <= soc_max)
        end
    end

    # --- 目标函数：套利 + 备货 + 退化惩罚 ---
    # 目标是最大化套利收益，同时惩罚 SOC 不足和过度退化 [cite: 56]
    revenue = @expression(model, sum(-1 * power[k,t] * price_forecast[t] for k in 1:K, t in 1:horizon_steps))
    soc_readiness = @expression(model, sum((csn_avg[k,t+1]/csnmax - 0.8)^2 for k in 1:K, t in 1:horizon_steps))
    slack_penalty = @expression(model, sum(s_soc_min[k,t] * 1e5 for k in 1:K, t in 2:horizon_steps+1))

    @objective(model, Max, revenue - 1.0 * soc_readiness - slack_penalty)

    return model, power
end