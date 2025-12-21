# run_mpc_main.jl
using Dates
using Printf
using Statistics
using LinearAlgebra
using JLD

# 引入所有模块 
include("setup.jl")
include("bss_simulator.jl")
include("bss_manager.jl")
include("bss_initial_state.jl")
include("bss_mpc_model.jl") 

# --- 配置参数 ---
SIM_DAYS = 1
TOTAL_HOURS = 12 * SIM_DAYS # 模拟总时长
HORIZON_H = 24              # 预测视界 (小时)
DT_SIM = 3600.0            # 模拟步长 (单位：秒）

function run_rolling_mpc()
    println("=== 启动 BSS 滚动时域控制系统 (Fixed 24h Horizon) ===")
    
    manager = create_bss_manager(NUM_BATTERIES_IN_STATION, NUM_BATTERIES_IN_AREA)
    u0_current = get_initial_states(manager) 
    init_solver = get_initial_state_solver()
    
    history_profit = 0.0
   
    for current_step in 1:TOTAL_HOURS
        println("\n--- Step $current_step / $TOTAL_HOURS ---")
        
        # 1. 获取预测数据
        idx_start = current_step
        idx_end = current_step + HORIZON_H - 1
        
        # 简单的边界处理，防止索引越界 (如果数据不够长)
        if idx_end > length(Prices)
            idx_end = length(Prices)
            # 补齐长度或者缩短预测域，这里假设 setup.jl 已经处理好数据长度
        end

        price_forecast = Prices[idx_start:idx_end]
        demand_forecast = swap_counts_data[idx_start:idx_end]
        
        # 2. MPC 决策
        mpc_model, power_var = build_mpc_model(
            u0_current, 
            HORIZON_H, 
            price_forecast, 
            demand_forecast
        )
        
        optimize!(mpc_model)
        
        # 3. 提取决策
        optimal_p = zeros(NUM_BATTERIES_IN_STATION)
        if termination_status(mpc_model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
            optimal_p = value.(power_var[:, 1])
        else
            @warn "Step $current_step: MPC 未收敛，执行安全策略"
        end
        
        # 简单的充放电保护
        optimal_p = clamp.(optimal_p, -P_max, P_max)
        for k in 1:NUM_BATTERIES_IN_STATION
            soc = u0_current[k,3] / csnmax
            # 强制保护：SOC过低强制充电，SOC过高禁止充电
            if soc < 0.15
                optimal_p[k] = P_max 
            elseif soc > 0.95
                optimal_p[k] = min(0.0, optimal_p[k])
            end
        end

        # 4. 执行物理模拟
        sim_instance = Simulator(u0_current[1, :], 0.0)
        # === 新增步骤：预平衡 ===
        # 在送入 DAE 之前，先用代数求解器计算出与 optimal_p 匹配的初始电压/电流
        # 这样 DAE 求解器在 t=0 时刻看到的残差就是 0，极大地减少 IDAICFailFlag
        for k in 1:NUM_BATTERIES_IN_STATION
             u0_current[k, :] = safe_initial_state(init_solver, u0_current[k, :], optimal_p[k])
        end
        u_next, profit, p_real, _, _, _ =
        simulate(
            sim_instance,
            init_solver,
            u0_current,
            price_forecast[1],
            optimal_p,
            Int[],       # 模拟阶段不传入换电索引，换电在模拟后手动处理
            zeros(0,4)
        )

        history_profit += profit
        
        # 5. === 修改后的换电逻辑 ===
        demand = Int(swap_counts_data[current_step])
        
        if demand > 0
            # 找到所有 SOC >= 0.7 的电池
            soc_list = u_next[:, 3] ./ csnmax
            eligible_idx = findall(x -> x >= replaceable_soc, soc_list)
            
            # 实际能换的数量
            num_swap = min(length(eligible_idx), demand)
            
            if num_swap > 0
                # 优先换出 SOC 最高的电池
                swap_indices = sortperm(soc_list, rev=true)[1:num_swap]
                
                # --- 定义换入电池的固定状态 (Scalar) ---
                soc_in = 0.25
                delta_sei_in = 5.0e-9  # 合理常数：代表有一定老化的 SEI 膜厚度
                cf_in = 1.0e-4         # 合理常数：代表有一定容量衰减
                
                # 根据 SOC 计算物理浓度
                # 负极平均浓度
                csn_avg_new = csnmax * soc_in
                # 正极平均浓度 (根据锂离子守恒计算)
                # 公式参考 setup.jl 中的参数: csp_avg = cspmax - (csn_avg * Ln * en * area) / (Lp * ep * area)
                csp_avg_new = cspmax * 1.0 - csnmax * soc_in * lnn * en / lp / ep
                
                for idx in swap_indices
                    # 1. 重置微分状态变量
                    u_next[idx, 1] = csp_avg_new  # csp_avg
                    u_next[idx, 2] = csp_avg_new  # csp_s (平衡态：表面=平均)
                    u_next[idx, 3] = csn_avg_new  # csn_avg
                    u_next[idx, 4] = csn_avg_new  # csn_s (平衡态：表面=平均)
                    
                    # 2. 重置健康状态
                    u_next[idx, 11] = delta_sei_in # delta_sei
                    u_next[idx, 12] = cf_in        # cf
                    
                    # 3. 清零代数变量 (电流、电势差等)
                    u_next[idx, 5:10] .= 0.0 
                    
                    # 4. 如果有额外的状态位 (如 Simulator 中定义的 differential_vars 之外的)，也建议清零
                    if size(u_next, 2) > 12
                        u_next[idx, 13:end] .= 0.0
                    end

                    # 5. 关键：调用求解器计算平衡态的电势 (U, phi)
                    # 这一步会根据新的浓度计算 OCV，并填入 u_next 对应的电势位置
                    u_next[idx, :] = compute_initial_state(init_solver, u_next[idx, :], 0.0)
                    
                    # 同时更新 manager 中的记录 (仅用于日志显示，不再涉及 ID 轮换)
                    manager.delta_sei_list[manager.bat_in_station_ids[idx]] = delta_sei_in
                    manager.cf_list[manager.bat_in_station_ids[idx]] = cf_in
                end
                
                println("  执行换电: $num_swap 块 (换入电池 SOC重置为 $soc_in)")
            else
                println("  换电需求: $demand, 但可用电池不足 (仅 $(length(eligible_idx)) 块)")
            end
        end

        # 6. 全局 DAE 平衡 (为下一小时准备一致初值)
        # 虽然换电电池已经平衡过，但为了保险，对所有电池再做一次检查
        for k in 1:NUM_BATTERIES_IN_STATION
            u_next[k, :] = safe_initial_state(init_solver, u_next[k, :], 0.0)
        end

        # 打印统计信息
        # 注意：不再调用 update_bss_manager!，因为我们不再追踪外部 ID
        @printf("Step %d | Mean CF: %.10e | Max CF: %.10e | Profit: %.4f\n", 
                current_step, mean(u_next[:, 12]), maximum(u_next[:, 12]), profit)

        u0_current = copy(u_next)
    end
    println("累计总收益: $history_profit")
end


# 执行主程序
run_rolling_mpc()