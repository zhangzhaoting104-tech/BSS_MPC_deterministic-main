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
    println("=== 启动 BSS 套利控制系统 ===")
    
    manager = create_bss_manager(NUM_BATTERIES_IN_STATION, NUM_BATTERIES_IN_AREA)
    u0_current = get_initial_states(manager) 
    init_solver = get_initial_state_solver()
    
    history_profit = 0.0
    
    for current_step in 1:TOTAL_HOURS
        println("\nStep $current_step / $TOTAL_HOURS")
        
        # 1. 获取预测
        idx_end = min(current_step + HORIZON_H - 1, length(Prices))
        price_forecast = Prices[current_step:idx_end]
        demand_forecast = swap_counts_data[current_step:idx_end]
        
        # 2. MPC 决策
        mpc_model, power_var = build_mpc_model(u0_current, length(price_forecast), price_forecast, demand_forecast)
        optimize!(mpc_model)
        
        # 3. 容错处理：若求解失败则保持不充电 [cite: 6]
        optimal_p = zeros(NUM_BATTERIES_IN_STATION)
        if termination_status(mpc_model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
            optimal_p = value.(power_var[:, 1])
        end

        optimal_p = clamp.(optimal_p, -P_max, P_max)
        for k in 1:NUM_BATTERIES_IN_STATION
            soc = u0_current[k,3] / csnmax
            if soc < 0.25 || soc > 0.85
                optimal_p[k] = 0.0
             end
        end

        # 4. 执行物理模拟 [cite: 8]
        sim_instance = Simulator(u0_current[1, :], 0.0)

u_next, profit, p_real, _, _, _ =
    simulate(
        sim_instance,
        init_solver,              # 👈 显式传入
        u0_current,
        price_forecast[1],
        optimal_p,
        Int[],
        zeros(0,4)
    )

        history_profit += profit

        # # 5. 换电执行逻辑 [cite: 12, 13]
        # demand = Int(swap_counts_data[current_step])
        # if demand > 0
        #     # 索引说明：1:csp_avg, 3:csn_avg, 11:delta_sei, 12:cf [cite: 82, 87]
        #     soc_list = u_next[:, 3] ./ csnmax
        #     eligible_idx = findall(x -> x >= replaceable_soc, soc_list)
        #     num_swap = min(length(eligible_idx), demand)
            
        #     if num_swap > 0
        #         swap_indices = sortperm(soc_list, rev=true)[1:num_swap]
        #         new_states = get_swap_state(manager, num_swap)
                
        #         for (i, idx) in enumerate(swap_indices)
        #             # 重置状态为换入电池
        #             u_next[idx, 1] = new_states[i, 1]  # csp_avg
        #             u_next[idx, 2] = new_states[i, 1]  # csp_s
        #             u_next[idx, 3] = new_states[i, 2]  # csn_avg
        #             u_next[idx, 4] = new_states[i, 2]  # csn_s
        #             u_next[idx, 11] = new_states[i, 3] # delta_sei
        #             u_next[idx, 12] = new_states[i, 4] # cf
        #             # 代数变量清零，待下一步平衡
        #             u_next[idx, 5:10] .= 0.0 
        #         end
        #         update_bss_manager!(manager, swap_indices, u_next[:, 11], u_next[:, 12])
        #         println("  执行换电: $num_swap 块")
        #     end
        # end

        # # 6. DAE 平衡（关键：为下一步提供一致初值） [cite: 20]
        # for k in 1:NUM_BATTERIES_IN_STATION
        #     u_next[k, :] = compute_initial_state(init_solver, u_next[k, :], 0.0)
        # end
        # u0_current = copy(u_next)
        # end
        # 5. 换电执行逻辑
        demand = Int(swap_counts_data[current_step])
        if demand > 0
            soc_list = u_next[:, 3] ./ csnmax # csn_avg is index 3
            eligible_idx = findall(x -> x >= replaceable_soc, soc_list)
            num_swap = min(length(eligible_idx), demand)
            
            if num_swap > 0
                swap_indices = sortperm(soc_list, rev=true)[1:num_swap]
                new_states = get_swap_state(manager, num_swap) # new_states: [csp_avg, csn_avg, delta_sei, cf]
                
                for (i, idx) in enumerate(swap_indices)
                    # === 关键：最严格的零电流初始状态重置 ===
                    
                    # 1. 浓度状态 (微分变量)
                    u_next[idx, 1] = new_states[i, 1]  # csp_avg
                    u_next[idx, 2] = new_states[i, 1]  # csp_s (在 I=0 时，c_s = c_avg)
                    u_next[idx, 3] = new_states[i, 2]  # csn_avg
                    u_next[idx, 4] = new_states[i, 2]  # csn_s (在 I=0 时，c_s = c_avg)
                    
                    # 2. 退化状态 (微分变量)
                    u_next[idx, 11] = new_states[i, 3] # delta_sei
                    u_next[idx, 12] = new_states[i, 4] # cf
                    
                    # 3. 代数变量 (电流、电势): 必须全部清零，让 compute_initial_state 重新计算平衡点
                    # 假设代数变量在 5:10
                    u_next[idx, 5:10] .= 0.0 
                    u_next[idx, :] = compute_initial_state(
                        init_solver,
                        u_next[idx, :],
                        0.0
                        )

                    # 4. 如果状态向量更长，确保其余也归零
                    if size(u_next, 2) > 12
                        u_next[idx, 13:end] .= 0.0 
                    end
                    # === 重置结束 ===
                end
                update_bss_manager!(manager, swap_indices, u_next[:, 11], u_next[:, 12])
                
                println("  执行换电: $num_swap 块")
            end
        end

        # 6. DAE 平衡
        for k in 1:NUM_BATTERIES_IN_STATION
            # compute_initial_state 内部会使用 Ipopt 找到 DAE 兼容的初值。
            u_next[k, :] = safe_initial_state(init_solver, u_next[k, :], 0.0)
        end
        u0_current = copy(u_next)
    end
    println("累计总收益: $history_profit")
end
# 执行主程序
run_rolling_mpc()