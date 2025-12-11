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
TOTAL_HOURS = 1 * SIM_DAYS
HORIZON_H = 6 # 预测时域 24 小时
DT_SIM = 3600.0 # 每次执行 1 小时 (对应 setup.jl 中的 DT_STATE)

# run_mpc_main.jl

# ... (前面的引用保持不变)

function run_rolling_mpc()
    println("=== 启动 BSS MPC 滚动时域控制 ===")
    
    # 1. 初始化管理器和电池
    manager = create_bss_manager(NUM_BATTERIES_IN_STATION, NUM_BATTERIES_IN_AREA)
    u0_current = get_initial_states(manager) 
    
    # 平衡代数方程，获取物理一致的初始状态
    init_solver = get_initial_state_solver()
    for k in 1:NUM_BATTERIES_IN_STATION
        u0_current[k, :] = compute_initial_state(init_solver, u0_current[k, :], 0.0)
    end
    
    history_profit = Float64[]
    
    current_time_step = 1
    
    while current_time_step <= TOTAL_HOURS
        println("\n--- Time Step: $current_time_step / $TOTAL_HOURS ---")
        
        # A. 获取预测数据
        idx_start = current_time_step
        idx_end = min(current_time_step + HORIZON_H - 1, length(Prices))
        current_horizon_len = idx_end - idx_start + 1
        
        price_forecast = Prices[idx_start:idx_end]
        demand_forecast = swap_counts_data[idx_start:idx_end]
        
        # 补齐数据
        if current_horizon_len < HORIZON_H
            append!(price_forecast, fill(price_forecast[end], HORIZON_H - current_horizon_len))
            append!(demand_forecast, fill(0, HORIZON_H - current_horizon_len))
        end

        println("  预测视界: $HORIZON_H 小时 | 当前电价: $(price_forecast[1])")

        # B. 构建并求解 MPC 问题
        println("  正在求解 MPC 优化问题...")
        # 传入 u0_current 用于初始化优化变量
        mpc_model, power_var = build_mpc_model(u0_current, HORIZON_H, price_forecast, demand_forecast)
        optimize!(mpc_model)
        
        optimal_power_step1 = zeros(NUM_BATTERIES_IN_STATION)
        
       if termination_status(mpc_model) == MOI.OPTIMAL || termination_status(mpc_model) == MOI.LOCALLY_SOLVED
        optimal_power_step1 = value.(power_var[:, 1])
        println("  MPC 求解成功. 目标函数值: $(objective_value(mpc_model))")
    else
    # 即使 MPC 失败，也必须确保 optimal_power_step1 是 float 向量
        println("  Warning: MPC 求解失败 ($(termination_status(mpc_model)))！执行 P=0 策略")
        optimal_power_step1 = zeros(Float64, NUM_BATTERIES_IN_STATION) 
    end
        
        # C. 模拟物理环境 (1 小时)
        println("  正在模拟物理环境 (1 小时)...")
        
        # === 修复点 1: 正确初始化 Simulator ===
        # Simulator 需要单个电池的向量作为模板，而不是整个矩阵
        # 这里的 u0_current[1, :] 是向量，optimal_power_step1[1] 是标量
        sim_instance = Simulator(u0_current[1, :], optimal_power_step1[1])
        
        # 构造模拟参数
        dummy_swap_idx = Int[] 
        dummy_swap_state = zeros(0, 4)
        
        # 调用 simulate
        # 注意: simulate 内部会根据传入的 u0_current (矩阵) 和 optimal_power_step1 (向量)
        # 循环对每个电池进行 remake 和求解
        u_next, step_profit, _, _, _, _ = simulate(
            sim_instance, 
            u0_current, 
            price_forecast[1], # 当前电价
            optimal_power_step1, 
            dummy_swap_idx, 
            dummy_swap_state, 
            log=false
        )
        
        push!(history_profit, step_profit)
        println("  模拟完成. 本小时实际收益: $step_profit")

        # D. 换电逻辑 (简化版)
        current_demand = Int(swap_counts_data[current_time_step])
        if current_demand > 0
            # 找到 SOC 足够高且能被换电的电池
            # 注意: csn_avg 位于索引 Ncp+1
            soc_current = u_next[:, Ncp+1] ./ csnmax
            candidate_indices = findall(x -> x >= replaceable_soc, soc_current)
            sorted_indices = sort(candidate_indices, by=i->soc_current[i], rev=true)
            
            num_swaps = min(length(sorted_indices), current_demand)
            
            if num_swaps > 0
                swap_indices = sorted_indices[1:num_swaps]
                new_batteries_state = get_swap_state(manager, num_swaps) 
                
                # 更新 Manager
                update_bss_manager!(
                    manager, 
                    swap_indices, 
                    u_next[swap_indices, Ncp+Ncn+7], 
                    u_next[swap_indices, Ncp+Ncn+Nsei+5]
                )
                
                # 写入新电池状态
                for (i, idx) in enumerate(swap_indices)
                    u_next[idx, 1] = new_batteries_state[i, 1] # csp_avg
                    u_next[idx, 3] = new_batteries_state[i, 1] # csp_s (重置)
                    u_next[idx, Ncp+1] = new_batteries_state[i, 2] # csn_avg
                    u_next[idx, Ncp+2] = new_batteries_state[i, 2] # csn_s (重置)
                    u_next[idx, Ncp+Ncn+7] = new_batteries_state[i, 3] # delta_sei
                    u_next[idx, Ncp+Ncn+Nsei+5] = new_batteries_state[i, 4] # cf
                end
                println("  执行换电: $num_swaps 块 (Indices: $swap_indices)")
            end
        end
        
        # E. 重新平衡代数方程 (Solver Reset)
        # 为下一步 MPC 提供合法的起点
        for k in 1:NUM_BATTERIES_IN_STATION
            u_next[k, :] = compute_initial_state(init_solver, u_next[k, :], 0.0)
        end
        
        u0_current = copy(u_next)
        current_time_step += 1
    end
    
    println("=== 模拟结束 ===")
    println("累计收益: $(sum(history_profit))")
end

# 执行主程序
run_rolling_mpc()