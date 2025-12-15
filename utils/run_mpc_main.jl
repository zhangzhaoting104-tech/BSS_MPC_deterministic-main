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
TOTAL_HOURS = 1 * SIM_DAYS # 模拟总时长
HORIZON_H = 6              # 预测视界 (小时)
DT_SIM = 300.0            # 模拟步长 (单位：秒）

function run_rolling_mpc()
    println("=== 启动 BSS MPC 滚动时域控制 ===")
    
    # 1. 初始化管理器和电池状态
    manager = create_bss_manager(NUM_BATTERIES_IN_STATION, NUM_BATTERIES_IN_AREA)
    u0_current = get_initial_states(manager) 
    
    # 2. 平衡代数方程，获取物理一致的初始状态
    println("  正在计算初始状态平衡点...")
    init_solver = get_initial_state_solver()
    for k in 1:NUM_BATTERIES_IN_STATION
        # compute_initial_state 使用 power=0.0 来平衡 DAE 系统
        u0_current[k, :] = compute_initial_state(init_solver, u0_current[k, :], 0.0)
    end
    
    history_profit = Float64[]
    current_time_step = 1
    
    # 3. 开始滚动时域循环
    while current_time_step <= TOTAL_HOURS
        println("\n--- Time Step: $current_time_step / $TOTAL_HOURS ---")
        
        # A. 获取预测数据 (电价和需求)
        # ------------------------------------------------------------------
        idx_start = current_time_step
        idx_end = min(current_time_step + HORIZON_H - 1, length(Prices))
        current_horizon_len = idx_end - idx_start + 1
        
        # 定义 price_forecast (确保在当前作用域可见)
        price_forecast = Prices[idx_start:idx_end]
        demand_forecast = swap_counts_data[idx_start:idx_end]
        
        # 数据补齐 (如果接近数据末尾，视界不足 HORIZON_H)
        if current_horizon_len < HORIZON_H
            append!(price_forecast, fill(price_forecast[end], HORIZON_H - current_horizon_len))
            append!(demand_forecast, fill(0, HORIZON_H - current_horizon_len))
        end
        # ------------------------------------------------------------------

        println("  预测视界: $HORIZON_H 小时 | 当前电价: $(price_forecast[1])")

        # B. 构建并求解 MPC 问题
        # ------------------------------------------------------------------
        println("  正在求解 MPC 优化问题...")
        mpc_model, power_var = build_mpc_model(u0_current, HORIZON_H, price_forecast, demand_forecast)
        optimize!(mpc_model)
        
        optimal_power_step1 = zeros(Float64, NUM_BATTERIES_IN_STATION)
        
        status = termination_status(mpc_model)
        # 接受 OPTIMAL, LOCALLY_SOLVED, 甚至 ITERATION_LIMIT 作为可行解
        if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ITERATION_LIMIT]
            optimal_power_step1 = value.(power_var[:, 1])
            println("  MPC 求解结束 ($status). 目标函数值: $(objective_value(mpc_model))")
        else
            println("  Warning: MPC 求解失败 ($status)！执行 P=0 安全策略")
            # 保持 optimal_power_step1 全为 0
        end
        # ------------------------------------------------------------------
        
        # C. 模拟物理环境 (执行决策)
        # ------------------------------------------------------------------
        println("  正在模拟物理环境 (1 小时)...")
        
        # === 关键修复: Simulator 构造 ===
        # Simulator 构造函数现在期望 (u0_single, battery_charge_scalar)
        # 我们使用第1个电池的状态和0.0电流作为模板来初始化 DAE 系统结构
        sim_instance = Simulator(u0_current[1, :], 0.0)
        
        dummy_swap_idx = Int[] 
        dummy_swap_state = zeros(0, 4)
        
        # 调用 simulate
        # simulate 内部会循环遍历每个电池，使用 remake 更新状态和电流
        u_next, step_profit, _, _, _, _ = simulate(
            sim_instance, 
            u0_current, 
            price_forecast[1], # 当前时刻电价
            optimal_power_step1, 
            dummy_swap_idx, 
            dummy_swap_state, 
            log=false
        )
        
        push!(history_profit, step_profit)
        println("  模拟完成. 本小时实际收益: $step_profit")
        # ------------------------------------------------------------------

        # D. 换电逻辑 (State Transition - Swapping)
        # ------------------------------------------------------------------
        current_demand = Int(swap_counts_data[current_time_step])
        if current_demand > 0
            # 计算当前 SOC (根据 csn_avg)
            soc_current = u_next[:, Ncp+1] ./ csnmax
            
            # 找到 SOC >= replaceable_soc 的电池
            candidate_indices = findall(x -> x >= replaceable_soc, soc_current)
            # 按 SOC 从高到低排序，优先换出高电量电池
            sorted_indices = sort(candidate_indices, by=i->soc_current[i], rev=true)
            
            num_swaps = min(length(sorted_indices), current_demand)
            
            if num_swaps > 0
                swap_indices = sorted_indices[1:num_swaps]
                
                # 获取新电池状态 (从 Manager)
                new_batteries_state = get_swap_state(manager, num_swaps) 
                
                # === 修正开始 ===
                # update_bss_manager! 需要全站所有电池的最新状态来更新记录
                # 因此这里传入 u_next 整列，而不是仅传入 swap_indices
                update_bss_manager!(
                    manager, 
                    swap_indices, 
                    u_next[:, Ncp+Ncn+7],       # 修正：传入所有电池的 delta_sei
                    u_next[:, Ncp+Ncn+Nsei+5]   # 修正：传入所有电池的 cf
                )
                # === 修正结束 ===
                
                # 将站内电池状态重置为新换入的电池状态
                for (i, idx) in enumerate(swap_indices)
                    # new_batteries_state 结构: [csp_avg, csn_avg, delta_sei, cf]
                    u_next[idx, 1] = new_batteries_state[i, 1] # csp_avg
                    u_next[idx, 2] = new_batteries_state[i, 1] # csp_s (近似重置为平均值)
                    u_next[idx, Ncp+1] = new_batteries_state[i, 2] # csn_avg
                    u_next[idx, Ncp+2] = new_batteries_state[i, 2] # csn_s (近似重置为平均值)
                    u_next[idx, Ncp+Ncn+7] = new_batteries_state[i, 3] # delta_sei
                    u_next[idx, Ncp+Ncn+Nsei+5] = new_batteries_state[i, 4] # cf
                end
                println("  执行换电: $num_swaps 块 (Indices: $swap_indices)")
            end
        end
        # ------------------------------------------------------------------
        
        # E. 重新平衡代数方程 (Solver Reset for Next Step)
        # ------------------------------------------------------------------
        # 这一步至关重要：模拟器输出的状态可能在代数上不完全平衡（因为换电或模拟误差）
        # 或者我们需要为下一轮 MPC 提供一个以 Power=0 为基准的稳定初值
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