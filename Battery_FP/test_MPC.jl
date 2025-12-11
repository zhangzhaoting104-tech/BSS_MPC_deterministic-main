# test_mpc_flexible_original.jl
println("=== 测试原始 MPC_flexible.jl ===")

# 1. 首先修复 MPC_flexible.jl 中的潜在问题
mpc_flexible_code = read("MPC_flexible.jl", String)

# 检查并修复常见问题
# 1. 确保使用 optimize! 而不是 solve
# 2. 确保没有 IpoptSolver 的旧语法
# 3. 确保所有变量都已定义

# 创建修复版本
lines = split(mpc_flexible_code, "\n")
fixed_lines = []

for line in lines
    # 修复 IpoptSolver 语法
    if contains(line, "Model(solver=IpoptSolver(")
        line = replace(line, "Model(solver=IpoptSolver(" => 
                         "Model(optimizer_with_attributes(Ipopt.Optimizer, ")
    end
    
    # 修复 solve 调用
    if contains(line, "JuMP.solve(")
        line = replace(line, "JuMP.solve(" => "optimize!(")
    end
    
    push!(fixed_lines, line)
end

mpc_flexible_fixed = join(fixed_lines, "\n")

# 写入临时文件
write("MPC_flexible_fixed.jl", mpc_flexible_fixed)

# 2. 运行修复后的版本
println("运行修复后的 MPC_flexible...")

# 在执行之前，确保所有必要的变量都已定义
# 我们已经在前面的脚本中定义了它们

# 包含修复后的文件
include("MPC_flexible_fixed.jl")

# 调用 MPC 函数（只运行一次迭代）
println("\n调用 MPC 函数...")

# 修改 MPC 函数，使其只运行一次迭代
function MPC_test(u0, method)
    println("MPC_test 开始 (方法: $method)")
    
    # 只运行一次迭代
    i_start = 1
    i_end = Nt_FR_hour
    
    if method == "MPC_flexible"
        println("调用 OptimalControl...")
        try
            # 这里需要 OptimalControl 函数可用
            FR_band, grid_band, waste = OptimalControl(u0, 0.0, 0.2, 0.8)
            println("OptimalControl 成功")
            println("FR_band: $FR_band")
            println("grid_band: $grid_band")
        catch e
            println("OptimalControl 失败: ", e)
        end
    else
        println("其他方法暂不支持测试")
    end
    
    println("MPC_test 完成")
end

# 运行测试
MPC_test(u0, "MPC_flexible")

println("\n测试完成!")