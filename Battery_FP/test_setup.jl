# test_simple_fixed.jl
include("setup.jl")

println("=== 测试修复后的 getinitial 函数 ===")

# 使用出错时的参数进行测试
csp_avg = 5173.778657414506
csn_avg = 14740.0
delta_sei = 1e-10
value = -36.909589447586875
mode = 3

println("调用 getinitial 函数...")
try
    result = getinitial(csp_avg, csn_avg, delta_sei, value, mode)
    println("getinitial 调用成功!")
    println("结果: csp_s = $(result[1]), csn_s = $(result[2])")
    println("      pot = $(result[6]), it = $(result[7])")
catch e
    println("getinitial 调用失败: ", e)
    println("使用备用值继续...")
end

# 初始化电池状态进行完整测试
println("\n=== 初始化电池状态 ===")
soc = 0.5
csp_avg0 = cspmax * 1 - csnmax * soc * lnn * en / lp / ep
csn_avg0 = csnmax * soc
delta_sei0 = 1e-10

u0 = zeros(Ncp + Ncn + 4 + Nsei + Ncum)
u0[1:Ncp] .= csp_avg0
u0[(Ncp+1):(Ncp+Ncn)] .= csn_avg0
if Sei
    u0[Ncp+Ncn+7] = delta_sei0
end
if Cum
    u0[Ncp+Ncn+Nsei+5] = 0
    u0[Ncp+Ncn+Nsei+6] = 0
    if Sei
        u0[Ncp+Ncn+Nsei+7] = 0
        u0[Ncp+Ncn+Nsei+8] = 0
    end
end

println("初始化完成!")
println("初始 SOC: $soc")
println("csp_avg0: $csp_avg0")
println("csn_avg0: $csn_avg0")