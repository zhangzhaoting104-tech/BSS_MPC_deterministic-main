# bss_simulator.jl

using DelimitedFiles
using Sundials
using Interpolations
using ModelingToolkit
using BlockDiagonals
using LinearAlgebra
using SymbolicIndexingInterface: parameter_values

# 引入 setup 和 utils (假设已在主程序引入，这里为了独立性保留，实际运行时如果是 include 链不需要重复)
include("setup.jl")
include("utils.jl")

# f_common 保持不变，直接复用原代码中的定义
# ... (f_common 代码省略，保持原样) ...
function f_common(out, du, u, p, t)
    

  csp = u[1:Ncp]
  csn = u[(Ncp+1):(Ncp+Ncn)]
  # csp = max.(1, min.(csp, cspmax - 1))
  # csn = max.(1, min.(csn, csnmax - 1))
  csp_avg = csp[1]
  csp_s = csp[2]
  csn_avg = csn[1]
  csn_s = csn[2]

  iint = u[Ncp+Ncn+1]
  phi_p = u[Ncp+Ncn+2]
  phi_n = u[Ncp+Ncn+3]
  pot = u[Ncp+Ncn+4]

  it = u[Ncp+Ncn+5]
  isei = u[Ncp+Ncn+6]
  delta_sei = u[Ncp+Ncn+7]

  cf = u[Ncp+Ncn+Nsei+5]

  #C2. Additional Equaions
  #Positive electrode
  theta_p = csp_s / cspmax
  Up = 7.49983 - 13.7758 * theta_p^0.5 + 21.7683 * theta_p - 12.6985 * theta_p^1.5 + 0.0174967 / theta_p - 0.41649 * theta_p^(-0.5) - 0.0161404 * exp(100 * theta_p - 97.1069) +
       0.363031 * tanh(5.89493 * theta_p - 4.21921)
  jp = 2 * kp * ce^(0.5) * (cspmax - csp_s)^(0.5) * csp_s^(0.5) * sinh(0.5 * F / R / T * (phi_p - Up))

  #Negative electrode
  theta_n = csn_s / csnmax
  Un = 9.99877 - 9.99961 * theta_n^0.5 - 9.98836 * theta_n + 8.2024 * theta_n^1.5 + 0.23584 / theta_n - 2.03569 * theta_n^(-0.5) -
       1.47266 * exp(-1.14872 * theta_n + 2.13185) -
       9.9989 * tanh(0.60345 * theta_n - 1.58171)
  jn = 2 * kn * ce^(0.5) * (csnmax - csn_s)^(0.5) * csn_s^(0.5) * sinh(0.5 * F / R / T * (phi_n - Un + (Rsei + delta_sei / Kappa_sei) * it / an / lnn))
    # f_common 开头
csp_s = clamp(csp_s, 1e-6, cspmax - 1e-6)
csn_s = clamp(csn_s, 1e-6, csnmax - 1e-6)

theta_p = clamp(csp_s / cspmax, 1e-6, 1 - 1e-6)
theta_n = clamp(csn_s / csnmax, 1e-6, 1 - 1e-6)
  #C1. Governing Equations
  #Positive electrode
  out[1] = -3 * jp / Rpp - du[1]
  out[2] = 5 * (csp_s - csp_avg) + Rpp * jp / Dp

  #Negative electrode
  out[3] = -3 * jn / Rpn - du[Ncp+1]
  out[4] = 5 * (csn_s - csn_avg) + Rpn * jn / Dn

  out[5] = (jp - (it / ap / F / lp))
  out[6] = (jn + (iint / an / F / lnn))
  out[7] = pot - phi_p + phi_n
  out[8] = p - pot * it

  #C3. SEI layer Equations
  out[9] = -iint + it - isei
  out[10] = -isei + an * lnn * ksei * exp(-1 * F / R / T * (phi_n - Urefs + it / an / lnn * (delta_sei / Kappa_sei + Rsei)))
  out[11] = isei * M_sei / F / rho_sei / an / lnn - du[Ncp+Ncn+4+3]

  #C4. Charge stored
  out[12] = isei / 3600 - du[Ncp+Ncn+Nsei+5]
  @assert isfinite(theta_p) && theta_p > 0 "theta_p invalid"
    @assert isfinite(theta_n) && theta_n > 0 "theta_n invalid"
end

mutable struct Simulator
  prob
  tspan

  # 修改构造函数：battery_charge 应该是一个标量 (单个电池的电流)
  function Simulator(u0_single, battery_charge_scalar)

    TIME_SEGMENT = 0:dt:DT_STATE
    tspan = (float(TIME_SEGMENT[1]), float(TIME_SEGMENT[end]))

    differential_vars = falses(Ncp + Ncn + 4 + Nsei + Ncum)
    differential_vars[1] = true
    differential_vars[3] = true
    differential_vars[11] = true
    differential_vars[12] = true

    # 关键修改：传入标量 battery_charge_scalar
    # prob = SciMLBase.DAEProblem(f_common, zero(u0_single), u0_single, tspan, battery_charge_scalar, differential_vars=differential_vars)
    du0 = zeros(length(u0_single))
    du0[1] = 0.0          # csp_avg
    du0[Ncp+1] = 0.0     # csn_avg
    du0[Ncp+Ncn+4+3] = 0.0  # delta_sei
    du0[Ncp+Ncn+Nsei+5] = 0.0 # cf

    prob = DAEProblem(
      f_common,
      du0,
      u0_single,
      tspan,
      battery_charge_scalar;
      differential_vars = differential_vars
)

    new(prob, tspan)
  end
end


# function simulate(simulator, u0, grid_price, battery_charge, swap_battery_idx, swap_battery_state; log::Bool=false)
function simulate(
    simulator::Simulator,
    init_solver::InitialStateSolver,
    u0::Matrix,
    grid_price::Float64,
    battery_charge::Vector,
    swap_battery_idx::Vector,
    swap_battery_state::Matrix;
    log::Bool = false
)
    swap_set = Set(swap_battery_idx)
    start_time = time()

    u1 = zero(u0)

    cf0 = u0[:, Ncp+Ncn+Nsei+5]
    fade = cf0 ./ Qmax
    capacity_remain = 1 .- fade

    for k in 1:NUM_BATTERIES_IN_STATION

        # ================================
        # 0️⃣ 默认：什么都不做
        # ================================
        u1[k, :] = u0[k, :]

        # ================================
        # 1️⃣ 换电电池：只做准静态
        # ================================
        if k in swap_set
            u1[k, :] .= safe_initial_state(
                init_solver,
                u0[k, :],
                battery_charge[k]
            )
            continue
        end

        # ================================
        # 2️⃣ 非换电电池：DAE
        # ================================
        u0_consistent = safe_initial_state(
            init_solver,
            u0[k, :],
            battery_charge[k]
        )

        function stop_cond(u, t, integrator)
            csn_avg = u[Ncp+1]
            soc_in = csn_avg / csnmax
            min_stop = capacity_remain[k] * soc_min_stop
            max_stop = capacity_remain[k] * soc_max_stop

            if soc_in >= (min_stop + max_stop) / 2
                return max_stop - soc_in
            else
                return soc_in - min_stop
            end
        end

        affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(stop_cond, affect!; rootfind = true)

        prob = remake(
            simulator.prob,
            u0 = copy(u0_consistent),
            p  = Float64(battery_charge[k])
        )

        sol = solve(
            prob,
            IDA();
            initializealg = DiffEqBase.BrownFullBasicInit(),
            reltol = 1e-4,
            abstol = 1e-4,
            adaptive = false,
            callback = cb,
            verbose = false
        )

        # ================================
        # 3️⃣ DAE 成功 / 失败处理
        # ================================
        if sol.retcode == :Success || sol.retcode == :Terminated
            u1[k, :] = sol.u[end]
        else
            # fallback：准静态
            u_proj, flag = quasi_static_project(
                init_solver,
                u0_consistent,
                Float64(battery_charge[k])
            )

            if flag == :success
                u1[k, :] = u_proj
            else
                u1[k, :] = compute_initial_state(init_solver, u0[k, :], 0.0)
            end
        end

        # ================================
        # 4️⃣ 功率修正（保留你原逻辑）
        # ================================
        if battery_charge[k] < 0 && sol.retcode == :Success
            battery_charge[k] *= sol.t[end] / simulator.tspan[end]
        end
    end

    # ================================
    # 后处理（你原来的逻辑）
    # ================================
    delta_sei_list = copy(u1[:, 11])
    cf_list = copy(u1[:, 12])

    cost = (expectedrevenue / (1 - soc_retire)) *
           sum((u1[:, 12] - u0[:, 12])) / Qmax

    soc_cost_coeffi = 20
    num_soc_violate = [0, length(swap_battery_idx)]

    for i in 1:length(swap_battery_idx)
        idx = swap_battery_idx[i]
        soc = u1[idx, 3] / csnmax
        if soc < replaceable_soc
            cost += soc_cost_coeffi * (replaceable_soc - soc)
            num_soc_violate[1] += 1
        end

        # 状态替换
        u1[idx, 1] = swap_battery_state[i, 1]
        u1[idx, 3] = swap_battery_state[i, 2]
        u1[idx, Ncp+Ncn+4+3] = swap_battery_state[i, 3]
        u1[idx, Ncp+Ncn+Nsei+5] = swap_battery_state[i, 4]
    end

    profit = -grid_price * sum(battery_charge) - cost

    if log
        println("simulation succeeded, elapsed time: ", time() - start_time)
    end

    return u1, profit, battery_charge, delta_sei_list, cf_list, num_soc_violate
end
#kexingbanben.jl