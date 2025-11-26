using CSV, DataFrames


function record(filename, cf_avg, cf_std, cumulative_profit, num_soc_violate_list)
    data = DataFrame(
        cf_avg=Float64[],
        cf_std=Float64[],
        cumulative_profit=Float64[],
        num_soc_violate=Int32[],
        num_swap=Int32[],
    )

    for t in 1:length(cf_avg)
        row = vcat(cf_avg[t], cf_std[t], cumulative_profit[t], num_soc_violate_list[t][1], num_soc_violate_list[t][2])
        push!(data, collect(row'))
    end

    names = ["cf_avg", "cf_std", "cumulative_profit", "num_soc_violate", "num_swap"]
    rename!(data, names)
    CSV.write(filename, data)
    println("✅ Data written to $filename")
end