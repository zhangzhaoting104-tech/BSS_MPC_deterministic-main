using DelimitedFiles, Plots, DataFrames, XLSX, Measures

include("../utils/utils.jl")


Price_File = "../data/rtm_cal.csv"
Demand_File = "../data/长泰国际金融大厦.xlsx"
required_length = 24 * 180

Prices = readdlm("../data/slow_Price.csv", ',', Float64)
Prices = vcat(Prices'...)[1:required_length]
Prices = (Prices .- minimum(Prices)) ./ (maximum(Prices) - minimum(Prices))

swap_counts_data = load_hourly_data(Demand_File)
swap_counts_data = vcat(swap_counts_data, swap_counts_data)[1:required_length]
swap_counts_data = (swap_counts_data .- minimum(swap_counts_data)) ./ (maximum(swap_counts_data) - minimum(swap_counts_data))

price_matrix = reshape(Prices, 24, 180)
swap_matrix = reshape(swap_counts_data, 24, 180)


xticks = vcat(1, collect(20:20:180))
push!(xticks, 180)
xticks = unique(sort(xticks))
yticks = collect(4:4:24)
push!(yticks, 24)
yticks = unique(sort(yticks))

colorgradient = cgrad(
    [:yellow, :orange, :red, :black],
    [0.0, 0.1, 0.2, 0.3, 1.0],
    scale=:linear
)
p1 = heatmap(price_matrix,
    title="Electricity Price",
    xlabel="Day Index", ylabel="Hour of the day",
    xticks=(xticks, string.(xticks)),
    yticks=(yticks, string.(yticks)),
    color=colorgradient,
    aspect_ratio=2,
    framestyle=:left,
    grid=false,
    colorbar=false,
    xrotation=90,
    xlims=(1, 180),
    ylims=(1, 24),
    titlefont=font(4),
    guidefont=font(4),
    tickfont=font(4),
    tick_direction=:out,
    titlelocation=:center,
    top_margin=-2mm
)

colorgradient = cgrad(
    [:white, :yellow, :orange, :red, :black],
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    scale=:linear
)
p2 = heatmap(swap_matrix,
    title="Battery Swaps",
    xlabel="Day Index", ylabel="Hour of the day",
    xticks=(xticks, string.(xticks)),
    yticks=(yticks, string.(yticks)),
    color=colorgradient,
    aspect_ratio=2,
    framestyle=:left,
    grid=false,
    colorbar=false,
    xrotation=90,
    xlims=(1, 180),
    ylims=(1, 24),
    titlefont=font(4),
    guidefont=font(4),
    tickfont=font(4),
    tick_direction=:out,
    titlelocation=:center,
    top_margin=-2mm
)

plot(p1, p2, layout=@layout([a; b]), size=(400, 250))


gui()
sleep(100)