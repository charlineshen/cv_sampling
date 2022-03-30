using Surrogates
using Plots
default()

# Define salustowicz function
function salustowicz(x)
    term1 = 2.72^(-x) * x^3 * cos(x) * sin(x);
    term2 = (cos(x) * sin(x)*sin(x) - 1);
    y = term1 * term2;
end

# Define gramacylee function
function gramacylee(x)
    term1 = sin(10*pi*x) / 2*x;
    term2 = (x - 1)^4;
    y = term1 + term2;
end

n_samples = 40      # number of sampling points
lower_bound = -0.5
upper_bound = 2.5
num_round = 2

# x = AbstractFloat[i*2 for i in 0:5]
x = AbstractFloat[-0.5, 0, 0.5, 1.5, 2, 2.5]    # Set initial sampling points

anim = @animate for sample_iter in 1:n_samples-length(x)
    tempx = copy(x)     # Make a copy of current sampled points
    # tempy = salustowicz.(tempx)
    tempy = gramacylee.(tempx)      # Calculate the true values of sampled points

    # function that calcualtes the cv error of a potential point
    function cv_error(new_sample_point_x)
        norm = 0
        radial_surrogate = RadialBasis(tempx, tempy, lower_bound, upper_bound)  # find the surrogate function using current sampled points
        
        # for each sampled point, we find the leave-one-out surrogate function
        for sampled_point in 1:length(tempx)
            loo_x = copy(tempx)
            loo_y = copy(tempy)
            deleteat!(loo_x, sampled_point)
            deleteat!(loo_y, sampled_point)
            loo_radial_surrogate = RadialBasis(loo_x, loo_y, lower_bound, upper_bound)  # find the surrogate function that leaves one sampled point out
            norm = norm + (loo_radial_surrogate(new_sample_point_x) - radial_surrogate(new_sample_point_x))^2   # Add up the norm to calculate cv error
        end
        e = (norm / length(tempx))^(1/2)    # Return the cv error by mean squaring the norm
    end

    # function that returns the minimum distance between the input point and one of the already sampled points
    function min_distance(new_sample_point_x)
        sample_point_x_arr = fill(new_sample_point_x, length(tempx))
        distance_arr = broadcast(-, sample_point_x_arr, tempx)
        distance_arr = broadcast(abs, distance_arr)
        d = minimum(distance_arr)
    end

    new_sample_point = 0
    max_opt = 0     # number to maximize (cv_error * minimum_distance)
    for target_sample_x in lower_bound:0.01:upper_bound
        if target_sample_x in tempx
            continue        # skip the points already sampled
        else
            
            opt = cv_error(target_sample_x) * min_distance(target_sample_x)     # Calculate the output of optimization function for the point
            if opt > max_opt
                max_opt = opt
                new_sample_point = target_sample_x      # Update the new sampling point as the point with biggest max_opt
            end
        end
    end

    print(new_sample_point)
    print("  ")

    push!(x, new_sample_point)      # Add the new sampled point into the sampling points list


    # Graph surrogate plot with new point
    y = gramacylee.(x)
    radial_surrogate = RadialBasis(x, y, lower_bound, upper_bound)
    scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
    plot!(radial_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound), legend=:top, size=(800, 600))

end

# Create an animation of the sampling process
gif(anim, "1d cv sampling.gif", fps = 2)


# y = salustowicz.(x)
y = gramacylee.(x)  # find the true values of all sampling points

radial_surrogate = RadialBasis(x, y, lower_bound, upper_bound)  # find the surrogate function with all sampling points

# Graph the true function and the surrogate function using out sampled points
xs = lower_bound:0.001:upper_bound
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
# plot!(xs, salustowicz.(xs), label="True function", legend=:top)
plot!(xs, gramacylee.(xs), label="True function", legend=:top)
plot!(radial_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound), legend=:top)