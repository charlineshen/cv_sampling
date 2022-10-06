# Run:
# import Pkg; Pkg.add("Surrogates")
# Pkg.add("PolyChaos")
# Pkg.add("Plots")
# Pkg.add("LinearAlgebra")
# Pkg.add("QuasiMonteCarlo")
# Pkg.add("MLBase")
using Surrogates
using PolyChaos
using Plots
using LinearAlgebra
using QuasiMonteCarlo
using MLBase
default()

# Define the 2d Rosenbrock function
function Rosenbrock2d(x)
    x1 = x[1]
    x2 = x[2]
    return (1-x1)^2 + 100*(x2-x1^2)^2
end

# Some Helper Functions
# A function that plots the Polynomial surrogate created by a set of points
function plot_surrogate(xys, zs)
    # Plot the PolynomialChaosSurrogate created by points
    radial_surrogate = RadialBasis(xys, zs, lb, ub, rad = cubicRadial)
    xs = [xy[1] for xy in xys]
    ys = [xy[2] for xy in xys]
    x, y = 0.0:8.0, 0.0:8.0
    num_points = length(xys)
    # Surface Plot
    ps = surface(x, y, (x, y) -> radial_surrogate([x y]), title="$num_points Points rbf expansion")
    scatter!(xs, ys, zs, marker_z=zs)
    # Contour Plot
    pc = contour(x, y, (x, y) -> radial_surrogate([x y]), title="$num_points Points rbf expansion")
    scatter!(xs, ys, marker_z=zs)
    display(plot(ps, pc, size = (1000, 800)))
end


# Function that returns cv error of a point
function cv_error(surrogate_func, xs, ys, lb, ub, point_x)
    surrogate = surrogate_func(xs, ys, lb, ub, rad = cubicRadial)  # Get a surrogate by fitting current sampled points
    num_sampled = length(xs)

    loocv_indices = collect(LOOCV(num_sampled)) # Get an array of loo indices; Each element of the array is an array of loo indices
    loo_f(i) = xs[loocv_indices[i]]   # A loo function that select one sery of loo indices and apply it on the sampled points
    loo_xs = loo_f.(1:num_sampled)  # Broadcast the loo function on all kinds of loo indices
    calculatey_f(i) = Rosenbrock2d.(loo_xs[i])  # A function that calculates the y-values of a sery of loo points in loo_xs given its index
    loo_ys = calculatey_f.(1:num_sampled)   # Get all y-values of all kinds of loo series
    loo_surrogate_f(i) = surrogate_func(loo_xs[i], loo_ys[i], lb, ub, rad = cubicRadial)   # A function that gets loo surrogate function by fitting loo points 
    loo_surrogate = loo_surrogate_f.(1:num_sampled) # An array of loo_surrogates

    calculate_difference_f(i) = loo_surrogate[i](point_x) - surrogate(point_x)  # A function that calculates the difference of loo_surrogate and surrogate on the point
    norms = norm(calculate_difference_f.(1:num_sampled))  # Calculate the norm of all differences
    e = norms / num_sampled^(1/2)    # Return the cv error
end


# Function that returns the minimum distance between the input point and one of the already sampled points
function min_distance(sampled_points_arr, point_x)
    point_x_arr = fill(point_x, length(sampled_points_arr))
    distance_arr = norm.(point_x_arr - sampled_points_arr)
    d = minimum(distance_arr)
end


# A function that takes in a surrogate and a set of training samples and adaptively chooses the next point to sample the next training point
# surrogate_func -- a surrogate function
# training_samples_mat -- a matrix of sampled points with each row as a sampled point
# target_sample_xs -- a vector of target points to evaluate the optimization function on in each iteration
# lb, ub -- lower bound and upper bound of the input domain
# n -- number of points we want to sample
# Return: a set of sampling points in matrix, including the initial sampled points and n points sampled during the algo
function cv_sampling(surrogate_func, training_samples_mat, target_sample_xs, lb, ub, n)
    # Convert the training samples matrix into a vector of sampled points
    training_xs = collect(eachrow(training_samples_mat))
    training_ys = Rosenbrock2d.(training_xs)


    cleaned_target_sample_xs = [collect(x) for x in target_sample_xs]     # Get a list of target sample points in list datatype
    cv_errors = cv_error.((surrogate_func,), (training_xs,), (training_ys,), (lb,), (ub,), target_sample_xs)
    min_distances = min_distance.((training_xs,), cleaned_target_sample_xs)
    opt = cv_errors .* min_distances    # Optimization function: cv_error * min_distance
    new_sample_points = target_sample_xs[partialsortperm(opt, 1:n, rev=true)]    # Find the optimal point
    print("Sample Points: $new_sample_points   ")

    # Add the new sampled point into the sampled points list
    training_samples_mat = vcat(training_samples_mat, reduce(hcat, [collect(x) for x in new_sample_points])')

    out = training_samples_mat
end


initial_n = 17    # Number of initial sampling points
total_n = 25     # Number of total points we want to sample (initial_n + n)
eval_n = 100    # Number of potential points to evaluate in each iteration
d = 2   # dimension of the problem
lb = [0.0,0.0]
ub = [8.0,8.0]

# Sample initial points
xs = QuasiMonteCarlo.sample(initial_n, lb, ub, QuasiMonteCarlo.LatinHypercubeSample())'
# Sample some target points to evaluate the optimization function on in each iteration
target_sample_xs = Surrogates.sample(eval_n,lb,ub,Surrogates.SobolSample())

# Sample points on the edges
reorder_lb_ub = collect(eachrow([lb ub]))
cornor_points = reshape(collect(Iterators.product(reorder_lb_ub...)), 2^d, 1)
xs = vcat(xs, reduce(hcat, [collect(x) for x in cornor_points])')
initial_n = initial_n + 4


# Plot the surrogate plot with initial sampled points
vec_xs = collect(eachrow(xs))
vec_ys = Rosenbrock2d.(vec_xs)
plot_surrogate(vec_xs, vec_ys)

# Do the sampling process
xs = cv_sampling(RadialBasis, xs, target_sample_xs, lb, ub, total_n - initial_n)


# Plot the true function
true_xys = Surrogates.sample(total_n,lb,ub,Surrogates.SobolSample());
true_zs = Rosenbrock2d.(true_xys);
x, y = 0:8, 0:8
p1 = surface(x, y, (x1,x2) -> Rosenbrock2d((x1,x2)), title="True function")
true_xs = [xy[1] for xy in true_xys]
true_ys = [xy[2] for xy in true_xys]
p2 = contour(x, y, (x1,x2) -> Rosenbrock2d((x1,x2)), title="True function")
scatter!(true_xs, true_ys)
display(plot!(p1, p2, size=(1000,800), reuse=false))

# Plot the surrogate function created by all sampled points
vec_xs = collect(eachrow(xs))
vec_ys = Rosenbrock2d.(vec_xs)  # find the true values of all sampled points
plot_surrogate(vec_xs, vec_ys)