# Run:
# import Pkg; Pkg.add("Surrogates")
# import Pkg; Pkg.add("PolyChaos")
# import Pkg; Pkg.add("Plots")
# import Pkg; Pkg.add("LinearAlgebra")
# import Pkg; Pkg.add("LatinHypercubeSampling")
using Surrogates
using PolyChaos
using Plots
using LinearAlgebra
using LatinHypercubeSampling
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
    poly_surrogate = PolynomialChaosSurrogate(xys, zs, lb, ub)
    xs = [xy[1] for xy in xys]
    ys = [xy[2] for xy in xys]
    x, y = 0:8, 0:8
    num_points = length(xys)
    # Surface Plot
    ps = surface(x, y, (x, y) -> poly_surrogate([x y]), title="$num_points Points Polynomial expansion")
    scatter!(xs, ys, zs, marker_z=zs)
    # Contour Plot
    pc = contour(x, y, (x, y) -> poly_surrogate([x y]), title="$num_points Points Polynomial expansion")
    scatter!(xs, ys, marker_z=zs)
    display(plot(ps, pc, size = (1000, 800)))
end

# Convert a matrix into a vector with each element as a tuple of matrix's row
function convert_to_vector(matrix)
    vec = collect(eachrow(matrix))
end

# A function that takes in a surrogate and a set of training samples and adaptively chooses the next point to sample the next training point
# surrogate_func -- a surrogate function
# training_samples_mat -- a matrix of sampled points with each row as a sampled point
# n -- number of points we want to sample
# Return: a set of sampling points in matrix, including the initial sampled points and n points sampled during the algo
function cv_sampling(surrogate_func, training_samples_mat, n)
    dim = size(training_samples_mat)[2] # Get the dimension of the input point

    # Start the sampling process; Create an animation of sampling points
    anim = @animate for sample_iter in 1:n
        # Convert the training samples matrix into a vector of sampled points
        training_samples = convert_to_vector(training_samples_mat)

        num_sampled = length(training_samples)
        print("$(num_sampled+1)th Sampled Point: ")

        # Make copies of current sampled points
        copy_xs = copy(training_samples)
        copy_ys = Rosenbrock2d.(copy_xs)


        # Function that returns cv error of a point
        function cv_error(point_x)
            norm = 0
            surrogate = surrogate_func(copy_xs, copy_ys, lb, ub)  # Get a surrogate by fitting current sampled points
        
            # for each sampled point, we find the leave-one-out surrogate function
            for sampled_point in copy_xs
                loo_xs = copy(copy_xs)
                loo_ys = copy(copy_ys)
                deleteat!(loo_xs, findall(x->x==sampled_point,loo_xs))
                deleteat!(loo_ys, findall(x->x==Rosenbrock2d(sampled_point),loo_ys))
                loo_surrogate = surrogate_func(loo_xs, loo_ys, lb, ub)  # find the surrogate function that leaves one sampled point out
                norm = norm + (loo_surrogate(point_x) - surrogate(point_x))^2   # Add up the norm to calculate cv error
            end
            e = (norm / length(copy_xs))^(1/2)  # Return the cv error by mean squaring the norm
        end


        sampled_points_arr = [collect(x) for x in copy_xs]  # turn the terms in copy_xs into arrays
        # Function that returns the minimum distance between the input point and one of the already sampled points
        function min_distance(point_x)
            point_x_arr = fill(point_x, length(copy_xs))
            distance_arr = norm.(point_x_arr - sampled_points_arr)
            d = minimum(distance_arr)
        end


        clean_datatypef = x -> [i for i in x]       # Defines a function that put a point in list datatype that can be input to other functions
        target_sample_xs = sample(eval_n,lb,ub,SobolSample())      # Sample some target points to evaluate the optimization function on
        cleaned_target_sample_xs = broadcast(clean_datatypef, target_sample_xs)     # Get a list of target sample points in list datatype
        opt = cv_error.(cleaned_target_sample_xs) .* min_distance.(cleaned_target_sample_xs)    # Optimization function: cv_error * min_distance
        new_sample_point = target_sample_xs[argmax(opt)]    # Find the optimal point
        print("$new_sample_point   ")

        # Add the new sampled point into the sampled points list
        training_samples_mat = vcat(training_samples_mat, reshape(collect(new_sample_point), 1, dim))


        # Graph surrogate plot with new point
        vec_xs = convert_to_vector(training_samples_mat)
        vec_ys = Rosenbrock2d.(vec_xs)
        plot_surrogate(vec_xs, vec_ys)
    end

    # Create an animation of the sampling process
    gif(anim, "results/2d cv sampling.gif", fps = 2)

    out = training_samples_mat
end


initial_n = 17    # Number of initial sampling points
total_n = 23     # Number of total points we want to sample (initial_n + n)
eval_n = 350    # Number of potential points to evaluate in each iteration
lb = [0.0,0.0]
ub = [8.0,8.0]

# Sample initial points
xs, _ = LHCoptim(initial_n, 2, 1000);
xs = scaleLHC(xs, [(0.0, 8.0), (0.0, 8.0)])
# Sample points on the edges
xs = [xs;[0.0 0.0]]
xs = [xs;[0.0 8.0]]
xs = [xs;[8.0 0.0]]
xs = [xs;[8.0 8.0]]
initial_n = initial_n + 4

# Plot the surrogate plot with initial sampled points
vec_xs = convert_to_vector(xs)
vec_ys = Rosenbrock2d.(vec_xs)
plot_surrogate(vec_xs, vec_ys)

# Do the sampling process
xs = cv_sampling(PolynomialChaosSurrogate, xs, total_n - initial_n)



# Plot the true function
true_xys = sample(total_n,lb,ub,SobolSample());
true_zs = Rosenbrock2d.(true_xys);
x, y = 0:8, 0:8
p1 = surface(x, y, (x1,x2) -> Rosenbrock2d((x1,x2)), title="True function")
true_xs = [xy[1] for xy in true_xys]
true_ys = [xy[2] for xy in true_xys]
p2 = contour(x, y, (x1,x2) -> Rosenbrock2d((x1,x2)), title="True function")
scatter!(true_xs, true_ys)
display(plot!(p1, p2, size=(1000,800), reuse=false))

# Plot the surrogate function created by all sampled points
vec_xs = convert_to_vector(xs)
vec_ys = Rosenbrock2d.(vec_xs)  # find the true values of all sampled points
plot_surrogate(vec_xs, vec_ys)