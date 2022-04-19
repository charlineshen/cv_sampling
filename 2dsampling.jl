# Run:
# import Pkg; Pkg.add("Surrogates")
# import Pkg; Pkg.add("PolyChaos")
# import Pkg; Pkg.add("Plots")
# import Pkg; Pkg.add("LinearAlgebra")
using Surrogates
using PolyChaos
using Plots
using LinearAlgebra
default()

# Define the 2d Rosenbrock function
function Rosenbrock2d(x)
    x1 = x[1]
    x2 = x[2]
    return (1-x1)^2 + 100*(x2-x1^2)^2
end


n = 100     # Number of total sampling points
lb = [0.0,0.0]
ub = [8.0,8.0]
initial_n = 17      # Number of initial sampling points

xys = sample(initial_n,lb,ub,SobolSample());    # Sample initial points
# Sample points on the edges
push!(xys, (0.0, 0.0))
push!(xys, (0.0, 8.0))
push!(xys, (8.0, 0.0))
push!(xys, (8.0, 8.0))
zs = Rosenbrock2d.(xys);

# Plot the PolynomialChaosSurrogate created by initial sampling points
poly_surrogate = PolynomialChaosSurrogate(xys, zs, lb, ub)
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
x, y = 0:8, 0:8
# Surface Plot
ps = surface(x, y, (x, y) -> poly_surrogate([x y]), title="Initial Polynomial expansion")
scatter!(xs, ys, zs, marker_z=zs)
# Contour Plot
pc = contour(x, y, (x, y) -> poly_surrogate([x y]), title="Initial Polynomial expansion")
scatter!(xs, ys, marker_z=zs)
display(plot(ps, pc))

# Start the sampling process; Create an animation of sampling points
anim = @animate for sample_iter in 1:(n-initial_n-4)
    # Record the nth sampled point
    curr_sampled_n = length(xys)
    print(curr_sampled_n)

    # Make copies of current sampled points
    tempxys = copy(xys)
    tempzs = Rosenbrock2d.(tempxys)


    # Function that returns cv error of a point
    function cv_error(new_sample_point_x)
        norm = 0
        poly_surrogate = PolynomialChaosSurrogate(tempxys, tempzs, lb, ub)  # Get PolynomialChaosSurrogate of current sampled points
    
        # for each sampled point, we find the leave-one-out surrogate function
        for sampled_point in tempxys
            loo_xys = copy(tempxys)
            loo_zs = copy(tempzs)
            deleteat!(loo_xys, findall(x->x==sampled_point,loo_xys))
            deleteat!(loo_zs, findall(x->x==Rosenbrock2d(sampled_point),loo_zs))
            loo_poly_surrogate = PolynomialChaosSurrogate(loo_xys, loo_zs, lb, ub)  # find the surrogate function that leaves one sampled point out
            norm = norm + (loo_poly_surrogate(new_sample_point_x) - poly_surrogate(new_sample_point_x))^2   # Add up the norm to calculate cv error
        end
        e = (norm / length(tempxys))^(1/2)  # Return the cv error by mean squaring the norm
    end


    # Function that returns the minimum distance between the input point and one of the already sampled points
    function min_distance(new_sample_point_x)
        arr_xys = [collect(i) for i in tempxys]
        sample_point_x_arr = fill(new_sample_point_x, length(tempxys))
        distance_arr = broadcast(-, sample_point_x_arr, arr_xys)
        distance_arr = broadcast(norm, distance_arr)
        d = minimum(distance_arr)
    end


    new_sample_point = (0.0, 0.0)
    max_opt = 0     # number to maximize (cv_error * minimum_distance)
    for target_sample_xys in sample(curr_sampled_n,lb,ub,SobolSample())
        if target_sample_xys in tempxys
            continue       # skip the points already sampled
        else
            arr_target_sample_xys = [i for i in target_sample_xys]
            opt = cv_error(arr_target_sample_xys) * min_distance(arr_target_sample_xys)     # Calculate the output of optimization function for the point
            if opt > max_opt
                max_opt = opt
                new_sample_point = target_sample_xys    # Update the new sampling point as the point with biggest max_opt
            end
        end
    end

    print(new_sample_point)
    print("  ")

    push!(xys, new_sample_point)    # Add the new sampled point into the sampling points list


    # Graph surrogate plot with new point
    tempzs = Rosenbrock2d.(xys);
    poly_surrogate = PolynomialChaosSurrogate(xys, tempzs, lb, ub)
    tempxs = [xy[1] for xy in xys]
    tempys = [xy[2] for xy in xys]
    tempps = surface(x, y, (x, y) -> poly_surrogate([x y]), title="$curr_sampled_n Points Polynomial expansion")
    scatter!(tempxs, tempys, tempzs, marker_z=tempzs)
    temppc = contour(x, y, (x, y) -> poly_surrogate([x y]), title="$curr_sampled_n Points Polynomial expansion")
    scatter!(tempxs, tempys, marker_z=tempzs)
    plot(tempps, temppc, size = (1000, 800))
end

# Create an animation of the sampling process
gif(anim, "2d cv sampling.gif", fps = 2)


zs = Rosenbrock2d.(xys);    # find the true values of all sampling points
poly_surrogate = PolynomialChaosSurrogate(xys, zs, lb, ub)  # create the surrogate with all sampling points

# Plot the true function
true_xys = sample(n,lb,ub,SobolSample());
true_zs = Rosenbrock2d.(true_xys);
x, y = 0:8, 0:8
p1 = surface(x, y, (x1,x2) -> Rosenbrock2d((x1,x2)), title="True function")
true_xs = [xy[1] for xy in true_xys]
true_ys = [xy[2] for xy in true_xys]
p2 = contour(x, y, (x1,x2) -> Rosenbrock2d((x1,x2)), title="True function")
scatter!(true_xs, true_ys)

# Plot the surrogate function created by all sampled points
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
p3 = surface(x, y, (x, y) -> poly_surrogate([x y]), title="Polynomial expansion")
scatter!(xs, ys, zs, marker_z=zs)
p4 = contour(x, y, (x, y) -> poly_surrogate([x y]), title="Polynomial expansion")
scatter!(xs, ys, marker_z=zs)
display(plot!(p1, p2, p3, p4, size=(1000,800), reuse=false))