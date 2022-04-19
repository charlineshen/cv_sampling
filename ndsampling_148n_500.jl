# Run:
# import Pkg; Pkg.add("Surrogates")
# import Pkg; Pkg.add("PolyChaos")
# import Pkg; Pkg.add("Plots")
using Surrogates
using PolyChaos
using Plots
using LinearAlgebra
default()

# Define the objective waterflow function (8 dimensions)
function f(x)
    r_w = x[1]
    r = x[2]
    T_u = x[3]
    H_u = x[4]
    T_l = x[5]
    H_l = x[6]
    L = x[7]
    K_w = x[8]
    log_val = log(r/r_w)
    return (2*pi*T_u*(H_u - H_l))/ ( log_val*(1 + (2*L*T_u/(log_val*r_w^2*K_w)) + T_u/T_l))
end


n = 148     # number of total sampling points
initial_n = 138     # number of initial sampling points
node_n = 500        # number of target points to evaluate in one sampling iteration
d = 8   # dimension
lb = [0.05,100,63070,990,63.1,700,1120,9855]
ub = [0.15,50000,115600,1110,116,820,1680,12045]
x = sample(initial_n,lb,ub,SobolSample())   # sample the initial points using sobolsample
y = f.(x)

sample_iters = initial_n:n
cv_mses = []
other_mses = []

# A function that returns the MSE of 1000 tested points evaluated on PolynomialChaos Surrogates created using input cv sampling points
function calculateCVMSE(sampled_x)
    n_test = 1000
    x_test = sample(n_test,lb,ub,GoldenSample());
    y_true = f.(x_test);

    sampled_y = f.(sampled_x)
    my_poli = PolynomialChaosSurrogate(sampled_x,sampled_y,lb,ub)
    y_poli = my_poli.(x_test)
    mse_poli = norm(y_true - y_poli,2)/n_test
end

# A function that returns the MSE of 1000 tested points evaluated on PolynomialChaos Surrogates created using input random sampling points
# Input is the number of random sampling points
# We choose the method of random sampling the same as the initial sampling method.
function calculateSobolMSE(sample_n)
    n_test = 1000
    x_test = sample(n_test,lb,ub,GoldenSample());
    y_true = f.(x_test);

    random_x = sample(sample_n,lb,ub,SobolSample());    # with same random sampling sequence as the initial samples
    random_y_true = f.(random_x)
    other_poli = PolynomialChaosSurrogate(random_x,random_y_true,lb,ub)
    other_y_poli = other_poli.(x_test)
    mse_other_poli = norm(y_true - other_y_poli, 2)/n_test
end


# each iteration of sampling a new point
for sample_iter in 1:(n-initial_n)
    # Record the nth sampled point
    curr_sampled_n = length(x)
    print(curr_sampled_n)

    # Calculate MSE with current number of sampled points and add it to list
    current_cv_mse = calculateCVMSE(x)
    push!(cv_mses, current_cv_mse)
    push!(other_mses, calculateSobolMSE(curr_sampled_n))    # Add the MSE of random sampling n points to the list
    print("     MSE PolynomialChaos: $current_cv_mse")

    # Make copies of current sampled points
    tempx = copy(x)
    tempy = f.(tempx)

    # Function that returns cv error of a point
    function cv_error(new_sample_point_x)
        norm = 0
        poly_surrogate = PolynomialChaosSurrogate(tempx, tempy, lb, ub) # Get PolynomialChaosSurrogate of current sampled points
    
        # for each sampled point, we find the leave-one-out surrogate function
        for index in collect(1:length(tempx))
            loo_x = copy(tempx)
            loo_y = copy(tempy)
            deleteat!(loo_x, index)
            deleteat!(loo_y, index)
            loo_poly_surrogate = PolynomialChaosSurrogate(loo_x, loo_y, lb, ub) # find the surrogate function that leaves one sampled point out
            norm = norm + (loo_poly_surrogate(new_sample_point_x) - poly_surrogate(new_sample_point_x))^2   # Add up the norm to calculate cv error
        end
        e = (norm / length(tempx))^(1/2)    # Return the cv error by mean squaring the norm
    end

    # Function that returns the minimum distance between the input point and one of the already sampled points
    function min_distance(new_sample_point_x)
        arr_x = [collect(i) for i in tempx]
        sample_point_x_arr = fill(new_sample_point_x, length(tempx))
        distance_arr = broadcast(-, sample_point_x_arr, arr_x)
        distance_arr = broadcast(norm, distance_arr)
        d = minimum(distance_arr)
    end

    new_sample_point = (0.05,100,63070,990,63.1,700,1120,9855)
    max_opt = 0 # number to maximize (cv_error * minimum_distance)
    for target_sample_x in sample(node_n,lb,ub,SobolSample())
        if target_sample_x in tempx
            continue    # skip the points already sampled
        else
            arr_target_sample_x = [i for i in target_sample_x]
            opt = cv_error(arr_target_sample_x) * min_distance(arr_target_sample_x) # Calculate the output of optimization function for the point
            if opt > max_opt
                max_opt = opt
                new_sample_point = target_sample_x  # Update the new sampling point as the point with biggest max_opt
            end
        end
    end

    print(new_sample_point)
    print("\n")

    push!(x, new_sample_point)  # Add the new sampled point into the sampling points list
end

y = f.(x)   # Get true values of sampled points



# Evalate the sampling results
# Find the MSE of 1000 tested points evaluated on RadialBasis and PolynomialChaos Surrogates created using our cv sampling points
n_test = 1000
x_test = sample(n_test,lb,ub,GoldenSample());
y_true = f.(x_test);

my_rad = RadialBasis(x,y,lb,ub)
y_rad = my_rad.(x_test)
my_poli = PolynomialChaosSurrogate(x,y,lb,ub)
y_poli = my_poli.(x_test)
mse_rad = norm(y_true - y_rad,2)/n_test
mse_poli = norm(y_true - y_poli,2)/n_test
print("MSE RadialBasis: $mse_rad    ")
print("MSE PolynomialChaos: $mse_poli    ")

push!(cv_mses, mse_poli)   # Push the final cv mse with all the sampled points to the cv mse list
push!(other_mses, calculateSobolMSE(n))     # Push the mse of sampling all the points randomly to the list


# Make a plot of how MSE changes during sampling
plot(sample_iters, cv_mses, title="PolynomialChaos MSE vs number of total sampling points", label="cv mse_poli")
plot!(sample_iters, other_mses, label="sobol mse_poli")
xlabel!("number of total sampling points")
ylabel!("PolynomialChaos MSE")
savefig("ndsampling_waterflow_MSE_148n_500.png")