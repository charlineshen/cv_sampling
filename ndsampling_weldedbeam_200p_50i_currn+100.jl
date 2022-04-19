# Run:
# import Pkg; Pkg.add("Surrogates")
# import Pkg; Pkg.add("PolyChaos")
# import Pkg; Pkg.add("Plots")
using Surrogates
using PolyChaos
using Plots
using LinearAlgebra
default()

# Define the objective welded beam function (3 dimensions)
function f(x)
    h = x[1]
    l = x[2]
    t = x[3]
    a = 6000/(sqrt(2)*h*l)
    b = (6000*(14+0.5*l)*sqrt(0.25*(l^2+(h+t)^2)))/(2*(0.707*h*l*(l^2/12 + 0.25*(h+t)^2)))
    return (sqrt(a^2+b^2 + l*a*b))/(sqrt(0.25*(l^2+(h+t)^2)))
end


n = 200     # number of total sampling points
initial_n = 50     # number of initial sampling points
node_n = 200        # number of target points to evaluate in one sampling iteration
# iter_n = 10         # number of sampling points in each iteration
d = 3   # dimension
lb = [0.125,5.0,5.0]
ub = [1.,10.,10.]
x = sample(initial_n,lb,ub,SobolSample())   # sample the initial points using sobolsample

# push!(x, (0.125,5.0,5.0))
# push!(x, (1.,10.,10.))
y = f.(x)

sample_iters = initial_n:n
cv_mses = []
other_mses = []

# A function that returns the MSE of 1000 tested points evaluated on PolynomialChaos Surrogates created using input cv sampling points and same number of random sampling points
function calculateMSE(sampled_x)
    n_test = 1000
    x_test = sample(n_test,lb,ub,GoldenSample());
    y_true = f.(x_test);

    sampled_y = f.(sampled_x)
    my_poli = LobachevskySurrogate(sampled_x,sampled_y,lb,ub)
    y_poli = my_poli.(x_test)

    sample_n = length(sampled_x)
    random_x = sample(sample_n,lb,ub,SobolSample());    # with same random sampling sequence as the initial samples
    random_y_true = f.(random_x)
    other_poli = LobachevskySurrogate(random_x,random_y_true,lb,ub)
    other_y_poli = other_poli.(x_test)

    mse_poli, mse_other_poli = norm(y_true - y_poli,2)/n_test, norm(y_true - other_y_poli, 2)/n_test
end


# each iteration of sampling a new point
for sample_iter in 1:(n-initial_n)
    # Record the nth sampled point
    curr_sampled_n = length(x)
    print(curr_sampled_n)

    # Calculate cv MSE and sobol MSE with current number of sampled points and add it to list
    current_cv_mse, current_other_mse = calculateMSE(x)
    push!(cv_mses, current_cv_mse)
    push!(other_mses, current_other_mse)
    print("     MSE Lobachevsky: $current_cv_mse")

    # Make copies of current sampled points
    tempx = copy(x)
    tempy = f.(tempx)

    # Function that returns cv error of a point
    function cv_error(new_sample_point_x)
        norm = 0
        poly_surrogate = LobachevskySurrogate(tempx, tempy, lb, ub) # Get PolynomialChaosSurrogate of current sampled points
    
        # for each sampled point, we find the leave-one-out surrogate function
        for index in collect(1:length(tempx))
            loo_x = copy(tempx)
            loo_y = copy(tempy)
            deleteat!(loo_x, index)
            deleteat!(loo_y, index)
            loo_poly_surrogate = LobachevskySurrogate(loo_x, loo_y, lb, ub) # find the surrogate function that leaves one sampled point out
            norm = norm + (loo_poly_surrogate(new_sample_point_x) - poly_surrogate(new_sample_point_x))^2   # Add up the norm to calculate cv error
        end
        e = (norm / length(tempx))^(1/2)    # Return the cv error by mean squaring the norm
    end

    # Function that returns the minimum distance between the input point and one of the already sampled points
    function min_distance(new_sample_point_x)
        arr_x = [collect(i) for i in tempx]
        sample_point_x_arr = fill(new_sample_point_x, length(tempx))
        distance_arr = sample_point_x_arr - arr_x
        distance_arr = norm.(distance_arr)
        d = minimum(distance_arr)
    end

    clean_datatypef = x -> [i for i in x]       # Defines a function that put a point in list datatype that can be input to other functions
    target_sample_xs = sample(curr_sampled_n+100,lb,ub,SobolSample())
    cleaned_target_sample_xs = broadcast(clean_datatypef, target_sample_xs)
    opt = cv_error.(cleaned_target_sample_xs) .* min_distance.(cleaned_target_sample_xs)
    # opt = min_distance.(cleaned_target_sample_xs)
    new_sample_point = target_sample_xs[argmax(opt)]

    print(new_sample_point)
    # print(cv_error([i for i in new_sample_point]), "    ", min_distance([i for i in new_sample_point]))
    print("\n")

    push!(x, new_sample_point)  # Add the new sampled point into the sampling points list
end

y = f.(x)   # Get true values of sampled points



# Evalate the sampling results
# Find the MSE of 1000 tested points evaluated on RadialBasis and PolynomialChaos Surrogates created using our cv sampling points
final_mse, final_mse_sobol = calculateMSE(x)
print("MSE Lobachevsky: $final_mse    ")
print("MSE Sobol Sampling: $final_mse_sobol    ")
push!(cv_mses, final_mse)   # Push the final cv mse with all the sampled points to the cv mse list
push!(other_mses, final_mse_sobol)     # Push the mse of sampling all the points randomly to the list


# Make a plot of how MSE changes during sampling
plot(sample_iters, cv_mses, title="Lobachevsky MSE vs number of total sampling points", label="cv mse_loba")
plot!(sample_iters, other_mses, label="sobol mse_loba")
xlabel!("number of total sampling points")
ylabel!("Lobachevsky MSE")
savefig("ndsampling_weldedbeam_MSE_200p_50i.pdf")