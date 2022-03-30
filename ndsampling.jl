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


n = 140
initial_n = 138
d = 8
lb = [0.05,100,63070,990,63.1,700,1120,9855]
ub = [0.15,50000,115600,1110,116,820,1680,12045]
x = sample(initial_n,lb,ub,SobolSample())
y = f.(x)




for sample_iter in 1:(n-initial_n)
    curr_sampled_n = length(x)
    print(curr_sampled_n)

    tempx = copy(x)
    tempy = f.(tempx)

    function cv_error(new_sample_point_x)
        norm = 0
        poly_surrogate = PolynomialChaosSurrogate(tempx, tempy, lb, ub)
    
        for sampled_point in tempx
            loo_x = copy(tempx)
            loo_y = copy(tempy)
            deleteat!(loo_x, findall(x->x==sampled_point,loo_x))
            deleteat!(loo_y, findall(x->x==f(sampled_point),loo_y))
            loo_poly_surrogate = PolynomialChaosSurrogate(loo_x, loo_y, lb, ub)
            norm = norm + (loo_poly_surrogate(new_sample_point_x) - poly_surrogate(new_sample_point_x))^2
        end
        e = (norm / length(tempx))^(1/2)
    end


    function min_distance(new_sample_point_x)
        arr_x = [collect(i) for i in tempx]
        sample_point_x_arr = fill(new_sample_point_x, length(tempx))
        distance_arr = broadcast(-, sample_point_x_arr, arr_x)
        distance_arr = broadcast(norm, distance_arr)
        d = minimum(distance_arr)
    end

    new_sample_point = (0.05,100,63070,990,63.1,700,1120,9855)
    max_opt = 0
    for target_sample_x in sample(n,lb,ub,UniformSample())
        if target_sample_x in tempx
            continue
        else
            arr_target_sample_x = [i for i in target_sample_x]
            opt = cv_error(arr_target_sample_x) * min_distance(arr_target_sample_x)
            if opt > max_opt
                max_opt = opt
                new_sample_point = target_sample_x
            end
        end
    end

    print(new_sample_point)
    print("  ")

    push!(x, new_sample_point)
end



n_test = 1000
x_test = sample(n_test,lb,ub,GoldenSample());
y_true = f.(x_test);

my_rad = RadialBasis(x,y,lb,ub)
y_rad = my_rad.(x_test)
my_poly = PolynomialChaosSurrogate(x,y,lb,ub)
y_poli = my_poli.(x_test)
mse_rad = norm(y_true - y_rad,2)/n_test
mse_poli = norm(y_true - y_poli,2)/n_test
print("MSE Radial: $mse_rad")
print("MSE Radial: $mse_poli")


