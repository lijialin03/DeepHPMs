from example import Example


lr = 0.0001
N_train = [30000, 30000, 30000]
# max_grad == 3 or 4 is not supported by paddle now
max_grad_dict = {"burgers": 1, "kdv": 1, "ks": 1}
mode = [
    # "load_gen_pde",
    "train_gen_pde",
    # "debug_gen_pde",
    # "save_gen_pde",
    "train_pinns",
    # "debug_pinns",
    # "save_pinns",
]
example = Example()

# Burgers_Different_Swap
example.run(
    "../../Data/burgers.mat",
    "../../Data/burgers_sine.mat",
    "Burgers_fail",
    max_grad_dict["burgers"],
    lr,
    N_train[0],
    N_train[1],
    N_train[2],
    mode,
)

# Burgers_Different
example.run(
    "../../Data/burgers_sine.mat",
    "../../Data/burgers.mat",
    "Burgers_Extrapolate",
    max_grad_dict["burgers"],
    lr,
    N_train[0],
    N_train[1],
    N_train[2],
    mode,
)

# Burgers_Same
example.run(
    "../../Data/burgers_sine.mat",
    "../../Data/burgers_sine.mat",
    "Burgers",
    max_grad_dict["burgers"],
    lr,
    N_train[0],
    N_train[1],
    N_train[2],
    mode,
)

# # Kdv_Different
# example.run(
#     "../../Data/KdV_sine.mat",
#     "../../Data/KdV_cos.mat",
#     "KdV_Extrapolate",
#     max_grad_dict["kdv"],
#     lr,
#     N_train[0],
#     N_train[1],
#     N_train[2],
#     mode,
# )

# # Kdv_Same
# example.run(
#     "../../Data/KdV_sine.mat",
#     "../../Data/KdV_sine.mat",
#     "KdV",
#     max_grad_dict["kdv"],
#     lr,
#     N_train[0],
#     N_train[1],
#     N_train[2],
#     mode,
# )

# # KS_chaotic
# example.run(
#     "../../Data/KS_chaotic.mat",
#     "../../Data/KS_chaotic.mat",
#     "KS_nasty",
#     max_grad_dict["ks"],
#     lr,
#     N_train[0],
#     N_train[1],
#     N_train[2],
#     mode,
# )

# # KS
# example.run(
#     "../../Data/KS.mat",
#     "../../Data/KS.mat",
#     "KS",
#     max_grad_dict["ks"],
#     lr,
#     N_train[0],
#     N_train[1],
#     N_train[2],
#     mode,
# )

# # Schrodinger
# example.run(
#     "../../Data/NLS.mat",
#     "../../Data/NLS.mat",
#     "NLS",
#     2,
#     lr,
#     N_train,
#     N_train,
#     N_train,
# )

# # NavierStokes
# example.run(
#     "../../Data/cylinder.mat",
#     "../../Data/cylinder.mat",
#     "NavierStokes",
#     2,
#     lr,
#     N_train,
#     N_train,
#     N_train,
# )
