#Import Json configuration
import json5

with open("config/bayesian_opt.json", "r") as f:
    config = json5.load(f)

#bayesian optimization
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

#config
num_trials = 10

#configure ax client
client = Client()

parameters = [
    RangeParameterConfig(
        name="batch_size",
        parameter_type="int",
        bounds=(16, 512),
    ),
    RangeParameterConfig(
        name="num_epochs",
        parameter_type="int",
        bounds=(10, 200),
    ),
    RangeParameterConfig(
        name="hidden_dim_ac",
        parameter_type="int",
        bounds=(50, 600),
    ),
    RangeParameterConfig(
        name="hidden_dim_cr",
        parameter_type="int",
        bounds=(50, 600),
    ),
    RangeParameterConfig(
        name="embedding_dim_ac_sel",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="embedding_dim_cr_sel",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="embedding_dim_ac_B",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="embedding_dim_cr_B",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="embedding_dim_ac_u",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="embedding_dim_cr_u",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="embedding_dim_ac_t",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="embedding_dim_cr_t",
        parameter_type="int",
        bounds=(50, 300),
    ),
    RangeParameterConfig(
        name="lr_ac",
        parameter_type="float",
        bounds=(1e-5, 1e-2),
        scaling="log",
    ),
    RangeParameterConfig(
        name="lr_cr",
        parameter_type="float",
        bounds=(1e-5, 1e-2),
        scaling="log",
    ),
    RangeParameterConfig(
        name="mom_ac",
        parameter_type="float",
        bounds=(0.0, 0.99),
    ),
    RangeParameterConfig(
        name="mom_cr",
        parameter_type="float",
        bounds=(0.0, 0.99),
    ),
    RangeParameterConfig(
        name="penalty_weight",
        parameter_type="float",
        bounds=(0.1, 10.0),
        scaling="log",
    ),
    RangeParameterConfig(
        name="cut_interval",
        parameter_type="int",
        bounds=(10, 200),
    ),
]


parameter_defs = config["parameters_space"]
parameters = []

for name, p in parameter_defs.items():
    parameters.append(
        RangeParameterConfig(
            name=name,
            parameter_type=p["type"],
            bounds=tuple(p["bounds"]),
            scaling=p.get("scaling", None)  # scaling is optional
        )
    )

client.configure_experiment(parameters=parameters)

metric_name = "hartmann6" # this name is used during the optimization loop in Step 5
objective = f"-{metric_name}" # minimization is specified by the negative sign

client.configure_optimization(objective=objective)

for _ in range(10): # Run 10 rounds of trials
    # We will request three trials at a time in this example
    trials = client.get_next_trials(max_trials=num_trials)

    for trial_index, parameters in trials.items():
        x1 = parameters["x1"]
        x2 = parameters["x2"]
        x3 = parameters["x3"]
        x4 = parameters["x4"]
        x5 = parameters["x5"]
        x6 = parameters["x6"]

        result = solve(x1, x2, x3, x4, x5, x6)

        # Set raw_data as a dictionary with metric names as keys and results as values
        raw_data = {metric_name: result}

        # Complete the trial with the result
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)

        print(f"Completed trial {trial_index} with {raw_data=}")

best_parameters, prediction, index, name = client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)

# display=True instructs Ax to sort then render the resulting analyses
cards = client.compute_analyses(display=True)