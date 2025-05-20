from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
import json5
from multiprocessing import Process
from multiprocessing import Queue
import os
import psutil
import argparse

class TrialRunner:
    def __init__(self, config_path = "config\bayesian_opt.json5", parameters = None, trial_index = None):
        self.check_required_args(config_path=config_path, parameters=parameters, trial_index=trial_index) #checking and raise error if none

        with open(config_path, "r") as f:
            self.config = json5.load(f)["config_runner"] #config for trial runner (input config)
        
        from train import solve #import dynamically correct train set in config (still on it, not working)
        self.solver = solve

        self.cfg = argparse.Namespace(**parameters) #config injected into the trainning function (output config)
        self.index = trial_index
        self.set_cpu_affinity_and_priority()

    def check_required_args(self, **kwargs):
        for name, value in kwargs.items():
            if value is None:
                raise ValueError(f"❌ Argument '{name}' must be provided and not None.")

        
    def set_cpu_affinity_and_priority(self):
        proc = psutil.Process(os.getpid())
        
        # Fixe l'affinité CPU (ex: [0,1])
        proc.cpu_affinity([self.index % os.cpu_count()])
        
        # Priorité maximale
        try:
            proc.nice(psutil.HIGH_PRIORITY_CLASS) # Windows
        except AttributeError:
            proc.nice(-10) # Linux

    def run_trial(self):
        # Exemple simplifié (remplace par ton vrai entraînement)
        print(f"🏃 Running trial {self.index} with batch_size={self.cfg.batch_size}")
        master_val, follower_val, iteration, start_time = solve(cfg)
        score = 100 - self.cfg.lr_ac * 1000 + self.cfg.batch_size * 0.1
        return score














class AxBayesianOptimizer:
    def __init__(self, config_path = "config/bayesian_opt.json5"):
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.config = json5.load(f)

        self.metric_name = self.config["metrics_to_optimize"]
        self.client = Client()
        self._build_parameter_space() #stored in self.parameters
        
        self.client.configure_experiment(parameters=self.parameters)
        objective = f"-{self.metric_name}" if self.config.get("minimize", True) else self.metric_name
        self.client.configure_optimization(objective=objective)

    def _build_parameter_space(self):
        param_defs = self.config["parameters_space"]
        self.parameters = []
        for name, p in param_defs.items():
            self.parameters.append(
                RangeParameterConfig(
                    name=name,
                    parameter_type=p["type"],
                    bounds=tuple(p["bounds"]),
                    scaling=p.get("scaling", None)
                )
            )

    def run_trial_process(self, trial_index, parameters, queue):
        runner = TrialRunner(self.config_path, parameters, trial_index)
        result = runner.run_trial()
        queue.put((trial_index, result))

    def optimize(self):
        for _ in range(self.config["num_trials"]):
            trials = self.client.get_next_trials(max_trials=self.config["trials_per_batch"])
            result_queue = Queue()
            jobs = []

            for index, params in trials.items():
                self.run_trial_process(index, params, result_queue)
                print("finished running in AxBayesianOptimizer class")
                exit()
                p = Process(target=self.run_trial_process, args=(index, params, result_queue))
                p.start()
                jobs.append(p)

            for p in jobs:
                p.join()

            while not result_queue.empty():
                trial_index, result = result_queue.get()
                self.client.complete_trial(trial_index=trial_index, raw_data={self.metric_name: result})
                print(f"✔ Trial {trial_index} done with score {result}")

        best_parameters, prediction, *_ = self.client.get_best_parameterization()
        print("\n🏆 Best Parameters:", best_parameters)
        print("📈 Prediction (mean, var):", prediction)



if __name__ == '__main__':
    optimizer = AxBayesianOptimizer()
    optimizer.optimize()
    exit()

exit()






#Bayesian optimization

#configure ax client
client = Client()


#creating parameters config for AX from bayesian_opt main config 
parameter_defs = config["parameters_space"]
parameters = []

for name, p in parameter_defs.items(): #name pour le nom de la variable et p pour le dictionnaire associe
    parameters.append(
        RangeParameterConfig(
            name=name,
            parameter_type=p["type"],
            bounds=tuple(p["bounds"]),
            scaling=p["scaling"]
        )
    )

client.configure_experiment(parameters=parameters)

metric_name = config["metrics_to_optimize"] # this name is used during the optimization loop in Step 5
objective = f"-{metric_name}" # minimization is specified by the negative sign

client.configure_optimization(objective=objective)

for _ in range(config["num_trials"]):
    # We will request three trials at a time in this example
    trials = client.get_next_trials(max_trials=config["trials_per_batch"])
    jobs = []
    result_queue = Queue()

    for trial_index, parameters in trials.items():
        p = Process(target=run_trial, args=(trial_index, parameters, config["metrics_to_optimize"], client, result_queue))
        p.start()
        jobs.append(p)
        exit()

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




