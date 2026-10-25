"""
AI-GENERATED
Diagnostic script: checks whether the ConfigSpace<->Tensor conversion in
carps_wrapper.py distorts the actual coordinates.
Runs in the carps-env on the cluster (where the ConfigSpace/CARPS version is correctly installed).
"""
import torch
from run_carps_experiments import TASKS, build_objective_function
from carps.utils.task import Task
from carps.utils.task import InputSpace, OutputSpace, OptimizationResources, TaskMetadata
from carps_wrapper import LOOCVOptimizer
from carps.loggers.file_logger import FileLogger
from pathlib import Path

def main():
    Path("debug_output").mkdir(parents=True, exist_ok=True)
    logger = FileLogger(overwrite=True, directory=Path("debug_output"))
    spec = next(t for t in TASKS if t.name == "bbob_f12_i1_d2")
    objective_function = build_objective_function(spec, seed=0, logger=logger)

    task = Task(
        name=spec.name,
        objective_function=objective_function,
        input_space=InputSpace(configuration_space=objective_function.configspace),
        output_space=OutputSpace(),
        optimization_resources=OptimizationResources(n_trials=spec.n_trials),
        metadata=TaskMetadata(),
        seed=0,
    )

    optimizer = LOOCVOptimizer(task=task)
    optimizer._setup_optimizer()

    # Mehrere Testpunkte, um sicherzugehen (nicht nur einer, falls Zufall reinspielt)
    test_points = [
        torch.tensor([0.3, -0.7], dtype=torch.double),
        torch.tensor([-3.0, 2.5], dtype=torch.double),
        torch.tensor([0.0, 0.0], dtype=torch.double),
        torch.tensor([-5.0, 5.0], dtype=torch.double),  # Randwerte
    ]

    print(f"Bounds laut Optimizer: {optimizer.bounds}")
    print()

    for x_original in test_points:
        config = optimizer.convert_to_trial(x_original).config
        x_roundtrip = optimizer._config_to_tensor(config)

        diff = (x_original - x_roundtrip).abs().max().item()
        status = "OK" if diff < 1e-6 else "!! ABWEICHUNG !!"

        print(f"Original:  {x_original.tolist()}")
        print(f"Config:    {dict(config)}")
        print(f"Roundtrip: {x_roundtrip.tolist()}")
        print(f"Max diff:  {diff:.2e}  [{status}]")
        print()

        # Zusätzlich: die tatsächliche Funktionsauswertung vergleichen
        val_direct = objective_function._problem(x_original.numpy().tolist())
        trial_info = optimizer.convert_to_trial(x_original)
        trial_value = objective_function.evaluate(trial_info=trial_info)
        print(f"f(x) direkt via ioh:       {val_direct}")
        print(f"f(x) via CARPS-Pipeline:   {trial_value.cost}")
        print(f"  -> {'OK' if abs(val_direct - trial_value.cost) < 1e-6 else '!! WEICHT AB !!'}")
        print("=" * 60)


if __name__ == "__main__":
    main()