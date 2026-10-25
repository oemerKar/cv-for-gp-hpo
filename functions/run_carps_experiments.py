#How to use it: python run_carps_experiments.py --array-index $SLURM_ARRAY_TASK_ID
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

from carps.loggers.file_logger import FileLogger
from carps.utils.task import InputSpace, OptimizationResources, OutputSpace, Task, TaskMetadata

from carps_wrapper import LOOCVOptimizer, MCMCOptimizer, MLLOptimizer

OPTIMIZERS = {
    "mll": MLLOptimizer,
    "loocv": LOOCVOptimizer,
    "mcmc": MCMCOptimizer,
}

N_INITIAL = 10
SEEDS = list(range(5))
OUTPUT_ROOT = Path("results")


# Task Definitions; Copied out of the CARPS Subset

@dataclass
class TaskSpec:
    category: str  # "bbob" or "hpobench" or "yahpo"
    name: str
    n_trials: int
    kwargs: dict = field(default_factory=dict)


TASKS = [
    #BBOB(has to be done in own env (carps-bbob-env), because ioh needs numpy > 2.0 which is incompatible with YAHPO
    TaskSpec(category="bbob", name="bbob_f12_i0_d2", n_trials=77,
             kwargs=dict(fid=12, instance=0, dimension=2)),
    TaskSpec(category="bbob", name="bbob_f12_i1_d2", n_trials=77,
             kwargs=dict(fid=12, instance=1, dimension=2)),
    TaskSpec(category="bbob", name="bbob_f20_i0_d2", n_trials=77,
             kwargs=dict(fid=20, instance=0, dimension=2)),
    TaskSpec(category="bbob", name="bbob_f6_i1_d4", n_trials=100,
             kwargs=dict(fid=6, instance=1, dimension=4)),

    #HPOBench tabular ML (3/5)
    TaskSpec(category="hpobench", name="hpobench_lr_146818", n_trials=77,
             kwargs=dict(model="lr", task_id=146818, metric=["function_value"])),
    TaskSpec(category="hpobench", name="hpobench_rf_146212", n_trials=100,
             kwargs=dict(model="rf", task_id=146212, metric=["function_value"])),
    TaskSpec(category="hpobench", name="hpobench_xgboost_146212", n_trials=100,
             kwargs=dict(model="xgboost", task_id=146212, metric=["function_value"])),
    # 2/5 HPOBench-NAS-Tasks are missing as Singularity is needed, and does not exist on the cluster

    #YAHPO:
    TaskSpec(category="yahpo", name="yahpo_lcbench_168335", n_trials=126,
             kwargs=dict(bench="lcbench", instance="168335", metric=["val_accuracy"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_aknn_1462", n_trials=118,
             kwargs=dict(bench="rbv2_aknn", instance="1462", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_aknn_312", n_trials=118,
             kwargs=dict(bench="rbv2_aknn", instance="312", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_aknn_40498", n_trials=118,
             kwargs=dict(bench="rbv2_aknn", instance="40498", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_aknn_458", n_trials=118,
             kwargs=dict(bench="rbv2_aknn", instance="458", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_glmnet_41157", n_trials=90,
             kwargs=dict(bench="rbv2_glmnet", instance="41157", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_ranger_40927", n_trials=134,
             kwargs=dict(bench="rbv2_ranger", instance="40927", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_svm_24", n_trials=118,
             kwargs=dict(bench="rbv2_svm", instance="24", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_svm_182", n_trials=118,
             kwargs=dict(bench="rbv2_svm", instance="182", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_xgboost_42", n_trials=170,
             kwargs=dict(bench="rbv2_xgboost", instance="42", metric=["acc"])),
    TaskSpec(category="yahpo", name="yahpo_rbv2_xgboost_23512", n_trials=170,
             kwargs=dict(bench="rbv2_xgboost", instance="23512", metric=["acc"])),
]


def build_objective_function(spec: TaskSpec, seed: int, logger: FileLogger):
    """
    Builds an objective function for the given Taskspec, seed and logger, depending on the category of the spec
    """
    if spec.category == "bbob":
        from carps.objective_functions.bbob import BBOBObjectiveFunction
        return BBOBObjectiveFunction(seed=seed, loggers=[logger], **spec.kwargs)
    
    if spec.category == "hpobench":
        from carps.objective_functions.hpo_bench import HPOBenchObjectiveFunction
        return HPOBenchObjectiveFunction(seed=seed, loggers=[logger], **spec.kwargs)
    
    if spec.category == "yahpo":
        from carps.objective_functions.yahpo import YahpoObjectiveFunction
        return YahpoObjectiveFunction(seed=seed, loggers=[logger], **spec.kwargs)
    raise ValueError(f"Unknown category: {spec.category}")

# Combines task, with optimizer and seed
@dataclass
class RunSpec:
    optimizer_name: str
    task_spec: TaskSpec
    seed: int

    @property
    def run_name(self) -> str:
        return f"{self.optimizer_name}_{self.task_spec.name}_seed{self.seed}"


def build_experiment_grid(tasks = None) -> list[RunSpec]:
    """
    Builds a grid out of the tasks, optimizers and seeds and returns the list of RunSpecs for each of these runs
    If no specific tasks are given, all tasks are used
    """
    tasks = tasks if tasks is not None else TASKS
    grid = itertools.product(OPTIMIZERS.keys(), tasks, SEEDS)
    return [
        RunSpec(optimizer_name=opt, task_spec=task, seed=seed)
        for opt, task, seed in grid
    ]


def run_single(spec: RunSpec, output_root: Path = OUTPUT_ROOT) -> None:
    """
    Runs the single Task with the optimizer and the seed out of runSpec and saves it at the output_root
    """
    run_dir = output_root / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = FileLogger(overwrite=True, directory=run_dir)

    # Write metadata seperately
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "optimizer_id": spec.optimizer_name,
                "task_id": spec.task_spec.name,
                "task_category": spec.task_spec.category,
                "seed": spec.seed,
            },
            f,
        )

    objective_function = build_objective_function(spec.task_spec, seed=spec.seed, logger=logger)

    task = Task(
        name=spec.task_spec.name,
        objective_function=objective_function,
        input_space=InputSpace(configuration_space=objective_function.configspace),
        output_space=OutputSpace(),
        optimization_resources=OptimizationResources(n_trials=spec.task_spec.n_trials),
        metadata=TaskMetadata(),
        seed=spec.seed,
    )

    OptimizerCls = OPTIMIZERS[spec.optimizer_name]
    optimizer = OptimizerCls(
        task=task,
        loggers=[logger],
        n_initial=N_INITIAL,
        num_loop=spec.task_spec.n_trials,
    )

    print(f"[{spec.run_name}] Start...")
    incumbent = optimizer.run()
    _, trial_value = incumbent
    print(f"[{spec.run_name}] Done. Best cost: {trial_value.cost:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--category", type= str, default= None)
    parser.add_argument("--include-bbob", action="store_true")
    args = parser.parse_args()

    if args.category is not None:
        tasks_to_run = [t for t in TASKS if t.category == args.category]
    elif not args.include_bbob:
        tasks_to_run = [t for t in TASKS if t.category != "bbob"]
    else:
        tasks_to_run = TASKS
    
    grid = build_experiment_grid(tasks_to_run)
    output_root = Path(args.output_root)

    if args.array_index is not None:
        run_single(grid[args.array_index], output_root=output_root)
    elif args.local:
        print(f"{len(grid)} Runs, (local, serial).")
        for spec in grid:
            run_single(spec, output_root=output_root)
    else:
        print(f"{len(grid)} Runs in TASKS x {list(OPTIMIZERS.keys())} x {len(SEEDS)} Seeds.")
        print("SLURM: --array-index $SLURM_ARRAY_TASK_ID, --array=0-" + str(len(grid) - 1))
        print("Lokal: --local")


if __name__ == "__main__":
    main()