"""Optional experiment logging. Disabled unless `experiment.backend: comet`.

Comet logging also needs `pip install -e ".[comet]"` and the COMET_API_KEY
environment variable (optionally COMET_WORKSPACE and COMET_PROJECT_NAME), for
example from a local .env file that stays out of version control.
"""

import os


class _CometLogger:
    def __init__(self, cfg):
        from comet_ml import Experiment
        exp_cfg = cfg.get("experiment", {})
        self.exp = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name=os.getenv("COMET_PROJECT_NAME", exp_cfg.get("project", "vedje")),
            workspace=os.getenv("COMET_WORKSPACE"),
        )
        if exp_cfg.get("run_name"):
            self.exp.set_name(exp_cfg["run_name"])
        self.exp.log_parameters({k: v for k, v in cfg.items() if not isinstance(v, dict)})

    def log_metric(self, name, value, step=None):
        self.exp.log_metric(name, value, step=step)

    def log_checkpoint(self, path, epoch=None):
        self.exp.log_parameter(f"checkpoint.{epoch}.path", os.path.abspath(path))

    def end(self):
        self.exp.end()


def ExperimentLogger(cfg):
    """Return a logger, or None when logging is disabled."""
    backend = cfg.get("experiment", {}).get("backend", "none").lower()
    if backend == "comet" and os.getenv("COMET_API_KEY"):
        return _CometLogger(cfg)
    return None
