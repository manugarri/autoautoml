from .awsbatch import run_awsbatch_job
from .docker import run_docker_job

PLATFORMS = {
    "docker": run_docker_job,
    "awsbatch": run_awsbatch_job
}


def run_job(job_settings):
    PLATFORMS[job_settings["platform"]](job_settings)
