"""
This script launches an autoautoml  job from a configuration file
"""
import argparse
import json
from pprint import pprint

from platforms import run_job


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run an autoautoml job.")
    parser.add_argument("job_settings_path",
                        help="Path to the job settings file")
    parser.add_argument("--containers", default=None,
                        help="ONLY run the specified containers, comma separated")

    args = parser.parse_args()
    return vars(args)


def main(job_settings_path, containers):
    with open(job_settings_path) as fname:
        job_settings = json.load(fname)
    if containers:
        print(f"RUNNING JOB ONLY ON CONTAINER {containers}")
        job_settings["containers"] = [c for c in job_settings["containers"] if c["name"] in containers.split(",")]
    print("\nCONFIGURATION FILE METADATA:\n")
    pprint(job_settings)
    run_job(job_settings)


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(**args)
