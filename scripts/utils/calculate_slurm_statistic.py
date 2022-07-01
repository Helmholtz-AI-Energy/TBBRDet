import click
import os
import numpy as np
import math
from pathlib import Path
import time
from datetime import datetime


@click.command()
@click.option(
    '--topdir',
    'top_dir',
    required=False,
    type=click.Path(exists=True),
    help='Top-level directory containing jobIDs',
)
@click.option(
    '-j',
    '--job-ids',
    'job_ids',
    required=False,
    type=str,
    multiple=True,
    help='List of job IDs to calculate statistic of',
)
@click.option(
    '-s',
    '--statistic',
    'statistic',
    required=False,
    type=str,
    default='ConsumedEnergy',
    help='Job statistic to lookup, must produce a number',
)
def main(
    top_dir: Path,
    job_ids: list,
    statistic: str,
):
    """ Read the jobIDs from the directory, lookup the statistic and disply it.

    This assumes a stucture of <topdir>/<experiment>/<jobid>
    """

    if top_dir is not None:
        # Iterate through experiments
        for exp in Path(top_dir).glob("*"):

            print(exp.name)
            job_ids = [p.name for p in Path(exp).glob("*")]
            calculate_job_statistic(job_ids, statistic)
    elif job_ids:
        calculate_job_statistic(job_ids, statistic)
    else:
        raise RuntimeError("At least one of topdir or job-ids must be set.")

    return


def calculate_job_statistic(job_ids: list, statistic: str):

    # print(",".join(job_ids))
    # print(f"{lookup_cmd} {','.join(job_ids)}")

    lookup_cmd = f"sacct -XPn --noconvert --format {statistic} -j"

    stream = os.popen(f"{lookup_cmd} {','.join(job_ids)}")
    # This will give us the output as a list of strings
    output = stream.read().splitlines()

    # Need to convert times to total minutes if necessary
    is_time = isTimeFormat(output[0])

    if is_time:
        # Convert to seconds
        results = [convert_to_seconds(e) for e in output]
    else:
        results = [int(e) for e in output]

    scaling = 1
    if statistic == "ConsumedEnergy":
        scaling = 1e6
    elif statistic == "Elapsed":
        # Use minutes, easier to interpret
        scaling = 60

    mean = np.mean(results) / scaling
    std = np.std(results) / scaling
    sig_fig = -int(math.log10(abs(std))) if std > 0. else 0
    print(f"$\\num{{{mean:.{sig_fig+1}f} \\pm {std:.{sig_fig+1}f}}}$\n")


def convert_to_seconds(input):
    ''' Convert given time to seconds'''
    td = datetime.strptime(input, '%H:%M:%S') - datetime(1900, 1, 1)
    return td.total_seconds()


def isTimeFormat(input):
    ''' Crude check for time format '''
    try:
        time.strptime(input, '%H:%M:%S')
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()
