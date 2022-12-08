import click
import math
from pathlib import Path
import json
import numpy as np

mAP_labels = ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l')


@click.command()
@click.option('--topdir', 'top_dir', required=True,
              type=click.Path(exists=True), help='Top-level directory containing jobs and evaluations')
def main(
    top_dir: Path,
):
    """ Combine all evaluation scores in the directory to produce a mean+-std.

    This assumes a stucture of <topdir>/<seed>/metrics.json
    The latest json file for each job will be used.
    mAP copypaste labels are hardcoded as: ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l')
    """

    results = {}
    for job_dir in Path(top_dir).glob("*"):
        # Get the latest evaluation
        results_file = Path(job_dir, 'metrics.json')

        # Load the JSON file
        # This is a really hackish way to get the last line
        # Assuming file is short and metrics are in last line
        for line in open(results_file):
            results[Path(job_dir).stem] = json.loads(line)

    # Now collate the scores
    scores = {}
    for j in results.values():
        for metric, val in j.items():

            scores.setdefault(metric, []).append(val)

    # Finally, print the mean and std_dev of each score
    # TODO: Make it print the latex for me
    for metric, score in scores.items():
        # Regular formatting
        # print(f"{metric:<20}{np.mean(score):.3f} {np.std(score):.3f}")
        # Latex siunitx numbers
        mean = np.mean(score)
        std_dev = np.std(score)
        # print(metric, mean, std_dev)
        # Find the first non-zero element of std_dev
        sig_fig = -int(math.log10(abs(std_dev))) if std_dev > 0. else 0
        # print(sig_fig)
        print(f"{metric:<20}$\\num{{{mean:.{sig_fig+1}f} \\pm {std_dev:.{sig_fig+1}f}}}$         ")

    return


if __name__ == '__main__':
    main()
