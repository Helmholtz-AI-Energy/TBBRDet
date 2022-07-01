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

    This assumes a stucture of <topdir>/<jobid>/evalulation/*.json
    The latest json file for each job will be used.
    mAP copypaste labels are hardcoded as: ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l')
    """

    results = {}
    for job_dir in Path(top_dir).glob("*"):
        # Get the latest evaluation
        results_file = list(Path(job_dir, 'evaluation').glob("*.json"))[-1]

        # Load the JSON file
        # We want the metric key from this
        with open(results_file) as json_file:
            results[Path(job_dir).stem] = json.load(json_file)['metric']

    # Now collate the scores
    scores = {}
    for j in results.values():
        for metric, val in j.items():

            # Some scores are actually lists
            if isinstance(val, float):
                scores.setdefault(metric, []).append(val)
            else:
                bbox_or_segm = metric[:4]

                for idx, score in enumerate(val.split(' ')):
                    scores.setdefault(f"{bbox_or_segm}_{mAP_labels[idx]}", []).append(float(score))

    # Finally, print the mean and std_dev of each score
    # TODO: Make it print the latex for me
    for metric, score in scores.items():
        # Regular formatting
        # print(f"{metric:<20}{np.mean(score):.3f} {np.std(score):.3f}")
        # Latex siunitx numbers
        mean = np.mean(score)
        std_dev = np.std(score)
        # Find the first non-zero element of std_dev
        sig_fig = -int(math.log10(abs(std_dev))) if std_dev > 0. else 0
        # print(sig_fig)
        print(f"{metric:<20}$\\num{{{mean:.{sig_fig+1}f} \\pm {std_dev:.{sig_fig+1}f}}}$         ")

    return


if __name__ == '__main__':
    main()
