import click
import math
import csv
from pathlib import Path
import numpy as np


@click.command()
@click.option('--infile', 'infile', required=True,
              type=click.Path(exists=True),
              help='Path to input CSV containing eval scores')
def main(
    infile: Path,
):
    """ Combine all evaluation scores in the input file.

    Expects the usual CSV format:
        metric1,metric2
        score,score
        score,score
    """

    scores = {}
    with open(infile, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')

        for row in reader:
            # Drop the seed
            if "seed" in row:
                del row["seed"]
            for metric, val in row.items():
                scores.setdefault(metric, []).append(float(val))

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
