import click
import logging
import sys
import csv
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
# import matplotlib.pyplot as plt

'''
This registers RGB images onto Thermal images via homography.
RGB images are 4000*3000, thermal images are 640*512.
Combining the images comes in a later step
'''

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-i', '--input', 'input_dir', required=True,
              type=click.Path(exists=True), help='Directory containing input images')
@click.option('-s', '--source', 'source', required=True,
              type=click.Path(exists=True), help='Path to file of source points')
@click.option('-d', '--dest', 'dest', required=True,
              type=click.Path(exists=True), help='Path to file of destination points')
@click.option('-o', '--output', 'output_dir', required=True,
              type=click.Path(), help='Directory to write out aligned images')
@click.option('-j', '--jobs', 'njobs', required=False,
              type=int, help='Number of parallel jobs to run')
@click.option('-f', '--force', 'force', is_flag=True)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(
    input_dir: Path,
    source: Path,
    dest: Path,
    output_dir: Path,
    njobs: int,
    force: bool,
    log_level: int,
):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set up output directory
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=force)
    except FileExistsError:
        print("Output directory exists, use -f/--force flag to allow overwriting.")

    # Load source points
    with open(source, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        src_pts = np.array(list(reader)).astype(np.float)
    # Load destincation points
    with open(dest, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        dst_pts = np.array(list(reader)).astype(np.float)

    # Destination points are offset by 4000 on the x-axes
    dst_pts[:, 0] = dst_pts[:, 0] - 4000

    assert src_pts.shape == dst_pts.shape, f"source ({src_pts.shape}) and dest ({dst_pts.shape}) shapes must match"
    # print(src_pts, dst_pts)

    # Calculate Homography
    hom, status = cv2.findHomography(src_pts, dst_pts)

    filetypes = ("*.jpg", "*.JPG")
    images = [f for t in filetypes for f in Path(input_dir).glob(t)]
    # for image in tqdm(images)
    Parallel(n_jobs=njobs)(
        delayed(align_single_image)(image, hom, output_dir) for image in tqdm(images)
    )


def align_single_image(image: Path, homography: np.ndarray, output_dir: Path, filetypes: tuple = (".jpg", ".JPG")):
    ''' Align one single image '''
    filename = image.stem
    # print(f"Processing {filename}")

    # Read source image, uncomment top line to resize on load
    # src_im = cv2.resize(cv2.imread(str(image)), (4000, 3000))
    src_im = cv2.imread(str(image))
    # print(hom)

    # Warp source image to destination based on homography
    out_im = cv2.warpPerspective(
        src_im,
        homography,
        (src_im.shape[1], src_im.shape[0])
    )

    cv2.imwrite(str(Path(output_dir) / f"{filename}.jpg"), out_im)

    return


if __name__ == '__main__':
    main()
