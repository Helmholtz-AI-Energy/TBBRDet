import click
import logging
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
# import matplotlib.pyplot as plt

'''
This mergers all image layer (RGB, Thermal, Depth) into a single combined image.

Originally RGB images were 4000*3000, thermal images were 640*512
The images that are input to this script are all 4000*3000 (width*height)

'''

_logger = logging.getLogger(__name__)


@click.command()
@click.option('--rgb', 'rgb_dir', required=True,
              type=click.Path(exists=True), help='Directory containing RGB images')
@click.option('--thermal', 'thermal_dir', required=True,
              type=click.Path(exists=True), help='Directory containing Thermal images')
@click.option('--depth', 'depth_dir', required=True,
              type=click.Path(exists=True), help='Directory containing Depth images')
@click.option('-o', '--output', 'output_dir', required=True,
              type=click.Path(), help='Directory to write out merged images')
@click.option('-j', '--jobs', 'njobs', required=False,
              type=int, help='Number of parallel jobs to run')
@click.option('-f', '--force', 'force', is_flag=True)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(
    rgb_dir: Path,
    thermal_dir: Path,
    depth_dir: Path,
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
        _logger.exception("Output directory exists, use -f/--force flag to allow overwriting.")
        raise

    # Find the filenames, we use those from thermal images
    # These all have a number one less than their RGB/Depth conterparts
    filetypes = (".jpg", ".JPG")
    T_image_names = [f for t in filetypes for f in Path(thermal_dir).glob(f"*{t}")]

    assert len(T_image_names) > 0, _logger.error(f"No images found in {thermal_dir}")

    Parallel(n_jobs=njobs)(
        delayed(merge_single_image)(image, rgb_dir, depth_dir, output_dir) for image in tqdm(T_image_names)
    )


def merge_single_image(image: Path, rgb_dir: Path, depth_dir: Path, output_dir: Path, filetypes: tuple = (".jpg", ".JPG")):
    ''' Function to merge a single image '''

    # Set params for resizing/cropping, we need all images to be the same size before merging
    # The large ones are to resize everything to the RGB size, then crop.
    large_width = 4000
    large_height = 3000
    x = 300   # 360   # 0
    y = 150   # 160   # 0
    h = 2680  # 2400  # 3000
    w = 3370  # 3200  # 4000

    # Name format is DJI_XXXX_R.JPG
    filename = image.stem
    _logger.debug(f"Processing image {filename}")

    # Set name for corresponding depth/RGB image
    image_ID = int(filename.split("_")[1])

    rgb_image_name = [f for t in filetypes for f in Path(rgb_dir).glob(f"*{(image_ID + 1):04}*{t}")]
    _logger.debug(f"RGB image: {rgb_image_name}")

    depth_image_name = [f for t in filetypes for f in Path(depth_dir).glob(f"*{(image_ID + 1):04}*{t}")]
    _logger.debug(f"Depth image: {depth_image_name}")

    # Confirm they exist before loading anythin
    if len(rgb_image_name) < 1:
        _logger.warn(f"No matching RGB image for {filename} found, skipping...")
    elif len(depth_image_name) < 1:
        _logger.warn(f"No matching Depth image for {filename} found, skipping...")

    # And catch the cases that more than one matching image is found
    assert len(rgb_image_name) == 1, f"More than one RGB image for {filename} found, this will cause unintended behaviour"
    assert len(depth_image_name) == 1, f"More than one Depth image for {filename} found, this will cause unintended behaviour"

    # Load the first image found (really should only be one
    # Load Thermal image
    ims = {
        "Thermal": cv2.imread(str(image))[:, :, :1],
        "RGB": cv2.imread(str(rgb_image_name[0])),
        "Depth": cv2.imread(str(depth_image_name[0]))[:, :, :1],
    }
    _logger.debug(
        f"All layers loaded successfully, shapes (T,RGB,D): {ims['Thermal'].shape} {ims['RGB'].shape} {ims['Depth'].shape}"
    )

    # Resize everything to the same shape and crop
    for key, val in ims.items():
        ims[key] = cv2.resize(val, (large_width, large_height))[y:(y + h), x:(x + w)].reshape((h, w, -1))
    _logger.debug(
        f"All layers resized/cropped, shapes (T,RGB,D): {ims['Thermal'].shape} {ims['RGB'].shape} {ims['Depth'].shape}"
    )

    # Merge all layers
    # We only drop the extra layers in T and D here because otherwise cv2 resize makes (w, h, 1) -> (w, h)
    output_im = np.concatenate(
        (ims["RGB"], ims["Thermal"], ims["Depth"]),
        axis=2
    )
    _logger.debug(
        f"All layers merged as (RGBTD), shape: {output_im.shape}"
    )

    # Write out using thermal layer's name as they were used to create the annotations
    output_path = Path(output_dir, f"{filename}.npy")
    np.save(output_path, output_im)
    _logger.debug(f"Merged layers saved as {output_path}")

    return


if __name__ == '__main__':
    main()
