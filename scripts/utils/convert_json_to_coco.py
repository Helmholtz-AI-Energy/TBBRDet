import json
import torch
from pathlib import Path
import click
import logging
import sys

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode


# Specify the category_id to match with the class.
labels = {
    'Generic Building Component': ['Surface', 'Junction', 'Window', 'Cantilever', 'Technique'],
}


_logger = logging.getLogger(__name__)


@click.command()
@click.option('-j', 'json_dir', required=True,
              type=click.Path(exists=True), help='Dir containing JSON annotation files to merge and convert into COCO, only those with a matching image subdir will be used.')
@click.option('-i', '--images', 'img_dir', required=True,
              type=click.Path(exists=True), help='Directory containing image_subdirs, must be named XXX for each JSON file XXX_json.json')
# @click.option('-l', '--label', 'label', required=True,
#               type=click.Path(exists=True), help='Directory containing Depth images')
@click.option('-o', '--output', 'output_filename', required=True,
              type=click.Path(), help='Output COCO annotations filename')
@click.option('-f', '--force', 'force', is_flag=True)
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
def main(
    json_dir: Path,
    img_dir: Path,
    output_filename: Path,
    force: bool,
    log_level: int,
):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    DatasetCatalog.register(
        'thermals',
        lambda: get_all_flights(Path(img_dir), Path(json_dir)),
    )

    # Leaving commented out, can specify which label to use
    # Leaving it as one class for now
    # if args.label:
    #     MetadataCatalog.get('thermals').thing_classes = labels[args.label]
    # else:
    MetadataCatalog.get('thermals').thing_classes = ['Thermal bridge']

    _logger.info('All JSON files loaded, converting to COCO JSON')

    convert_to_coco_json('thermals', str(output_filename), allow_cached=(not force))

    return


def get_all_flights(img_dir: Path, json_dir: Path) -> list:
    ''' path is train or test directory '''
    dataset_dicts = []
    # Step through all directories in path
    for flight in img_dir.glob('*'):
        # Skip direct files
        if flight.is_file():
            _logger.info(f'File {flight} is not a directory, skipping...')
            continue

        _logger.info(f'Loading {flight}')

        flight_dicts = get_thermal_dicts(
            json_path=json_dir / f'{flight.stem}_json.json',
            img_path=flight,
            starting_image_id=len(dataset_dicts),
        )
        dataset_dicts.extend(flight_dicts)
        _logger.debug("Flight successfully added")

    return dataset_dicts


def get_thermal_dicts(json_path: Path, img_path: Path, label=None, starting_image_id=0):
    ''' This function loads the JSON file created with the VGG annotator and returns the annotations within.

    Args
    ----
    json_path: str or Path object
        Location of JSON annotations file to load
    img_path: str or Path object
        Directory containing corresponding images
    label: str
        Name of the annotation label to use
    verbose: bool, optional
        Whether to print cropped annotation warnings

    Returns
    ----
    dict: Dictionary containing file annotations in form {filename: [{bbox: tensor(N, 4), segmentation: [[]], category_id: tensor(N)}]}
    '''

    # Load the JSON file
    with open(json_path) as f:
        imgs_anns = json.load(f)
    _logger.info(f"JSON file {json_path} successfully loaded")

    dataset_dicts = []
    skipped = 0
    # loop through the entries in the JSON file
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # add file_name, image_id, height and width information to the records
        # NOTE: We drop the extension since it varies between annotations (jpg) and images (npy)
        filename = Path(v['filename']).stem
        filename = img_path / f'{filename}.npy'

        x = 300
        y = 150
        h = 2680
        w = 3370

        # Check the image actually exists
        if not filename.exists():
            _logger.info(f'Image file missing, skipping: {filename}')
            skipped += 1
            continue

        record["file_name"] = str(Path(*filename.parts[-2:]))
        record["image_id"] = starting_image_id + idx
        record["height"] = h
        record["width"] = w

        regions = v["regions"]

        objs = []
        # One image can have multiple annotations, therefore this loop is needed
        for region in regions:
            # reformat the polygon information to fit the specifications
            anno = region['shape_attributes']
            # Need to handle each shape differently
            if anno['name'] == 'rect':
                # Coords are top left corner of rectangle
                x = anno['x']
                y = anno['y']
                w = anno['width']
                h = anno['height']
                px = [x, x + w, x + w, x]
                py = [y, y, y - h, y - h]
            elif anno['name'] == 'polyline':
                px = anno["all_points_x"]
                py = anno["all_points_y"]
            else:
                raise KeyError(f'Annotation shape {anno["name"]} not supported')

            poly = torch.ones((3, 1, len(px)))
            poly[0] = torch.tensor(px)
            poly[1] = torch.tensor(py)
            # Perform the crop on polygon coordinates and shift
            # With transform before clamp
            poly[0] = torch.clamp(poly[0] - x, 0, w)
            poly[1] = torch.clamp(poly[1] - y, 0, h)

            segmentation = poly[:2].T.reshape(-1)  # Now it's (x1, y1, x2, y2, ...)

            # Before saving the region we need to drop it if it's been cropped out
            # If it has it will have zero size in at least on dimension of the bounding box
            bbox = torch.tensor([poly[0].min(), poly[1].min(), poly[0].max(), poly[1].max()], dtype=torch.float)
            if (bbox[0] == bbox[2]) or (bbox[1] == bbox[3]) or (poly.shape[-1] == 0):
                continue

            if label is not None:
                region_attribute = region['region_attributes'][label]

                # NOTE: The background is considered index of num_classes, no one will tell you this great secret...
                category_id = labels[label].index(region_attribute)
            else:
                # This is a case for just one label
                category_id = 0

            obj = {
                "bbox": bbox.tolist(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [segmentation.tolist()],
                "category_id": category_id,
                # "iscrowd": 0,
            }
            objs.append(obj)

        _logger.debug(f"All image objects registered and cropped: {objs}")

        # Also final check that there are any annotations left after cropping
        if len(objs) == 0:
            _logger.info(f'All annotations were cropped, skipping image {filename}')
            skipped += 1
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    _logger.info(f'Skipped {skipped}/{len(imgs_anns)} files in directory {img_path}')
    return dataset_dicts


if __name__ == '__main__':
    main()
