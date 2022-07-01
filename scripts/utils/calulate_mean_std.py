from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description='''Calculate pixel mean and std for all images.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', type=Path, required=True,
                        help="Directory containing image subdirs", metavar="IMG_DIR",
                        dest='img_dir')
    parser.add_argument('-n', type=int, required=False,
                        help="Number of images to process", metavar="NUM",
                        dest='num')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    mean = None
    std = None
    file_list = [f for f in args.img_dir.glob('**/*') if f.is_file()]
    if args.num:
        file_list = file_list[:args.num]

    for f in tqdm(file_list):

        img = np.load(f)

        if mean is None:
            mean = img.mean((0, 1))
            std = img.std((0, 1))
        else:
            mean += img.mean((0, 1))
            std += img.std((0, 1))

    mean /= len(list(file_list))
    std /= len((file_list))

    print(f'Pixel means: {mean}')
    print(f'Pixel stds: {std}')
