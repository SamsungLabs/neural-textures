import argparse
import os

import cv2
import numpy as np

from utils.common import tti
from utils.demo import DemoInferer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint', help='Path to model checkpoint')
    parser.add_argument('--smplx_models_dir', type=str, default='data/smplx', help='Path to smplx models')
    parser.add_argument('--out_path', type=str, default='data/results/', help='Path to a directory to save rendered images in')
    parser.add_argument('--person_id', type=str, help='id of a person from AzurePeople dataset')
    parser.add_argument('--smplx_dict_path', type=str, help='Path to a .pkl file with smplx parameters')
    parser.add_argument('--n_rotimgs', type=int, default=8, help='Number of rotation steps to render textured model in')
    parser.add_argument('--imsize', type=int, default=512, help='Resolution in which to render images (512 recommended)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run rendering process on')
    args = parser.parse_args()

    renderer_path = os.path.join(args.checkpoint_path, 'renderer.pth')
    ntex_path = os.path.join(args.checkpoint_path, 'ntex.pth')
    inferer = DemoInferer(renderer_path, ntex_path, smplx_models_dir=args.smplx_models_dir, imsize=args.imsize,
                          pid_list=[args.person_id])

    assert args.n_rotimgs > 0, 'no image to be saved'

    if args.smplx_dict_path is None:
        smplx_dict_path = f'data/smplx_dicts/{args.person_id}.pkl'
    else:
        smplx_dict_path = args.smplx_dict_path

    rot_images = inferer.make_rotation_images(args.person_id, args.n_rotimgs, smplx_path=smplx_dict_path)

    out_dir = os.path.join(args.out_path, args.person_id)
    os.makedirs(out_dir, exist_ok=True)

    for j, rgb in enumerate(rot_images):
        rgb = tti(rgb)
        rgb = (rgb * 255).astype(np.uint8)

        rgb_out_path = os.path.join(out_dir, f"{j:04d}.png")
        os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
        cv2.imwrite(rgb_out_path, rgb[..., ::-1])
