import argparse
import os

import cv2
import numpy as np

from utils.common import tti
from utils.demo import DemoInferer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint')
    parser.add_argument('--smplx_model_path', type=str, default='data/SMPLX_NEUTRAL.pkl')
    parser.add_argument('--out_path', type=str, default='data/results/')
    parser.add_argument('--person_id', type=str)
    parser.add_argument('--smplx_dict_path', type=str)
    parser.add_argument('--n_rotimgs', type=int, default=8)
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    renderer_path = os.path.join(args.checkpoint_path, 'renderer.pth')
    ntex_path = os.path.join(args.checkpoint_path, 'ntex.pth')
    inferer = DemoInferer(renderer_path, ntex_path, smplx_model_path=args.smplx_model_path, imsize=args.imsize,
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
