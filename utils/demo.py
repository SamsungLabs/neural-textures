import pickle

import numpy as np
import smplx
import torch
import os

from models.ntex import NeuralTexStack
from models.renderer import Renderer
from utils.bbox import get_ltrb_bbox, crop_resize_verts
from utils.common import get_rotation_matrix, to_sigm, itt
from utils.uv_renderer import UVRenderer


def rotate_verts(vertices, angle, K, K_inv, axis='y', mean_point=None):
    rot_mat = get_rotation_matrix(angle, axis)
    rot_mat = torch.FloatTensor(rot_mat).to(vertices.device).unsqueeze(0)

    vertices_world = torch.bmm(vertices, K_inv.transpose(1, 2))
    if mean_point is None:
        mean_point = vertices_world.mean(dim=1)

    vertices_rot = vertices_world - mean_point
    vertices_rot = torch.bmm(vertices_rot, rot_mat.transpose(1, 2))
    vertices_rot = vertices_rot + mean_point
    vertices_rot_cam = torch.bmm(vertices_rot, K.transpose(1, 2))

    return vertices_rot_cam, mean_point


def load_models(checkpoint_path='data/renderer.pth', ntex_path='data/ntex.pth', texsegm_path='data/texsegm.npy',
                pid_list=None, device='cuda:0'):
    model_cp = torch.load(checkpoint_path, map_location=device)
    ntex_cp = torch.load(ntex_path, map_location=device)
    texsegm = itt(np.load(texsegm_path)).unsqueeze(0)

    if 'generator' in model_cp:
        model_cp = model_cp['generator']

    if pid_list is None:
        pid_list = list(ntex_cp.keys())

    renderer = Renderer(18, 3, texsegm).to(device)
    renderer.load_state_dict(model_cp)
    renderer.eval()

    nstack = NeuralTexStack(len(pid_list), texsegm).to(device)
    nstack.load_state_dict_tex(ntex_cp, pids=pid_list)

    return renderer, nstack

def build_smplx_model_dict(smplx_model_dir, device):
    gender2filename = dict(neutral='SMPLX_NEUTRAL.pkl', male='SMPLX_MALE.pkl', female='SMPLX_FEMALE.pkl')
    gender2path = {k:os.path.join(smplx_model_dir, v) for (k, v) in gender2filename.items()}
    gender2model = {k:smplx.body_models.SMPLX(v).to(device) for (k, v) in gender2path.items()}

    return gender2model


class DemoInferer():

    def __init__(self, checkpoint_path, ntex_path, smplx_models_dir, imsize=512, pid_list=None, v_inds_path='data/v_inds.npy',
                 device='cuda:0'):
        self.smplx_models_dict = build_smplx_model_dict(smplx_models_dir, device)
        # smplx.body_models.SMPLX(smplx_model_path).to(device)
        self.renderer, self.nstack = load_models(checkpoint_path, ntex_path, pid_list=pid_list)
        self.v_inds = torch.LongTensor(np.load(v_inds_path)).to(device)
        self.imsize = imsize

        self.uv_renderer = UVRenderer(self.imsize, self.imsize).to(device)

        self.device = device

    def load_smplx(self, sample_path):
        with open(sample_path, 'rb') as f:
            smpl_params = pickle.load(f)
        gender = smpl_params['gender']

        for k, v in smpl_params.items():
            if type(v) == np.ndarray:
                smpl_params[k] = torch.FloatTensor(v).to(self.device)

        smpl_params['left_hand_pose'] = smpl_params['left_hand_pose'][:, :6]
        smpl_params['right_hand_pose'] = smpl_params['right_hand_pose'][:, :6]

        smpl_output = self.smplx_models_dict[gender](**smpl_params)
        vertices = smpl_output.vertices
        vertices = vertices[:, self.v_inds]
        K = smpl_params['camera_intrinsics'].unsqueeze(0)
        vertices = torch.bmm(vertices, K.transpose(1, 2))
        return vertices, K

    def crop_vertices(self, vertices, K):
        ltrb = get_ltrb_bbox(vertices)
        vertices, K = crop_resize_verts(vertices, K, ltrb, self.imsize)
        return vertices, K, ltrb

    def make_rgb(self, vertices, pid):
        uv = self.uv_renderer(vertices, negbg=False)

        ntexture = self.nstack.generate_batch([pid])
        ntexture = ntexture.sum(dim=1)
        nrender = torch.nn.functional.grid_sample(ntexture, uv.permute(0, 2, 3, 1), align_corners=True)
        renderer_input = dict(uv=uv, nrender=nrender)

        with torch.no_grad():
            renderer_output = self.renderer(renderer_input)

        fake_rgb = renderer_output['fake_rgb']
        fake_segm = renderer_output['fake_segm']
        fake_rgb = to_sigm(fake_rgb) * (fake_segm[:, :1] > 0.8)

        return fake_rgb

    def make_rotation_images(self, pid, n_rotimgs, smplx_path='data/smplx_sample.pkl'):
        vertices, K = self.load_smplx(smplx_path)
        vertices, K, ltrb = self.crop_vertices(vertices, K)

        K_inv = torch.inverse(K)

        rgb_frames = []
        for j in range(n_rotimgs):
            angle = np.pi * 2 * j / n_rotimgs
            verts_rot, mean_point = rotate_verts(vertices, angle, K, K_inv, axis='y')
            rgb = self.make_rgb(verts_rot, pid)
            rgb_frames.append(rgb)

        return rgb_frames
