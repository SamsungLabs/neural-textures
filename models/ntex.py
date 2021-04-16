import kornia
import torch
from torch import nn


class NeuralTex(torch.nn.Module):
    def __init__(self, texsegm, texchannels=16, texsize=512):
        super().__init__()

        self.texsize = texsize

        texsegm = texsegm.clone()
        self.texsegm = torch.nn.Parameter(texsegm.unsqueeze(2), requires_grad=False)
        N_CLASSES = self.texsegm.shape[1]

        self.neuraltexs = nn.ParameterList([])

        minimal_size = self.texsize // 2 ** 6
        for i in range(7):
            S = minimal_size * 2 ** i
            neuraltex = torch.randn(N_CLASSES, texchannels, S, S)
            neuraltex = torch.nn.Parameter(neuraltex.clone(), requires_grad=True)
            self.neuraltexs.append(neuraltex)

    def forward(self):
        neuraltexs = [kornia.geometry.transform.resize(neuraltex, (self.texsize, self.texsize)) for
                      neuraltex in self.neuraltexs]
        neuraltexs = torch.stack(neuraltexs, dim=0)

        if self.texsegm is not None:
            neuraltexs = neuraltexs * self.texsegm

        neuraltexs = neuraltexs.sum(1)
        return neuraltexs


class NeuralTexStack(torch.nn.Module):
    def __init__(self, n_people, texsegm, texchannels=16, texsize=512):
        super().__init__()

        self.textures = nn.ModuleList([])
        for i in range(n_people):
            ntex = NeuralTex(texsegm, texchannels, texsize)
            self.textures.append(ntex)
        self.pid2ntid = nn.ParameterDict()

    def load_state_dict_tex(self, state_dict, pids=None):
        if pids is not None:
            assert len(pids) <= len(self.textures), f'trying to load {len(pids)} textures into a stack of size {len(self.textures)}'

        for k in state_dict.keys():
            if pids is not None and k not in pids:
                continue
            ntex = self.get_texmodule(k)
            ntex.load_state_dict(state_dict[k])

    def get_texmodule(self, pid):
        if pid not in self.pid2ntid:
            if len(self.pid2ntid) == 0:
                next_tid = 0
            else:
                next_tid = len(self.pid2ntid.keys())
            self.pid2ntid[pid] = nn.Parameter(torch.LongTensor([next_tid]), requires_grad=False)
        return self.textures[self.pid2ntid[pid]]

    def move_to(self, pids, device):
        for pid in pids:
            ntex_module = self.get_texmodule(pid)
            ntex_module.to(device)

    def generate_batch(self, pids):
        ntexs = []
        for pid in pids:
            pid = str(pid)
            ntex_module = self.get_texmodule(pid)
            nt = ntex_module()
            ntexs.append(nt)
        ntexs = torch.stack(ntexs, dim=0)

        return ntexs
