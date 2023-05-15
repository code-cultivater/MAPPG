#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/2 16:51
# @Author  : Wubing Chen
# @Site    : 
# @File    : transition_buffer.py
# @Software: PyCharm

import torch
import torch as th
import numpy as np
from types import SimpleNamespace as SN

class TransationBatch:
    def __init__(self,args,batch_size,scheme,device):
        self.args=args
        self.batch_size = batch_size
        self.scheme = scheme.copy()
    #     self.scheme={
    #     "state": {"vshape": env_info["state_shape"]},
    #     "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
    #     "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
    #     "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
    #     "reward": {"vshape": (1,)},
    #     "terminated": {"vshape": (1,), "dtype": th.uint8},
    # }
        self.data = SN()
        self.data.transition_data = {}

        self.store_device="cpu"
        self.run_device=device
        self._setup_data(self.scheme, self.groups, batch_size)
    def _setup_data(self, scheme, groups, batch_size):
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype", th.float32)
            if isinstance(vshape, int):
                vshape = (vshape,)
            shape = vshape
            self.data.transition_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype,
                                                                 device=self.store_device)

    def update(self, data, bs=slice(None)):
        slices = self._parse_slices((bs))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                _slices = slices
            else:
                raise KeyError("{} not found in transition  data".format(k))
            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.store_device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.transition_data:
                return self.data.transition_data[item].to(self.run_device)
            else:
                raise ValueError
    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1
    def _parse_slices(self, items):
        parsed=[]
        if (isinstance(items, slice)  # slice a:b
                or isinstance(items, int)  # int i
                or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
        ):
            print("transation buffer line 52")
        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def __repr__(self):
        return "TransitionBatch. Batch Size:{}  Keys:{} ".format(self.batch_size,
                                                                                     self.scheme.keys())
class ReplayBuffer(TransationBatch):
    def __init__(self, args,buffer_size,scheme,device):
        super(ReplayBuffer, self).__init__(args,buffer_size,scheme,device)
        self.args=args
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.transation_in_buffer = 0

    def insert_transition_batch(self, trans_batch):
        if self.buffer_index + trans_batch.batch_size <= self.buffer_size:
            self.update(trans_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + trans_batch.batch_size))
            self.buffer_index = (self.buffer_index + trans_batch.batch_size)
            self.transation_in_buffer = max(self.transation_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(trans_batch[0:buffer_left])
            self.insert_episode_batch(trans_batch[buffer_left:])

    def can_sample(self, batch_size):
        return self.transation_in_buffer >= batch_size

    def sample(self, batch_size):
        '''
                :param batch_size:int
                :return: EpisodeBatch
                '''
        assert self.can_sample(batch_size)
        if self.transation_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.transation_in_buffer, batch_size, replace=False)
            return self[ep_ids]
    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} ".format(self.transation_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys()
                                                                       )
