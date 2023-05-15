#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/29 22:03
# @Author  : Wubing Chen
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm

REGISTRY = {}

from .rnn_selector import RNNSelector


REGISTRY["rnn"] = RNNSelector
