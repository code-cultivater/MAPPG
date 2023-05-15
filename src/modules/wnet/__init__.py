#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/6 15:42
# @Author  : Wubing Chen
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm

REGISTRY={}

from .lica_v_net import LICAVNet
from .v_net import VNet
from .multihead_v_net import  MultiheadVNet
REGISTRY["licavnet"]=LICAVNet
REGISTRY["vnet"]=VNet
REGISTRY["multihead_v_net"]=MultiheadVNet
