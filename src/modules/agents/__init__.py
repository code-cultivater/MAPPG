REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .meta_rnn_agent import  MetaRNNAgent
from .multi_head_rnn_agent import  MultiHeadRNNAgent
from .rnn_agent_sigmoid import RNNAgent as RNNAgent_Sigmoid
from .rnn_agent_tanh import RNNAgent as RNNAgent_Tanh
from .rnn_agent_tanh2 import RNNAgent as RNNAgent_Tanh2
from .rnn_agent_ln import RNNAgent as RNNAgent_Ln
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_tanh"] = RNNAgent_Tanh
REGISTRY["rnn_tanh2"] = RNNAgent_Tanh2
REGISTRY["rnn_ln"] = RNNAgent_Ln
REGISTRY["rnn_sigmoid"] = RNNAgent_Sigmoid
REGISTRY["ff"] = FFAgent

REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["meta_rnn"] = MetaRNNAgent
REGISTRY["multihead_rnn_agent"]=MultiHeadRNNAgent


