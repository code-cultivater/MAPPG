path="../../../results/offline_data/2022-12-18_16-21-05__self_enhanced_ddpg__sc2__5m_vs_6m/mac_list/"
import pickle
with open(path+"mac_list_5188_5397.pkl",'rb') as file:
    rq  = pickle.loads(file.read())

    print(rq)