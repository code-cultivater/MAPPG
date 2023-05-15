
# Learning Explicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning via Polarization Policy Gradient (AAAI 23)

Based on Weighted QMIX. Please refer to that repo for more documentation.


## Running experiments

```shell
python main.py --config=enhanced_ddpg   --env-config=sc2 with env_args.map_name=3s_vs_4z  buffer_size=5000 epsilon_anneal_time=50000 t_max=2050000 learner="self_enhanced_ddpg_double_subtrain"  central_mixer="ff"  buffer_cpu_only=True central_rnn_hidden_dim=128
```

## Citing

```
@inproceedings{DBLP:conf/aaai/05511,
  author       = {Wubing Chen, Wenbin Li, Xiao Liu, Shangdong Yang, Yang Gao},
  title        = {Learning Explicit Credit Assignment for Cooperative Multi-Agent Reinforcement
Learning via Polarization Policy Gradient},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023},
  publisher    = {{AAAI} Press},
 }
```

        
