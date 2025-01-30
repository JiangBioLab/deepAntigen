import torch
from argparse import Namespace

state_dict = torch.load('seq-level_parameters.pt')
print(state_dict['opt'])
# state_dict['opt'] = vars(state_dict['opt'])
state_dict['opt']['num_process']=8
# state_dict['opt'].pop('num_workers')
state_dict['opt']['save_dir']='./antigenHLAII_training_log/'
state_dict['opt']['output']='./antigenHLAII_Output/seq-level/'
print(state_dict['opt'])
torch.save(state_dict, 'seq-level_parameters.pt')
