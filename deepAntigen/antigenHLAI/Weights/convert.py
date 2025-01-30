import torch
from argparse import Namespace

state_dict = torch.load('atom-level_parameters.pt')
print(state_dict['opt'])
# state_dict['opt'] = vars(state_dict['opt'])
# state_dict['opt']['num_process']=8
state_dict['opt'].pop('num_workers')
state_dict['opt']['save_dir']='./antigenHLAI_finetune_log/'
state_dict['opt']['output']='./antigenHLAI_Output/atom-level/'
print(state_dict['opt'])
torch.save(state_dict, 'atom-level_parameters.pt')
