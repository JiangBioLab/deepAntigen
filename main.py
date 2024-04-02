import warnings
warnings.filterwarnings('ignore')
from predictAntigen.antigenHLAI import run_antigenHLAI_seq
from predictAntigen.antigenHLAII import run_antigenHLAII_seq
from predictAntigen.antigenTCR import run_antigenTCR_seq
# from AntigenGCN.antigen_HLAII.run_atom import Train_AntigenGCN_Atom_for_antigen_HLAII
# from AntigenGCN.antigen_TCR.run_atom import Train_AntigenGCN_Atom_for_antigen_TCR
# import antigen_HLAII
# import antigen_TCR

run_antigenTCR_seq.Train('./test/test_antigenTCR/Data/sequence/meta/k_fold_dataset')
# run_antigenTCR_seq.Inference('./test/test_antigenHLAI/Data/sequence/test.csv')
# Train_AntigenGCN_Atom_for_antigen_HLAII('./antigen_HLAII/Data/crystal_structure/info_noredudant.csv')
# Train_AntigenGCN_Atom_for_antigen_TCR('./antigen_TCR/Data/crystal_structure/info_noredudant.csv')
# run_atom.Inference('./antigen_HLAI/Data/crystal_structure/sample.csv')