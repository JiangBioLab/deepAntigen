from deepAntigen.antigenHLAI import run_antigenHLAI_seq
from deepAntigen.antigenHLAII import run_antigenHLAII_seq
from deepAntigen.antigenTCR import run_antigenTCR_seq

from deepAntigen.antigenHLAI import run_antigenHLAI_atom
from deepAntigen.antigenHLAII import run_antigenHLAII_atom
from deepAntigen.antigenTCR import run_antigenTCR_atom


df1 = run_antigenHLAI_seq.Inference('./test_antigenHLAI/Data/sequence/test.csv')
df2 = run_antigenHLAII_seq.Inference('./test_antigenHLAII/Data/sequence/test.csv')
df3 = run_antigenTCR_seq.Inference('./test_antigenTCR/Data/sequence/zero-shot_sample.csv')

peptide_atoms1, HLAI_atoms, contact_maps1 = run_antigenHLAI_atom.Inference('./test_antigenHLAI/Data/crystal_structure/sample.csv')
peptide_atoms2, HLAII_atoms, contact_maps2 = run_antigenHLAII_atom.Inference('./test_antigenHLAII/Data/crystal_structure/sample.csv')
peptide_atoms3, TCR_atoms, contact_maps3 = run_antigenTCR_atom.Inference('./test_antigenTCR/Data/crystal_structure/sample.csv')

print('Done!')