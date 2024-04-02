import numpy as np
import os
import logging
from typing import List, Union, Tuple

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer

logger = logging.getLogger(__name__)

DEFAULT_ATOM_TYPE_SET = [
  "C",
  "O",
  "N",
  "S",
]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_RING_SIZE_SET = [3, 4, 5, 6, 7, 8]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
DEFAULT_GRAPH_DISTANCE_SET = [1, 2, 3, 4, 5, 6, 7]


class _ChemicalFeaturesFactory:
    """This is a singleton class for RDKit base features."""
    _instance = None

    @classmethod
    def get_instance(cls):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if not cls._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
        return cls._instance


def one_hot_encode(val: Union[int, str],
                   allowable_set: Union[List[str], List[int]],
                   include_unknown_set: bool = False) -> List[float]:
    """One hot encoder for elements of a provided set.

    Examples
    --------
    >>> one_hot_encode("a", ["a", "b", "c"])
    [1.0, 0.0, 0.0]
    >>> one_hot_encode(2, [0, 1, 2])
    [0.0, 0.0, 1.0]
    >>> one_hot_encode(3, [0, 1, 2])
    [0.0, 0.0, 0.0]
    >>> one_hot_encode(3, [0, 1, 2], True)
    [0.0, 0.0, 0.0, 1.0]

    Parameters
    ----------
    val: int or str
        The value must be present in `allowable_set`.
    allowable_set: List[int] or List[str]
        List of allowable quantities.
    include_unknown_set: bool, default False
        If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        An one-hot vector of val.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    Raises
    ------
    ValueError
        If include_unknown_set is False and `val` is not in `allowable_set`.
    """
    if include_unknown_set is False:
        if val not in allowable_set:
            logger.info("input {0} not in allowable set {1}:".format(
                    val, allowable_set))

    # init an one-hot vector
    if include_unknown_set is False:
        one_hot_legnth = len(allowable_set)
    else:
        one_hot_legnth = len(allowable_set) + 1
    one_hot = [0.0 for _ in range(one_hot_legnth)]

    try:
        one_hot[allowable_set.index(val)] = 1.0    # type: ignore
    except:
        if include_unknown_set:
            # If include_unknown_set is True, set the last index is 1.
            one_hot[-1] = 1.0
        else:
            pass
    return one_hot


#################################################################
# atom (node) featurization
#################################################################


def get_atom_type_one_hot(atom: RDKitAtom,
                          allowable_set: List[str] = DEFAULT_ATOM_TYPE_SET,
                          include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of an atom type.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[str]
        The atom types to consider. The default set is
        `["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]`.
    include_unknown_set: bool, default True
        If true, the index of all atom not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        An one-hot vector of atom types.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(atom.GetSymbol(), allowable_set, include_unknown_set)


def construct_hydrogen_bonding_info(mol: RDKitMol) -> List[Tuple[int, str]]:
    """Construct hydrogen bonding infos about a molecule.

    Parameters
    ---------
    mol: rdkit.Chem.rdchem.Mol
        RDKit mol object

    Returns
    -------
    List[Tuple[int, str]]
        A list of tuple `(atom_index, hydrogen_bonding_type)`.
        The `hydrogen_bonding_type` value is "Acceptor" or "Donor".
    """
    factory = _ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding = []
    for f in feats:
        hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))
    return hydrogen_bonding


def get_atom_hydrogen_bonding_one_hot(
        atom: RDKitAtom, hydrogen_bonding: List[Tuple[int, str]]) -> List[float]:
    """Get an one-hot feat about whether an atom accepts electrons or donates electrons.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    hydrogen_bonding: List[Tuple[int, str]]
        The return value of `construct_hydrogen_bonding_info`.
        The value is a list of tuple `(atom_index, hydrogen_bonding)` like (1, "Acceptor").

    Returns
    -------
    List[float]
        A one-hot vector of the ring size type. The first element
        indicates "Donor", and the second element indicates "Acceptor".
    """
    one_hot = [0.0, 0.0]
    atom_idx = atom.GetIdx()
    for hydrogen_bonding_tuple in hydrogen_bonding:
        if hydrogen_bonding_tuple[0] == atom_idx:
            if hydrogen_bonding_tuple[1] == "Donor":
                one_hot[0] = 1.0
            elif hydrogen_bonding_tuple[1] == "Acceptor":
                one_hot[1] = 1.0
    return one_hot


def get_atom_is_in_aromatic_one_hot(atom: RDKitAtom) -> List[float]:
    """Get ans one-hot feature about whether an atom is in aromatic system or not.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A vector of whether an atom is in aromatic system or not.
    """
    return [float(atom.GetIsAromatic())]


def get_atom_hybridization_one_hot(
    atom: RDKitAtom,
    allowable_set: List[str] = DEFAULT_HYBRIDIZATION_SET,
    include_unknown_set: bool = False) -> List[float]:
    """Get an one-hot feature of hybridization type.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[str]
        The hybridization types to consider. The default set is `["SP", "SP2", "SP3"]`
    include_unknown_set: bool, default False
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        An one-hot vector of the hybridization type.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(
            str(atom.GetHybridization()), allowable_set, include_unknown_set)


def get_atom_total_num_Hs_one_hot(
    atom: RDKitAtom,
    allowable_set: List[int] = DEFAULT_TOTAL_NUM_Hs_SET,
    include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of the number of hydrogens which an atom has.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        The number of hydrogens to consider. The default set is `[0, 1, ..., 4]`
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the number of hydrogens which an atom has.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(atom.GetTotalNumHs(), allowable_set,
                          include_unknown_set)


def get_atom_chirality_one_hot(atom: RDKitAtom) -> List[float]:
    """Get an one-hot feature about an atom chirality type.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A one-hot vector of the chirality type. The first element
        indicates "R", and the second element indicates "S".
    """
    one_hot = [0.0, 0.0]
    try:
        chiral_type = atom.GetProp('_CIPCode')
        if chiral_type == "R":
            one_hot[0] = 1.0
        elif chiral_type == "S":
            one_hot[1] = 1.0
    except:
        pass
    return one_hot


def get_atom_formal_charge(atom: RDKitAtom) -> List[float]:
    """Get a formal charge of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A vector of the formal charge.
    """
    return [float(atom.GetFormalCharge())]


def get_atom_partial_charge(atom: RDKitAtom) -> List[float]:
    """Get a partial charge of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object

    Returns
    -------
    List[float]
        A vector of the parital charge.

    Notes
    -----
    Before using this function, you must calculate `GasteigerCharge`
    like `AllChem.ComputeGasteigerCharges(mol)`.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0.0
    return [float(gasteiger_charge)]


def get_atom_total_degree_one_hot(
    atom: RDKitAtom,
    allowable_set: List[int] = DEFAULT_TOTAL_DEGREE_SET,
    include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of the degree which an atom has.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        The degree to consider. The default set is `[0, 1, ..., 5]`
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the degree which an atom has.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(atom.GetTotalDegree(), allowable_set,
                          include_unknown_set)


#################################################################
# bond (edge) featurization
#################################################################


def get_bond_type_one_hot(bond: RDKitBond,
                          allowable_set: List[str] = DEFAULT_BOND_TYPE_SET,
                          include_unknown_set: bool = False) -> List[float]:
    """Get an one-hot feature of bond type.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object
    allowable_set: List[str]
        The bond types to consider. The default set is `["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]`.
    include_unknown_set: bool, default False
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the bond type.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(
      str(bond.GetBondType()), allowable_set, include_unknown_set)


def get_bond_is_in_same_ring_one_hot(bond: RDKitBond) -> List[float]:
    """Get an one-hot feature about whether atoms of a bond is in the same ring or not.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    List[float]
        A one-hot vector of whether a bond is in the same ring or not.
    """
    return [int(bond.IsInRing())]


def get_bond_is_conjugated_one_hot(bond: RDKitBond) -> List[float]:
    """Get an one-hot feature about whether a bond is conjugated or not.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    List[float]
        A one-hot vector of whether a bond is conjugated or not.
    """
    return [int(bond.GetIsConjugated())]


def get_bond_stereo_one_hot(bond: RDKitBond,
                            allowable_set: List[str] = DEFAULT_BOND_STEREO_SET,
                            include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of the stereo configuration of a bond.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object
    allowable_set: List[str]
        The stereo configuration types to consider.
        The default set is `["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]`.
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the stereo configuration of a bond.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(
      str(bond.GetStereo()), allowable_set, include_unknown_set)


def get_bond_graph_distance_one_hot(
    bond: RDKitBond,
    graph_dist_matrix: np.ndarray,
    allowable_set: List[int] = DEFAULT_GRAPH_DISTANCE_SET,
    include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of graph distance.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object
    graph_dist_matrix: np.ndarray
        The return value of `Chem.GetDistanceMatrix(mol)`. The shape is `(num_atoms, num_atoms)`.
    allowable_set: List[int]
        The graph distance types to consider. The default set is `[1, 2, ..., 7]`.
    include_unknown_set: bool, default False
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of the graph distance.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    graph_dist = graph_dist_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    return one_hot_encode(graph_dist, allowable_set, include_unknown_set)

def _construct_atom_feature(
    atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool,
    use_partial_charge: bool) -> np.ndarray:
    """Construct an atom feature from a RDKit atom object.

    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    h_bond_infos: List[Tuple[int, str]]
        A list of tuple `(atom_index, hydrogen_bonding_type)`.
        Basically, it is expected that this value is the return value of
        `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
        value is "Acceptor" or "Donor".
    use_chirality: bool
        Whether to use chirality information or not.
    use_partial_charge: bool
        Whether to use partial charge data or not.

    Returns
    -------
    np.ndarray
        A one-hot vector of the atom feature.
    """
    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
    atom_feat = np.concatenate([
      atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree,
      total_num_Hs
    ])

    if use_chirality:
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, chirality])

    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, partial_charge])
    return atom_feat


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    np.ndarray
        A one-hot vector of the bond feature.
    """
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


class MolGraphConvFeaturizer(MolecularFeaturizer):
    """This class is a featurizer of general graph convolution networks for molecules.

    The default node(atom) and edge(bond) representations are based on
    `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_. If you want to use your own representations,
    you could use this class as a guide to define your original Featurizer. In many cases, it's enough
    to modify return values of `construct_atom_feature` or `construct_bond_feature`.

    The default node representation are constructed by concatenating the following values,
    and the feature length is 30.

    - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
    - Formal charge: Integer electronic charge.
    - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
    - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
    - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
    - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
    - Partial charge: Calculated partial charge. (Optional)

    The default edge representation are constructed by concatenating the following values,
    and the feature length is 11.

    - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
    - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
    - Conjugated: A one-hot vector of whether this bond is conjugated or not.
    - Stereo: A one-hot vector of the stereo configuration of a bond.

    If you want to know more details about features, please check the paper [1]_ and
    utilities in deepchem.utils.molecule_feature_utils.py.

    Examples
    --------
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> out[0].num_node_features
    30
    >>> out[0].num_edge_features
    11

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
         Journal of computer-aided molecular design 30.8 (2016):595-608.

    Note
    ----
    This class requires RDKit to be installed.
    """

    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False):
        """
        Parameters
        ----------
        use_edges: bool, default False
            Whether to use edge features or not.
        use_chirality: bool, default False
            Whether to use chirality information or not.
            If True, featurization becomes slow.
        use_partial_charge: bool, default False
            Whether to use partial charge data or not.
            If True, this featurizer computes gasteiger charges.
            Therefore, there is a possibility to fail to featurize for some molecules
            and featurization becomes slow.
        """
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality

    def _featurize(self, mol: RDKitMol) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
            RDKit mol object.

        Returns
        -------
        graph: GraphData
            A molecule graph with some features.
        """

        if self.use_partial_charge:
            try:
                mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(mol)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(mol)
        atom_features = np.asarray(
                [
                  _construct_atom_feature(atom, h_bond_infos, self.use_chirality,
                                          self.use_partial_charge)
                  for atom in mol.GetAtoms()
                ],
                dtype=float,
        )
        # construct edge (bond) index
        src, dest = [], []
        for bond in mol.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None    # deafult None
        if self.use_edges:
            features = []
            for bond in mol.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)
        cdr3_graph = GraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features)
        return cdr3_graph