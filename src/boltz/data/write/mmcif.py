import io
from collections.abc import Iterator
from typing import Optional

import ihm
import modelcif
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from torch import Tensor

from boltz.data import const
from boltz.data.types import Structure


def to_mmcif(structure: Structure, plddts: Optional[Tensor] = None) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output MMCIF file

    """
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Map entities to chain_ids
    entity_to_chains = {}
    entity_to_moltype = {}

    for chain in structure.chains:
        entity_id = chain["entity_id"]
        mol_type = chain["mol_type"]
        entity_to_chains.setdefault(entity_id, []).append(chain)
        entity_to_moltype[entity_id] = mol_type

    # Map entities to sequences
    sequences = {}
    for entity in entity_to_chains:
        # Get the first chain
        chain = entity_to_chains[entity][0]

        # Get the sequence
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = structure.residues[res_start:res_end]
        sequence = [str(res["name"]) for res in residues]
        sequences[entity] = sequence

    # Create entity objects
    lig_entity = None
    entities_map = {}
    for entity, sequence in sequences.items():
        mol_type = entity_to_moltype[entity]

        if mol_type == const.chain_type_ids["PROTEIN"]:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif mol_type == const.chain_type_ids["DNA"]:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif mol_type == const.chain_type_ids["RNA"]:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif len(sequence) > 1:
            alphabet = {}
            chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        # Handle smiles
        if len(sequence) == 1 and (sequence[0] == "LIG"):
            if lig_entity is None:
                seq = [chem_comp(sequence[0])]
                lig_entity = Entity(seq)
            model_e = lig_entity
        else:
            seq = [
                alphabet[item] if item in alphabet else chem_comp(item)
                for item in sequence
            ]
            model_e = Entity(seq)

        for chain in entity_to_chains[entity]:
            chain_idx = chain["asym_id"]
            entities_map[chain_idx] = model_e

    # We don't assume that symmetry is perfect, so we dump everything
    # into the asymmetric unit, and produce just a single assembly
    asym_unit_map = {}
    for chain in structure.chains:
        # Define the model assembly
        chain_idx = chain["asym_id"]
        chain_tag = str(chain["name"])
        asym = AsymUnit(
            entities_map[chain_idx],
            details="Model subunit %s" % chain_tag,
            id=chain_tag,
        )
        asym_unit_map[chain_idx] = asym
    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Add all atom sites.
            res_num = 0
            for chain in structure.chains:
                # We rename the chains in alphabetical order
                het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]

                residues = structure.residues[res_start:res_end]
                for residue in residues:
                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]
                    atoms = structure.atoms[atom_start:atom_end]
                    atom_coords = atoms["coords"]
                    for i, atom in enumerate(atoms):
                        # This should not happen on predictions, but just in case.
                        if not atom["is_present"]:
                            continue

                        name = atom["name"]
                        name = [chr(c + 32) for c in name if c != 0]
                        name = "".join(name)
                        element = periodic_table.GetElementSymbol(
                            atom["element"].item()
                        )
                        element = element.upper()
                        residue_index = residue["res_idx"] + 1
                        pos = atom_coords[i]
                        biso = (
                            100.00
                            if plddts is None
                            else round(plddts[res_num].item() * 100, 2)
                        )
                        yield Atom(
                            asym_unit=asym_unit_map[chain_idx],
                            type_symbol=element,
                            seq_id=residue_index,
                            atom_id=name,
                            x=f"{pos[0]:.5f}",
                            y=f"{pos[1]:.5f}",
                            z=f"{pos[2]:.5f}",
                            het=het,
                            biso=biso,
                            occupancy=1,
                        )

                    res_num += 1

        def add_plddt(self, plddts):
            res_num = 0
            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                residues = structure.residues[res_start:res_end]
                # We rename the chains in alphabetical order
                for residue in residues:
                    residue_idx = residue["res_idx"] + 1
                    self.qa_metrics.append(
                        _LocalPLDDT(
                            asym_unit_map[chain_idx].residue(residue_idx),
                            round(plddts[res_num].item() * 100, 2),
                        )
                    )
                    res_num += 1

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Model")
    if plddts is not None:
        model.add_plddt(plddts)

    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()
