import argparse
import torch
import numpy as np
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

# python create_pt.py 
# --pdb /dapustor/nilufer/jonathan/ProteinMPNN/flip2_data/input_pdb/5OCM.pdb 
# --chain B 
# --out /dapustor/nilufer/jonathan/ProteinMPNN/flip2_data/structure/5OCM_B.pt

ATOM14_NAMES = [
    "N", "CA", "C", "O",
    "CB", "CG", "CG1", "CG2",
    "CD", "CD1", "CD2",
    "CE", "CE1", "CE2"
]


def three_to_one(resname):
    resname = resname.capitalize()
    return protein_letters_3to1.get(resname, "X")


def pdb_to_pt(pdb_file, chain_id, output_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    model = structure[0]
    chain = model[chain_id]

    seq = []
    xyz_list = []
    mask_list = []

    for residue in chain:
        # Skip water/hetero atoms
        if residue.id[0] != " ":
            continue

        resname = residue.get_resname()
        aa = three_to_one(resname)
        seq.append(aa)

        coords = np.full((14, 3), np.nan, dtype=np.float32)
        atom_mask = np.zeros(14, dtype=np.float32)

        for i, atom_name in enumerate(ATOM14_NAMES):
            if atom_name in residue:
                coords[i] = residue[atom_name].get_coord()
                atom_mask[i] = 1.0

        # Residue is valid if backbone atoms exist
        has_backbone = all(atom in residue for atom in ["N", "CA", "C", "O"])

        xyz_list.append(coords)
        mask_list.append(1 if has_backbone else 0)

    seq = "".join(seq)
    xyz = torch.tensor(np.stack(xyz_list), dtype=torch.float32)
    mask = torch.tensor(mask_list, dtype=torch.uint8)

    seq_id = pdb_file.split("/")[-1].replace(".pdb", "") + f"_{chain_id}"

    data = {
        "seq_id": seq_id,
        "seq": seq,
        "xyz": xyz,
        "mask": mask,
    }

    torch.save(data, output_file)

    print("Saved:", output_file)
    print("seq_id:", seq_id)
    print("Sequence length:", len(seq))
    print("xyz shape:", xyz.shape)
    print("mask shape:", mask.shape)
    print("Number of valid residues:", mask.sum().item())
    print("All valid?", torch.all(mask == 1).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--chain", required=True, help="Chain ID, e.g. A or B")
    parser.add_argument("--out", required=True, help="Output ex 50CM_B.pt file")

    args = parser.parse_args()

    pdb_to_pt(args.pdb, args.chain, args.out)


file_path = "/dapustor/nilufer/jonathan/ProteinMPNN/flip2_data/structure/5OCM_B.pt"

data = torch.load(file_path, map_location="cpu")

print("=" * 50)
print("PT FILE CONTENT")
print("=" * 50)

print("\nType:", type(data))

# Print keys
if isinstance(data, dict):

    print("\nKeys:")
    for key in data.keys():
        print(" -", key)

    print("\nContents summary:")

    for key, value in data.items():
        print("\n" + "-" * 40)
        print("Key:", key)
        print("Type:", type(value))

        if isinstance(value, torch.Tensor):
            print("Shape:", value.shape)
            print("Dtype:", value.dtype)

            # For mask, print extra info
            if key == "mask":
                print("Mask values:")
                print(value)

                print("Number of ones:", value.sum().item())
                print("All ones?:", torch.all(value == 1).item())

            # For xyz show small sample
            if key == "xyz":
                print("First residue coordinates:")
                print(value[0])

        else:
            print("Value:", value)

            if key == "seq":
                print("Sequence length:", len(value))

else:
    print(data)