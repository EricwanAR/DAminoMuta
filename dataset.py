import pandas as pd
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset
import re
import json
from typing import Literal
import os
# import io
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
# from PIL import Image
import torchvision.io as tvio
# import torchvision.transforms as tvt
import torchvision.transforms.v2.functional as tvtF


AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
valid_aa = set(AMINO_ACIDS)

def is_valid_sequence(seq):
    for ch in seq:
        if not ch.isalpha():
            return False
        if ch.upper() not in valid_aa:
            return False
    return True

def parse_mic(mic_str):
    if not isinstance(mic_str, str):
        return float(mic_str)
    
    mic_str = mic_str.strip()
    mic_str = re.sub(r'\s+', '', mic_str)
    
    # 匹配纯数字
    if re.fullmatch(r'\d+(\.\d+)?', mic_str):
        return float(mic_str)
    
    # 匹配 >{数字} 或 ≥{数字}
    m = re.fullmatch(r'[>≥](\d+(\.\d+)?)', mic_str)
    if m:
        num = float(m.group(1))
        return num * 1.5
    
    # 匹配 <{数字} 或 ≤{数字}
    m = re.fullmatch(r'[<≤](\d+(\.\d+)?)', mic_str)
    if m:
        num = float(m.group(1))
        return num * 0.75
    
    # 匹配 {数字}±{数字}
    m = re.fullmatch(r'(\d+(\.\d+)?)[±](\d+(\.\d+)?)', mic_str)
    if m:
        return float(m.group(1))
    
    # 匹配 {数字}-{数字}
    m = re.fullmatch(r'(\d+(\.\d+)?)-(\d+(\.\d+)?)', mic_str)
    if m:
        num1 = float(m.group(1))
        num2 = float(m.group(3))
        return (num1 + num2) / 2.0
    
    print(f"Warning: Unable to parse MIC value {mic_str}")
    return np.nan

def encode_sequence(seq, pad_length):
    n = len(seq)
    arr = np.zeros((pad_length, 21), dtype=np.float32)
    
    # 对实际序列部分进行编码
    for i, char in enumerate(seq):
        if i >= pad_length:
            break  # 超出部分不处理（数据集构造时已过滤掉长序列）
        if char.islower():
            d_indicator = 1.0
            aa = char.upper()
        else:
            d_indicator = 0.0
            aa = char
        arr[i, 0] = d_indicator
        if aa in AA_to_index:
            idx = AA_to_index[aa]
            arr[i, idx + 1] = 1.0
        else:
            print(f"Warning: Amino acid {aa} not a standard AA")
    return torch.tensor(arr)

def geometric_mean(values):
    log_vals = np.log(np.array(values))
    return float(np.exp(log_vals.mean()))

def process_label(ratio, task):
    if ratio <= 0:
        return np.nan
    ratio_log = np.log2(ratio)
    if task == "reg":
        return np.float32(ratio_log)
    elif task == "cls":
        if ratio_log < 0.:
            return 1
        else:
            return 0
    else:
        raise ValueError("Unknown task, please use 'reg' or 'cls'")

def load_data(xlsx_file):
    df = pd.read_excel(xlsx_file, engine="calamine")
    
    groups = {}
    for _, row in df.iterrows():
        orig = row["SEQUENCE - Original"]
        variant = row["SEQUENCE - D-type amino acid substitution"]
        mic_raw = row["TARGET ACTIVITY - CONCENTRATION"]

        if not (isinstance(orig, str) and is_valid_sequence(orig)):
            continue
        if not (isinstance(variant, str) and is_valid_sequence(variant)):
            continue
        
        mic_val = parse_mic(mic_raw)
        
        if orig not in groups:
            groups[orig] = {}
        if variant not in groups[orig]:
            groups[orig][variant] = []
        groups[orig][variant].append(mic_val)
    
    groups_avg = {}
    for orig, var_dict in groups.items():
        groups_avg[orig] = {}
        for variant, mic_list in var_dict.items():
            mic_list = [x for x in mic_list if not np.isnan(x)]
            if len(mic_list) == 0:
                continue
            groups_avg[orig][variant] = geometric_mean(mic_list)
    return groups_avg


class PeptidePairDataset(Dataset):
    def __init__(self, mode=Literal['train', 'test', 'r2_case'], pad_length=30, task="cls", 
                 include_reverse=False, include_self=False, one_way=False, gf=False) :
        if mode == "train":
            xlsx_file = os.path.join(os.path.dirname(__file__), 'dataset', 'train.xlsx')
        elif mode in ["test", "r2_case"]:
            one_way = True
            xlsx_file = os.path.join(os.path.dirname(__file__), 'dataset', f'{mode}.xlsx')
        else:
            raise ValueError("Unknown mode")

        self.data = []
        self.pad_length = pad_length
        self.task = task
        groups_avg = load_data(xlsx_file)
        if gf:
            gf_dict = torch.load(os.path.join(os.path.dirname(__file__), 'dataset', 'protbert.pth'))
        
        for orig, variant_dict in groups_avg.items():
            filtered_variants = {variant: mic for variant, mic in variant_dict.items() 
                                 if len(variant) <= pad_length}
            variants = list(filtered_variants.keys())
            n_variants = len(variants)
            if n_variants == 0:
                continue
            
            if gf:
                glob_feat = gf_dict[orig.upper()]

            if include_self and (not one_way):
                for variant in variants:
                    encoded_seq = encode_sequence(variant, pad_length)
                    label = process_label(1.0, task)  # log2(1)=0
                    if gf:
                        self.data.append(((encoded_seq, encoded_seq, glob_feat), label))
                    else:
                        self.data.append(((encoded_seq, encoded_seq), label))
            
            for i in [0] if one_way else range(n_variants):
                for j in range(i + 1, n_variants):
                    var1 = variants[i]
                    var2 = variants[j]
                    mic1 = filtered_variants[var1]
                    mic2 = filtered_variants[var2]
                    
                    ratio = mic2 / mic1 if mic1 != 0 else np.nan
                    label = process_label(ratio, task)
                    if np.isnan(label):
                        continue
                    encoded_var1 = encode_sequence(var1, pad_length)
                    encoded_var2 = encode_sequence(var2, pad_length)
                    if gf:
                        self.data.append(((encoded_var1, encoded_var2, glob_feat), label))
                    else:
                        self.data.append(((encoded_var1, encoded_var2), label))
                    
                    if include_reverse and (not one_way):
                        rev_ratio = mic1 / mic2 if mic2 != 0 else np.nan
                        rev_label = process_label(rev_ratio, task)
                        if gf:
                            self.data.append(((encoded_var2, encoded_var1, glob_feat), rev_label))
                        else:
                            self.data.append(((encoded_var2, encoded_var1), rev_label))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class PeptidePairPicDataset(Dataset):
    def __init__(self, mode=Literal['train', 'test', 'r2_case'], pad_length=30, task="reg", 
                 include_reverse=False, include_self=False, one_way=False, gf=False,
                 side_enc=None, pcs=False, resize=None) :
        if mode == "train":
            xlsx_file = os.path.join(os.path.dirname(__file__), 'dataset', 'train.xlsx')
        elif mode in ["test", "r2_case"]:
            one_way = True
            xlsx_file = os.path.join(os.path.dirname(__file__), 'dataset', f'{mode}.xlsx')
        else:
            raise ValueError("Unknown mode")

        self.data = []
        self.pics = {}
        self.pad_length = pad_length
        self.task = task
        self.gf = gf
        self.side_enc = True if side_enc else False
        self.pcs = pcs
        self.resize = resize
        groups_avg = load_data(xlsx_file)
        if gf:
            gf_dict = torch.load(os.path.join(os.path.dirname(__file__), 'dataset', 'protbert.pth'))
        
        for orig, variant_dict in groups_avg.items():
            filtered_variants = {variant: mic for variant, mic in variant_dict.items() 
                                 if len(variant) <= pad_length}
            variants = list(filtered_variants.keys())
            for variant in variants:
                if self.pcs == 'mix' and variant == orig:
                    self.pics[variant] = self.read_img(variant, False)
                else:
                    self.pics[variant] = self.read_img(variant, self.pcs)
            n_variants = len(variants)
            if n_variants == 0:
                continue
            
            if gf:
                glob_feat = gf_dict[orig.upper()]

            if include_self and (not one_way):
                for variant in variants:
                    label = process_label(1.0, task)  # log2(1)=0
                    if gf:
                        self.data.append((variant, variant, glob_feat, label))
                    else:
                        self.data.append((variant, variant, label))
            
            for i in [0] if one_way else range(n_variants):
                for j in range(i + 1, n_variants):
                    var1 = variants[i]
                    var2 = variants[j]
                    mic1 = filtered_variants[var1]
                    mic2 = filtered_variants[var2]
                    
                    ratio = mic2 / mic1 if mic1 != 0 else np.nan
                    label = process_label(ratio, task)
                    if np.isnan(label):
                        continue
                    if gf:
                        self.data.append((var1, var2, glob_feat, label))
                    else:
                        self.data.append((var1, var2, label))
                    
                    if include_reverse and (not one_way):
                        rev_ratio = mic1 / mic2 if mic2 != 0 else np.nan
                        rev_label = process_label(rev_ratio, task)
                        if gf:
                            self.data.append((var2, var1, glob_feat, rev_label))
                        else:
                            self.data.append((var2, var1, rev_label))
    
    def read_img(self, peptide, pcs):
        image = draw_peptide(peptide, self.resize, pcs)
        return image
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.gf:
            seq1, seq2, glob_feat, label = self.data[idx]
        else:
            seq1, seq2, label = self.data[idx]
        img1 = self.pics[seq1]
        img2 = self.pics[seq2]

        if self.side_enc:
            img1 = (img1, encode_sequence(seq1, self.pad_length))
            img2 = (img2, encode_sequence(seq2, self.pad_length))

        if self.gf:
            return (img1, img2, glob_feat), label
        else:
            return (img1, img2), label


class PeptidePairPicCaseDataset(Dataset):
    def __init__(self, case:str ='r2', pad_length=30, side_enc=None, pcs=False, resize=None, gf=False):

        if case == 'r2':
            self.template = 'KWKIKWPVKWFKML'
        elif case == 'Indolicidin':
            self.template = 'ILPWKWPWWPWRR'
        elif case == 'Temporin-A':
            self.template = 'FLPLIGRVLSGIL'
        elif case == 'Melittin':
            self.template = 'GIGAVLKVLTTGLPALISWIKRKRQQ'
        elif case == 'Anoplin':
            self.template = 'GLLKRIKTLL'
        else:
            self.template = case.upper().strip()
        self.data = []
        self.pad_length = pad_length
        self.side_enc = True if side_enc else False
        self.pcs = pcs
        self.resize = resize
        self.gf = gf

        if gf:
            self.glob_feat = torch.load(os.path.join(os.path.dirname(__file__), 'dataset', 'protbert.pth'))[self.template]
        
        pools = [(ch.upper(), ch.lower()) if ch != 'G' else (ch.upper(),) for ch in self.template]
        # 笛卡尔积，即所有组合
        self.variants = [''.join(chars) for chars in itertools.product(*pools)][1:]

        self.template_pic = self.read_img(self.template)
        if self.side_enc:
            self.template_seq = encode_sequence(self.template, self.pad_length)
    
    def read_img(self, peptide):
        image = draw_peptide(peptide, self.resize, self.pcs)
        return image
    
    def __len__(self):
        return len(self.variants)
    
    def __getitem__(self, idx):
        variant  = self.variants[idx]
        seq2, label = variant, variant
        img1 = self.template_pic
        img2 = self.read_img(variant)

        if self.side_enc:
            img1 = (img1, self.template_seq)
            img2 = (img2, encode_sequence(seq2, self.pad_length))

        if self.gf:
            return (img1, img2, self.glob_feat), label
        else:
            return (img1, img2), label


aa_side = {
    "A": "C", "R": "CCCNC(N)=N", "N": "CC(=O)N", "D": "CC(=O)O", "C": "CS",
    "E": "CCC(=O)O", "Q": "CCC(=O)N", "G": "", "H": "Cc1cnc[nH]1", "I": "C(C)CC",
    "L": "CC(C)C", "K": "CCCCN", "M": "CCSC", "F": "Cc1ccccc1", "P": "C1CCN1",
    "S": "CO", "T": "C(C)O", "W": "Cc1c[nH]c2ccccc12", "Y": "Cc1ccc(O)cc1", "V": "C(C)C"
}

def build_peptide_smiles(seq: str) -> str:
    tpl = {}
    for aa, R in aa_side.items():
        for stereo, chir in (("L", "@"), ("D", "@@")):
            if aa == "G":
                backbone = "N[C:{idx}]C"
            else:
                backbone = f"N[C{chir}H:{'{idx}'}]({R})C"
            tpl[f"{aa}_{stereo}"]       = backbone + "(=O)"
            tpl[f"{aa}_{stereo}_term"]  = backbone + "(=O)O"

    if not seq:
        return ""

    out = []
    n = len(seq)
    for i, aa in enumerate(seq, start=1):
        key = f"{aa.upper()}_{'L' if aa.isupper() else 'D'}"
        if i == n:
            key += "_term"
        out.append(tpl[key].format(idx=i))
    return "".join(out)

protease_patterns = {
    'trypsin':       re.compile(r'(?<=[KR])(?!P)'),
    'chymotrypsin':  re.compile(r'(?<=[FYWL])(?!P)'),
    'elastase':      re.compile(r'(?<=[AVSGT])(?!P)'),
    'enterokinase':  re.compile(r'D{4}K(?=[^P])'),
    'caspase':       re.compile(r'(?<=D)(?=[GSA])'),
}

def draw_peptide(sequence, size=[768], pcs=False):
    smiles = build_peptide_smiles(sequence)
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    highlight_bonds = []
    bond_colors = {}

    d_positions = {i for i, aa in enumerate(sequence, start=1) if aa.islower()}

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() in d_positions:
            for b in atom.GetBonds():
                idx = b.GetIdx()
                if idx not in highlight_bonds:
                    highlight_bonds.append(idx)
                bond_colors[idx] = (0.0, 0.0, 1.0)

    if pcs:
        cleavage_sites = set()
        for pat in protease_patterns.values():
            for m in pat.finditer(sequence):
                cut = m.end()  # 切在 cut 之后
                if 1 <= cut < len(sequence):
                    cleavage_sites.add(cut)

        for pos in cleavage_sites:
            ca = next((a for a in mol.GetAtoms()
                       if a.GetAtomMapNum() == pos), None)
            if ca is None:
                continue

            carbonyl_c = None
            for nb in ca.GetNeighbors():
                if nb.GetSymbol() != "C":
                    continue
                if any(bond.GetBondType() == Chem.BondType.DOUBLE and
                       o.GetSymbol() == "O"
                       for bond in nb.GetBonds()
                       for o in (bond.GetBeginAtom(), bond.GetEndAtom())):
                    carbonyl_c = nb
                    break
            if carbonyl_c is None:
                continue

            peptide_bond = None
            for b in carbonyl_c.GetBonds():
                o_atom = b.GetOtherAtom(carbonyl_c)
                if o_atom.GetSymbol() == "N":
                    peptide_bond = b
                    break
            if peptide_bond is None:
                continue

            bidx = peptide_bond.GetIdx()
            if bidx not in highlight_bonds:
                highlight_bonds.append(bidx)
            bond_colors[bidx] = (1.0, 0.0, 0.0)
    
    if len(size) == 1:
        w = h = size[0]
    else:
        w, h = size

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=[],
        highlightBonds=highlight_bonds,
        highlightAtomColors={},
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()

    png_bytes = bytearray(drawer.GetDrawingText())
    byte_tensor = torch.frombuffer(png_bytes, dtype=torch.uint8)
    img = tvio.decode_png(byte_tensor, mode=tvio.ImageReadMode.RGB)       # [3, H, W], uint8
    img = tvtF.to_dtype(img, torch.float32)
    img = tvtF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img
