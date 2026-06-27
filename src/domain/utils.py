from __future__ import annotations

from enum import Enum


class AminoAcid(Enum):
    ALA = "ALA"
    ARG = "ARG"
    ASN = "ASN"
    ASP = "ASP"
    CYS = "CYS"
    GLN = "GLN"
    GLU = "GLU"
    GLY = "GLY"
    HIS = "HIS"
    ILE = "ILE"
    LEU = "LEU"
    LYS = "LYS"
    MET = "MET"
    PHE = "PHE"
    PRO = "PRO"
    SER = "SER"
    THR = "THR"
    TRP = "TRP"
    TYR = "TYR"
    VAL = "VAL"
    MSE = "MSE"
    HSD = "HSD"
    HSE = "HSE"
    HSP = "HSP"

    @classmethod
    def three_letter_from_one_letter(cls, code: str) -> "AminoAcid":
        mapping: dict[str, AminoAcid] = {
            "A": cls.ALA, "R": cls.ARG, "N": cls.ASN, "D": cls.ASP,
            "C": cls.CYS, "Q": cls.GLN, "E": cls.GLU, "G": cls.GLY,
            "H": cls.HIS, "I": cls.ILE, "L": cls.LEU, "K": cls.LYS,
            "M": cls.MET, "F": cls.PHE, "P": cls.PRO, "S": cls.SER,
            "T": cls.THR, "W": cls.TRP, "Y": cls.TYR, "V": cls.VAL,
        }
        if code not in mapping:
            raise ValueError(f"Unknown one-letter code: {code}")
        return mapping[code]

    @property
    def one_letter_from_three_letter(self) -> str:
        mapping: dict[AminoAcid, str] = {
            AminoAcid.ALA: "A", AminoAcid.ARG: "R", AminoAcid.ASN: "N",
            AminoAcid.ASP: "D", AminoAcid.CYS: "C", AminoAcid.GLN: "Q",
            AminoAcid.GLU: "E", AminoAcid.GLY: "G", AminoAcid.HIS: "H",
            AminoAcid.ILE: "I", AminoAcid.LEU: "L", AminoAcid.LYS: "K",
            AminoAcid.MET: "M", AminoAcid.PHE: "F", AminoAcid.PRO: "P",
            AminoAcid.SER: "S", AminoAcid.THR: "T", AminoAcid.TRP: "W",
            AminoAcid.TYR: "Y", AminoAcid.VAL: "V",
        }
        return mapping.get(self, "X")
