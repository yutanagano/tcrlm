from src.schema import AminoAcid
from src.schema.tcr import TravGene, TrbvGene
from enum import Enum
from itertools import permutations, product
from typing import Union

Chain = Enum("Chain", ["Alpha", "Beta"])
Position = Enum("Position", ["C_TERM", "M1", "M2", "M3", "N_TERM"])
Residue = Enum("Residues", [aa.name for aa in AminoAcid] + ["null"])


class JunctionEdit:
    def __init__(
        self, chain: Chain, position: Position, from_residue: Residue, to_residue: Residue
    ) -> None:
        self.chain = chain
        self.position = position
        self.from_residue = from_residue
        self.to_residue = to_residue

    def is_on_chain(self, chain: Chain) -> bool:
        return self.chain == chain

    def is_at_position(self, position: Position) -> bool:
        return self.position == position

    @property
    def is_central(self) -> bool:
        return self.position in (Position.M1, Position.M2, Position.M3)

    def is_from(self, from_residue: Residue) -> bool:
        return self.from_residue == from_residue

    def is_to(self, to_residue: Residue) -> bool:
        return self.to_residue == to_residue

    def __hash__(self) -> int:
        return hash((self.chain, self.position, self.from_residue, self.to_residue))

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, JunctionEdit):
            return False

        return (self.chain, self.position, self.from_residue, self.to_residue) == (
            __value.chain,
            __value.position,
            __value.from_residue,
            __value.to_residue,
        )

    def __repr__(self) -> str:
        return f"{self.chain.name}.{self.position.name}.{self.from_residue.name}.{self.to_residue.name}"

    @staticmethod
    def from_str(s: str) -> "JunctionEdit":
        chain_str, position_str, from_residue_str, to_residue_str = s.split(".")

        chain = Chain[chain_str]
        position = Position[position_str]
        from_residue = Residue[from_residue_str]
        to_residue = Residue[to_residue_str]

        return JunctionEdit(chain, position, from_residue, to_residue)


def get_all_tcr_edits():
    return [
        JunctionEdit(chain, position, from_residue, to_residue)
        for chain, position, (from_residue, to_residue) in product(
            Chain, Position, permutations(Residue, r=2)
        )
    ]
