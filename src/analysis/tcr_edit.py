from ..resources import AMINO_ACIDS

from enum import Enum
from itertools import permutations, product

Position = Enum("Position", ["C_TERM", "M1", "M2", "M3", "N_TERM"])
Residue = Enum("Residues", AMINO_ACIDS + ("null",))


class TcrEdit:
    def __init__(self, position: Position, from_residue: Residue, to_residue: Residue) -> None:
        self.position = position
        self.from_residue = from_residue
        self.to_residue = to_residue

    def is_at(self, position: Position) -> bool:
        return self.position == position
    
    def is_central(self) -> bool:
        return self.position in (Position.M1, Position.M2, Position.M3)
    
    def is_from(self, from_residue: Residue) -> bool:
        return self.from_residue == from_residue
    
    def is_to(self, to_residue: Residue) -> bool:
        return self.to_residue == to_residue
    
    def __hash__(self) -> int:
        return hash((self.position, self.from_residue, self.to_residue))
    
    def __eq__(self, __value: object) -> bool:
        return (self.position, self.from_residue, self.to_residue) == (__value.position, __value.from_residue, __value.to_residue)
    
    def __repr__(self) -> str:
        return f"{self.position.name}.{self.from_residue.name}.{self.to_residue.name}"
    
    @staticmethod
    def from_str(s: str) -> "TcrEdit":
        position_str, from_residue_str, to_residue_str = s.split(".")

        position = Position[position_str]
        from_residue = Residue[from_residue_str]
        to_residue = Residue[to_residue_str]
        
        return TcrEdit(position, from_residue, to_residue)


def get_all_tcr_edits():
    return [
        TcrEdit(position, from_residue, to_residue) for position, (from_residue, to_residue) in product(
            Position,
            permutations(Residue, r=2)
        )
    ]