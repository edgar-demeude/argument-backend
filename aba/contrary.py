from .literal import Literal


class Contrary:
    """
    Represents a contrary relationship between two literals.

    Attributes:
        contraried_literal (Literal): The literal that is being contraried.
        contrary_attacker (Literal): The literal that attacks the contraried literal.
    """

    def __init__(self, contraried_literal: Literal, contrary_attacker: Literal):
        self.contraried_literal: Literal = contraried_literal
        self.contrary_attacker: Literal = contrary_attacker

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Contrary):
            return False
        return (
            self.contraried_literal == other.contraried_literal
            and self.contrary_attacker == other.contrary_attacker
        )

    def __str__(self) -> str:
        return f" {str(self.contraried_literal)}\u0304 = {str(self.contrary_attacker)}"

    
    def __hash__(self) -> int:
        return hash((self.contraried_literal, self.contrary_attacker))
