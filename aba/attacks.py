from .argument import Argument


class Attacks:
    """
    Represents an attack between two arguments in the argumentation framework.

    Attributes:
        attacker (Argument): The argument that is attacking.
        target (Argument): The argument that is being attacked.
    """

    def __init__(self, attacker: Argument, target: Argument):
        self.attacker = attacker
        self.target = target

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Attacks):
            return False
        return self.attacker == other.attacker and self.target == other.target

    def __str__(self) -> str:
        return f"[{self.attacker.argument_name}] → [{self.target.argument_name}]"

    def __hash__(self) -> int:
        return hash((self.attacker, self.target))

    def __repr__(self) -> str:
        return f"[{self.attacker.argument_name}] → [{self.target.argument_name}]"
