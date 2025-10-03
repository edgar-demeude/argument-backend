class Literal:
    """
    Represents a literal in propositional logic.

    Attributes:
        literal_name (str): The name of the literal.
        is_neg (bool): Indicates if the literal is negated.
    """

    def __init__(self, literal_name: str, is_neg: bool = False):
        self.literal_name: str = literal_name
        self.is_neg: bool = is_neg

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return False
        return self.literal_name == other.literal_name and self.is_neg == other.is_neg

    def __str__(self) -> str:
        return f"{'¬' if self.is_neg else ''}{self.literal_name}"

    def __hash__(self) -> int:
        return hash((self.literal_name, self.is_neg))

    def __repr__(self) -> str:
        return f"{'¬' if self.is_neg else ''}{self.literal_name}"
