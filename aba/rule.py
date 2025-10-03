from .literal import Literal


class Rule:
    """
    Represents a rule in propositional logic.

    Attributes:
        rule_name (str): The name of the rule.
        head (Literal): The head literal of the rule.
        body (set[Literal]): The body literals of the rule.
    """

    def __init__(self, rule_name: str, head: Literal = None, body: set[Literal] = None):
        # TODO : According to the lecture notes : "We say that an ABA framework is flat iff no assumption is the head of a rule."
        # Should we check if the head is an instance of Assumption here ?

        self.rule_name: str = rule_name
        self.head: Literal = head
        self.body: set[Literal] = body if body is not None else set()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return False
        return (
            self.head == other.head
            and self.body == other.body
        )

    def __str__(self) -> str:
        body_str = ','.join(str(literal)
                            for literal in self.body) if self.body else ''
        return f"{self.head} ← {body_str}"

    def __hash__(self):
        return hash((self.rule_name, self.head, frozenset(self.body)))
