from typing import Dict, List, Union, NoReturn, Optional
import operator
from .utils import cut_literal, combinations

Value = Union[bool, int]


class Token:
    """Base class for tokens. Has no value and cannot be used in expression directly (would raise an error at some point)"""
    type = "unknown"
    identifier = "None"

    def same(self, other):
        if self.identifier.strip("()") == other.identifier.strip("()"):
            return True
        return False

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return f"Token:{self.type.upper()}[{self.identifier}]"

    def __and__(self, other):
        if self.same(other):
            return [self]
        if self.type == "literal":
            if self.value:
                return [other]
            return [Token.FALSE]
        if other.type == "literal":
            if other.value:
                return [self]
            return [Token.FALSE]
        return [self, Operator("&"), other]

    def __or__(self, other):
        if self.same(other):
            return [self]
        if self.type == "literal":
            if self.value:
                return [Token.TRUE]
            return [other]
        if other.type == "literal":
            if other.value:
                return [Token.TRUE]
            return [self]
        return [self, Operator("|"), other]

    def __pow__(self, other):
        if self.same(other):
            return [Token.TRUE]
        if self.type == "literal":
            if self.value:
                return [Token.TRUE]
            return -other
        if other.type == "literal":
            if other.value:
                return [self]
            return [Token.TRUE]
        return [other, Operator("->"), self]

    def __xor__(self, other):
        if self.same(other):
            return [Token.FALSE]
        if self.type == "literal":
            if self.value:
                return -other
            return [other]
        if other.type == "literal":
            if other.value:
                return -self
            return [self]
        return [self, Operator("^"), other]

    def __neg__(self):
        if self.type == "literal":
            return [Literal(1 - self.value)]
        return [Operator("~"), self]

    def __eq__(self, other):
        if self.same(other):
            return [Token.TRUE]
        if self.type != "expr" and other.type == "expr":
            return [Literal(other == self)]
        return [self, Operator("<->"), other]


class Atom(Token):
    """Atom token - variable value represented by one letter."""
    type = "atom"

    def __init__(self, identifier: str):
        self.identifier = identifier

    def solve(self, context: Dict[str, Value]) -> Union["Atom", "Literal"]:
        if self.identifier in context:
            return Literal(context[self.identifier])
        return self

    def copy(self):
        return Atom(self.identifier)


class Literal(Token):
    """Literal token - literal value (0 or 1)."""
    type = "literal"

    def __init__(self, value: Value):
        self.value = value
        self.identifier = str(cut_literal(value))
    
    def solve(self, context: Dict[str, Value]) -> "Literal":
        return self

    def copy(self):
        return Literal(self.value)


class Operator(Token):
    """Operator token - defines an operation. Has no value."""
    type = "op"

    operators = {
        "~": operator.neg,
        "&": operator.and_,
        "|": operator.or_,
        "->": (lambda x, y: y ** x),
        "==": operator.eq,
        "^": operator.xor,
        "<-": operator.pow,
    }

    priorities = {
        "~": 0,
        "&": 1,
        "|": 2,
    }

    unary_operators = {"~"}

    def __init__(self, identifier: str):
        self.identifier = identifier

    def solve(self, context: Dict[str, Value]) -> NoReturn:
        raise NotImplementedError("Operators can't have a value")

    def priority(self) -> int:
        return self.priorities.get(self.identifier, 3)

    def func(self, *args):
        return self.operators[self.identifier](*args)

    def copy(self):
        return Operator(self.identifier)


class Expression(Token):
    """Expression token - represents an expression"""

    type = "expr"

    def __init__(self, data: Union[str, List[Token]]):
        if isinstance(data, str):
            data = Expression.tokens_from_string(data).tokens # this one is from parser.py
        self.tokens = data

    @staticmethod
    def tokens_from_string(string: str) -> List[Token]:
        raise NotImplementedError("You have some mess with imports, as this function should've been overloaded by parser.py")

    @staticmethod
    def from_string(string: str) -> "Expression":
        return Expression(Expression.tokens_from_string(string))

    def solve(self, context: Optional[Dict[str, Value]] = None, treeset: Optional[set] = None, full_unwrap: bool = False, **kwargs) -> Token:
        if len(self.tokens) == 1:
            return self.unwrap(full_unwrap)

        context = context or kwargs

        # preventing infinite recursion.
        identifier = self.identifier.replace("(", "").replace(")", "")
        if treeset and identifier in treeset:
            return self
        treeset = treeset or set()
        treeset.add(identifier)

        result = self.tokens[:]
        indices = range(len(result))

        atoms = set([x for x in indices if result[x].type != "op"])
        for index in atoms:
            if result[index].type == "expr":
                result[index] = result[index].solve(context, full_unwrap=True)
            else:
                result[index] = result[index].solve(context)
        
        next_op = min(
            [x for x in indices if x not in atoms], 
            key=lambda x: result[x].priority()
        )
        if result[next_op].identifier in Operator.unary_operators:
            result[next_op:next_op + 2] = result[next_op].func(result[next_op + 1])
        else:
            result[next_op - 1:next_op + 2] = result[next_op].func(result[next_op - 1], result[next_op + 1])
        return Expression(result).solve(context, full_unwrap=full_unwrap, treeset=treeset)

    @property
    def identifier(self) -> str:
        return ''.join(["(", " ".join(map(str, self.tokens)), ")"])

    def unwrap(self, full_unwrap=False) -> Token:
        if self.tokens[0].type == "expr":
            return self.tokens[0].unwrap()
        if full_unwrap:
            return self.tokens[0]
        return self

    def __eq__(self, other: Token) -> bool:
        if self.same(other):
            return [Token.TRUE]
        if other.type != "expr":
            other = Expression([other])

        letters = [*(self.atoms() | other.atoms())]
        for dataset in combinations(letters):
            if self.solve(dataset).identifier != other.solve(dataset).identifier:
                return [Token.FALSE]
        return [Token.TRUE]

    def equals(self, other: Token) -> bool:
        return (self == other)[0].value

    def atoms(self) -> List[str]:
        letters = set()
        for token in self.tokens:
            if token.type == "expr":
                letters |= token.atoms
            elif token.type == "atom":
                letters.add(token.identifier)
        return letters

    def __neg__(self):
        if len(self.tokens) == 2 and self.tokens[0].identifier == "~":
            return [self.tokens[1]]
        return [Operator("~"), self]

    def __repr__(self):
        return ''.join(["Token.EXPR[", " ".join(map(lambda x: repr(x), self.tokens)), "]"])

    def copy(self):
        return Expression(self.tokens)

    def deep_copy(self):
        tokens = []
        for token in self.tokens:
            if token.type == "expr":
                tokens.append(token.deep_copy())
            else:
                tokens.append(token.copy())
        return Expression(tokens)
    def table(self, keep=None) -> Table:
        """
        Returns a truth table for Expression

        :param Optional[bool] keep: optional filter parameter. If specified, only rows with this value would be used.
        :returns: truth table as utils.Table object (can be printed for a pretty table) 

        """
        result = []
        identifier = self.identifier
        for dataset in self.datasets(sorted(self.atoms())):
            dataset[identifier] = self.value(dataset)
            if keep is None or dataset[identifier] == keep:
                result.append(dataset)
        return Table({
            "identifiers": [*sorted(self.atoms()), identifier],
            "values": result
        })


Token.TRUE = Literal(1)
Token.FALSE = Literal(0)