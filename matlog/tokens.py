from typing import Dict, List, Union, NoReturn, Optional, Set
import operator
from .utils import cut_literal, combinations, Table

Value = Union[bool, int]


class Token:
    """Base class for tokens. Has no value and cannot be used in expression directly (would raise an error at some point)"""
    type = "unknown"
    identifier = "None"

    def same(self, other: "Token") -> bool:
        """tries to say if two instances of Token are equal"""
        return self.identifier.strip("()") == other.identifier.strip("()")

    def __str__(self) -> str:
        return self.identifier

    def __repr__(self) -> str:
        return f"Token:{self.type.upper()}[{self.identifier}]"

    def __and__(self, other) -> "Token":
        if self.same(other):
            return self
        if self.type == "literal":
            if self.value:
                return other
            return Token.FALSE
        if other.type == "literal":
            if other.value:
                return self
            return Token.FALSE
        return Expression([self, Operator("&"), other], explicit=False)

    def __or__(self, other) -> "Token":
        if self.same(other):
            return self
        if self.type == "literal":
            if self.value:
                return Token.TRUE
            return other
        if other.type == "literal":
            if other.value:
                return Token.TRUE
            return self
        return Expression([self, Operator("|"), other], explicit=False)

    def __pow__(self, other) -> "Token":
        if self.same(other):
            return Token.TRUE
        if self.type == "literal":
            if self.value:
                return Token.TRUE
            return -other
        if other.type == "literal":
            if other.value:
                return self
            return Token.TRUE
        return Expression([other, Operator("->"), self], explicit=False)

    def __xor__(self, other) -> "Token":
        if self.same(other):
            return Token.FALSE
        if self.type == "literal":
            if self.value:
                return -other
            return other
        if other.type == "literal":
            if other.value:
                return -self
            return self
        return Expression([self, Operator("^"), other], explicit=False)

    def __neg__(self) -> "Token":
        if self.type == "literal":
            return Literal(1 - self.value)
        return Expression([Operator("~"), self], explicit=False)

    def __eq__(self, other) -> "Token":
        if self.same(other):
            return Token.TRUE
        if self.type == "literal":
            if self.value:
                return other
            return -other
        if other.type == "literal":
            if other.value:
                return self
            return -self
        if self.type != "expr" and other.type == "expr":
            return Literal(other == self)
        return Expression([self, Operator("=="), other], explicit=False)

    def deep_copy(self) -> "Token":
        """

        Makes a deep copy of Token.
        By default it is the same as .copy()

        """
        return self.copy()

    def copy(self) -> "Token":
        """Makes a shallow copy of Token."""
        raise NotImplementedError

    def solve(self, context: Dict[str, Value]) -> "Token":
        """

        Returns a Token representing a resolved current token.

        :param Dict[str, Value] context: context dictionary for name resolving (i.e. {"A": 1, "B": 0})

        """
        raise NotImplementedError


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
        """Raises NotImplementedError as Operator can't be solved"""
        raise NotImplementedError("Operators can't have a value")

    def priority(self) -> int:
        """Returns current operator priority as integer, from 0 [highest] to 3 [lowest]"""
        return self.priorities.get(self.identifier, 3)

    def func(self, *args) -> Token:
        """Returns a token """
        return self.operators[self.identifier](*args)

    def copy(self):
        return Operator(self.identifier)


class Expression(Token):
    """Expression token - represents an expression"""

    type = "expr"

    def __init__(self, data: Union[str, List[Token]], explicit: bool = True):
        """
        initializes Expression

        :param Union[str, List[Token]] data: list of tokens or expression string (to be parsed)
        :param bool explicit: specifies whether it is necessary to show brackets in string representation, defaults to True

        """
        if isinstance(data, str):
            data = Expression.tokens_from_string(data).tokens # this one is from parser.py
        self.tokens = data
        self.explicit = explicit

    @staticmethod
    def tokens_from_string(string: str) -> List[Token]:
        """
        Constructs a list of tokens from an expression string. 
        As this requires a Parser, this function is overloaded by parser.py, providing an implementation.

        :param str string: string to parse

        """
        raise NotImplementedError("You have some mess with imports, as this function should've been overloaded by parser.py")

    @staticmethod
    def from_string(string: str) -> "Expression":
        """
        Same as `Expression(string)`
        Use `Expression(string)` form instead of this function

        :param str string: string to parse

        """
        return Expression(Expression.tokens_from_string(string))

    def solve(self, context: Optional[Dict[str, Value]] = None, treeset: Optional[set] = None, full_unwrap: bool = False, **kwargs: Value) -> Token:
        """
        
        Solves expression with provided context (recursively). If context is unspecified, uses keyword arguments.

        :param Optional[Dict[str, Value]] context: Expression context (maps atoms to values) Exapmle: {"A": 1, "B": 0}
        :param Optional[set] treeset: set of previous identifiers for internal usage. 
        :param bool full_unwrap: passed to .unwrap() if expression length equals 1
        :param Value **kwargs: used as context if context is unspecified

        :rtype: Token
        :returns: new (solved) expression, if full_unwrap is False, else result of solving, unwrapped

        """

        if len(self.tokens) == 1:
            return self.unwrap(full_unwrap)

        context = context or kwargs

        # preventing infinite recursion.
        identifier = repr(self)
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
            result[next_op:next_op + 2] = [
                result[next_op].func(
                    result[next_op + 1]
                )
            ]
        else:
            result[next_op - 1:next_op + 2] = [
                result[next_op].func(
                    result[next_op - 1], 
                    result[next_op + 1]
                )
            ]
        return Expression(result).solve(context, full_unwrap=full_unwrap, treeset=treeset, explicit=self.explicit)
    
    @property
    def identifier(self) -> str:
        """Expression identifier (expression string)"""
        
        contents = " ".join(map(str, self.tokens))
        if not self.explicit:
            return contents
        return f"({contents})"

    def unwrap(self, full_unwrap: bool = False) -> Token:
        """

        Unwraps nested expressions without losing any important tokens

        :param bool full_unwrap: If True, unwraps Expressions with one non-Expression token, like (1) or (A).
        :return: unwrap result or the expression itself if it cannot be unwrapped.
        :rtype: Token

        """
        if len(self.tokens) != 1:
            return self
        if self.tokens[0].type == "expr":
            return self.tokens[0].unwrap()
        if full_unwrap:
            return self.tokens[0]
        return self

    def __eq__(self, other: Token) -> Literal:
        if self.same(other):
            return Token.TRUE
        if other.type != "expr":
            other = Expression([other])

        letters = tuple(self.atoms() | other.atoms())

        for dataset in self.datasets(letters):
            if self.solve(dataset).identifier != other.solve(dataset).identifier:
                return Token.FALSE
        return Token.TRUE

    @staticmethod
    def datasets(letters):
        for dataset in combinations(letters):
            yield dataset

    def equals(self, other: Token) -> bool:
        """
        Returns True if two Expression objects are equal.
        Note this is not the same as `self == other`: This function returns a boolean value, while == returns a Literal
        
        """
        return (self == other)[0].value

    def atoms(self) -> Set[str]:
        """Returns a set of letters representing expression atoms"""
        letters = set()
        for token in self.tokens:
            if token.type == "expr":
                letters |= token.atoms
            elif token.type == "atom":
                letters.add(token.identifier)
        return letters

    def __neg__(self) -> List[Token]:
        if len(self.tokens) == 2 and self.tokens[0].identifier == "~":
            return self.tokens[1]
        return Expression([Operator("~"), self], explicit=False)

    def __repr__(self) -> str:
        identifier = "\n  ".join(
            map(
                lambda x: "\n  ".join(repr(x).split("\n")),
                self.tokens
            )
        )
        return f"Token.{['GROUP', 'EXPR'][self.explicit]}[\n  {identifier}\n]"

    def copy(self) -> "Expression":
        """Makes a shallow copy of Expression"""     
        return Expression(self.tokens)

    def deep_copy(self) -> "Expression":  
        """Makes a deep copy of Expression"""     
        return Expression(
            list(
                map(
                    lambda token: token.deep_copy(),
                    self.tokens
                )
            )
        )

    def value(self, context=None) -> int:
        """

        Returns value of expression (0 or 1) in the given context.
        If the specific value cannot be calculated, raises ValueError

        :raises: ValueError

        """
        val = self.solve(context).unwrap(full_unwrap=True)
        if val.type != "literal":
            raise ValueError(f"Expression {self} can't be solved, remaining: {val}")
        return val.value

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