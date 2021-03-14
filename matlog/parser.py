from .tokens import Expression, Atom, Literal, Operator
from lark import Lark, Transformer


class MatlogT(Transformer):
    def unary_expr(self, tokens):
        return Expression(
            [Operator(tokens[0].children[0].value), tokens[1]], explicit=False
        )

    def parens_expr(self, tokens):
        tokens[0].explicit = True
        return tokens[0]

    def bin_expr(self, tokens):
        return Expression(
            [tokens[0], Operator(tokens[1].children[0].value), tokens[2]],
            explicit=False,
        )

    def literal(self, tokens):
        return Literal(bool(int(tokens[0])))

    def letter(self, tokens):
        return Atom(str(tokens[0]))


def parse(s, process_level=0):
    expr = Expression([transformer.transform(parser.parse(s))]).unwrap()

    if process_level:
        expr = Expression([expr.solve()]).unwrap()
    if process_level > 1:
        expr = expr.simplify()
    return expr


grammar = """
%import common.WS
%ignore WS
letter: "A".."Z" | "a".."z"
literal: "0".."1"
!and_: "&"
!or_: "|"
!xor: "^"
!equality: "=="
!impl: "->"
!rev_impl: "<-"
!invert: "~"
?low_op: rev_impl | impl | equality | xor
?bin_expr: or_expr (low_op bin_expr)*
?or_expr: and_expr or_ or_expr -> bin_expr | and_expr
?and_expr: unary_expr and_ and_expr -> bin_expr | unary_expr
?unary_expr: invert unary_expr | atom
parens_expr: "(" bin_expr ")"
?atom: letter 
     | literal
     | parens_expr
?start: bin_expr
"""

parser = Lark(grammar, parser="lalr")
transformer = MatlogT()
