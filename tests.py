import unittest
from matlog import parse
from matlog.tokens import Literal, Atom, Operator


class TestParse(unittest.TestCase):
    def test_literal(self):
        self.assertTrue(
            parse("0").matches([Literal(0)]),
        )
        self.assertTrue(
            parse("1").matches([Literal(1)]),
        )

    def test_atom(self):
        self.assertTrue(
            parse("A").matches([Atom("A")]),
        )

    def test_unary(self):
        self.assertTrue(
            parse("~A").matches([Operator("~"), Atom("A")]),
        )
        self.assertTrue(
            parse("~~~A").matches(
                [Operator("~"), [Operator("~"), [Operator("~"), Atom("A")]]]
            ),
        )

    def test_binary(self):
        self.assertTrue(
            parse("A & B").matches([Atom("A"), Operator("&"), Atom("B")]),
        )
        self.assertTrue(
            parse("A -> B").matches([Atom("A"), Operator("->"), Atom("B")]),
        )


class TestSolve(unittest.TestCase):
    def test_literal(self):
        self.assertTrue(
            parse("1 | 0").solve().matches(Literal(1)),
        )
        self.assertTrue(
            parse("1 & 0").solve().matches(Literal(0)),
        )
        self.assertTrue(
            parse("1 -> 0").solve().matches(Literal(0)),
        )
        self.assertTrue(
            parse("1 <- 0").solve().matches(Literal(1)),
        )
        self.assertTrue(
            parse("1 ^ 0").solve().matches(Literal(1)),
        )
        self.assertTrue(
            parse("1 == 0").solve().matches(Literal(0)),
        )
        self.assertTrue(
            parse("~1").solve().matches(Literal(0)),
        )

    def test_atom(self):
        self.assertTrue(
            parse("A").solve(A=0).matches(Literal(0)),
        )
        self.assertTrue(
            parse("A").solve(A=1).matches(Literal(1)),
        )

    def test_partial(self):
        self.assertTrue(
            parse("A | B").solve(B=1).matches(Literal(1)),
        )
        self.assertTrue(
            parse("A & B").solve(B=0).matches(Literal(0)),
        )
        self.assertTrue(
            parse("A ^ B").solve(B=0).matches(Atom("A")),
        )
        self.assertTrue(
            parse("A == B").solve(B=1).matches(Atom("A")),
        )
        self.assertTrue(
            parse("A | ~A").solve().matches(Literal(1)),
        )
        self.assertTrue(
            parse("A & ~A").solve().matches(Literal(0)),
        )

    def test_complex(self):
        expr = parse("A & (B -> C)")

        self.assertTrue(
            expr.solve(A=0).matches(Literal(0)),
        )

        self.assertTrue(
            expr.solve(A=1, B=0).matches(Literal(1)),
        )

        self.assertTrue(
            expr.solve(A=1, B=1, C=0).matches(Literal(0)),
        )


class TestSimplify(unittest.TestCase):
    expressions = [
        parse("A | (A | B)"),  # should be A | B
        parse("(~A | ~B) & (A | B)"),  # should be A ^ B
        parse("(A & ~B) | (A & B)"),  # should be A
        parse("A & (A | B | C | D | E)"),  # should be A
        parse("(A & B) & (A & C) & (A & ~C) & (A & ~B)"),  # should be 0
        parse("~A | ~B"),  # should be ~(A & B)
    ]

    def test_complexity(self):
        for expr in self.expressions:
            simplified = expr.simplify()
            oversimplified = simplified.simplify()

            self.assertLessEqual(
                simplified.complexity,
                expr.complexity,
                msg=f"simplify() with '{expr}' (c: {expr.complexity}) produced '{simplified}' (c: {simplified.complexity})",
            )

            self.assertTrue(
                expr.equals(simplified),
                msg=f"simplify() with '{expr}' produced '{simplified}'",
            )

            self.assertEqual(
                simplified.complexity,
                oversimplified.complexity,
                msg=f"Unfinished simplification with {expr} (c: {expr.complexity}): Produced: {simplified} (c: {simplified.complexity}). Produced on the second step: {oversimplified} (c: {oversimplified.complexity})",
            )


if __name__ == "__main__":
    unittest.main()
