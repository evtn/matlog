from .tokens import Expression, Atom, Literal, Operator


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, expr_string: str):
        self.string = expr_string
        if not expr_string:
            raise ParseError("Empty expression string")
        self.state: bool = False # False if expects atom, else expects operator

    def parse(self, return_index: bool = False) -> Expression:
        result = []
        i = 0
        while i < len(self.string):
            if self[i].isspace():
                i += 1
                continue
            if self.state:
                if self[i] == ")":
                    i += 1
                    break
                i, token = self.parse_op(i)
                result.append(token)
                self.state = False
            else:
                if any(self[i:].startswith(x) for x in Operator.unary_operators):
                    if result[-1].identifier in Operator.unary_operators:
                        raise ParseError("Can't parse two or more unary operators in a row, use brackets to split them: ~(~A)")
                    i, token = self.parse_op(i)
                else:
                    if self[i] == "(":
                        i += 1
                        j, token = Parser(self[i:]).parse(return_index=True)
                        result.append(token)
                        i += j
                    elif self[i] in "01":
                        token = Literal(int(self[i]))
                        result.append(token)
                        i += 1
                    else:
                        token = Atom(self[i])
                        if not self[i].isalpha():
                            raise ParseError(f"Expected letter, found {self[i]}")
                        i += 1
                        result.append(token)
                    self.state = True

        if not self.state:
            raise ParseError("Invalid expression string. Expected Non-Operator token, found EOL")

        expr = Expression(result).unwrap()

        if return_index:
            return i, expr
        return expr

    def __getitem__(self, i):
        return self.string[i]

    def parse_op(self, i):
        ops = 1
        j = 0
        while ops:
            ops = [*filter(lambda x: x[j:].startswith(self[i]), Operator.operators)]
            if len(ops) == 1:
                return i + len(ops[0]), Operator(ops[0])
            j += 1
        raise ParseError(f"Unknown operator: {self[i:j]}")



Expression.tokens_from_string = lambda string: Parser(string).parse()

