def is_letter(c):
    return 65 <= c <= 90 or 97 <= c <= 122


def same(tokens):
    return tokens[0].identifier == tokens[1].identifier and tokens[0].type == "atom" and tokens[1].type == "atom"


class Token:
    operators = ["~", "&", "|", "->", "<->", "^", "<-"]
    def __init__(self, identifier):
        self.type = ["atom", "operator"][identifier in self.operators]
        self.identifier = identifier
        self.value = None

    @staticmethod
    def of(val):
        t = Token(str(bool(val)))
        t.value = val
        t.type = "literal"
        return t

    def copy(self):
        t = Token(self.identifier)
        t.type = self.type
        t.value = self.value
        return t

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.type == "literal":
            return f"Token.{self.type}({self.value})"
        if self.value is not None:
            return f"Token.{self.type}({self.identifier})={self.value}"
        return f"Token.{self.type}({self.identifier})"

    def __eq__(self, other):
        tokens = self, other
        values = self.value, other.value
        for i in range(2):
            if type(tokens[i]) != Expression:
                continue
            if tokens[i].identifier == f"~{tokens[1 - i].identifier}":
                return Token.of(False)
        if same(tokens):
            return Token.of(True)
        if None in values:
            if values[0] == values[1]:
                return Expression.from_tokens([self, Token("<->"), other])
            if values[0] is None:
                return self if values[1] else Expression.from_tokens([Token("~"), self])
            if values[1] is None:
                return other if values[0] else Expression.from_tokens([Token("~"), other])
        if values[0] == values[1]:
            return Token.of(True)
        return Token.of(False)


    def __pow__(self, other):
        tokens = other, self
        values = other.value, self.value
        if same(tokens) or values[0] == False or values[1] == True:
            return Token.of(True)
        if None in values:
            if values[0]:
                return self
            if values[1] == False:
                return -other
            return Expression.from_tokens([other, Token("->"), self])
        return Token.of(bool(values[1] ** values[0]))

    def __neg__(self):
        if self.value is not None:
            return Token.of(not self.value)
        return Expression.from_tokens([Token("~"), self])

    def __and__(self, other):
        tokens = self, other
        values = self.value, other.value
        if False in values:
            return Token.of(False)
        if same(tokens):
            if None in values:
                return tokens[0]
            return Token.of(tokens[0].value)
        for i in range(2):
            if values[i]:
                return tokens[1 - i]
            if type(tokens[i]) is not Expression:
                continue
            if tokens[i].identifier == f"~{tokens[1 - i].identifier}":
                return Token.of(False)
        if values[0] and values[1]:
            return Token.of(True)
        if None in values:
            if values[0]:
                return tokens[1]
            elif values[1]:
                return tokens[0]
            return Expression.from_tokens([Token.of(self.value) if self.value is not None else self, 
                                           Token("&"),
                                           Token.of(other.value) if other.value is not None else other])
        return Token.of(values[0] & values[1])

    def __or__(self, other):
        tokens = self, other
        values = self.value, other.value
        if same(tokens):
            return tokens[0]
        for i in range(2):
            etok = tokens[i]
            if type(tokens[i]) != Expression:
                if etok.value:
                    return Token.of(True)
                if etok.value == False:
                    return tokens[1 - i]
                continue
            if tokens[i].identifier == f"~{tokens[1 - i].identifier}":
                return Token.of(True)
        if None in values:
            if values == (None, None):
                return Expression.from_tokens([self, Token("|"), other])
            return tokens[values.index(None)]
        return Token.of(values[0] | values[1])


class Expression(Token):
    def __init__(self, e):
        self.e = e
        self.type = "expression"
        self.parse()

    @staticmethod
    def from_tokens(tokens):
        e = Expression("a")
        e.tokens = [t.copy() for t in tokens]
        e.e = e.identifier
        return e

    @property
    def true_value(self):
        if len(self.tokens) == 1:
            if type(self.tokens[0]) == Expression:
                return self.tokens[0].true_value
            if self.tokens[0].value is not None:
                return self.tokens[0].value
            return self.tokens[0]
        return self

    @property
    def value(self):
        v = self.true_value
        if v is self:
            return None
        return v

    def parse(self):
        self.tokens = Parser(self.e).parse()

    def __str__(self):
        return f"Expr[{self.identifier.strip('()')}]"

    def __repr__(self):
        return self.__str__()

    @property
    def identifier(self):
        ism = len([t for t in self.tokens if t.type != "operator"]) > 1
        return ("(" * ism + " ".join([t.identifier for t in self.tokens]) + ")" * ism).replace("~ ", "~")

    def calc(self, return_expr=False, **atoms):
        tv = self.evaluate(**atoms)
        if type(tv) == Expression:
            v = Expression.from_tokens(tv.tokens)
        elif type(tv) == Token:
            v = Expression.from_tokens([tv])
        else:
            v = Expression.from_tokens([Token.of(tv)])
        if return_expr:
            return v
        if len(v.tokens) == 1 and v.tokens[0].type == "literal":
            return v.tokens[0].value
        return v.true_value
        
    def evaluate(self, **atoms):
        tokens = [t.copy() for t in self.tokens]
        indices = list(range(len(tokens)))
        atoms = {k: bool(atoms[k]) for k in atoms}
        for i in range(len(tokens)):
            if tokens[i].type == "atom":
                tokens[i] = Token.of(atoms[tokens[i].identifier]) if tokens[i].identifier in atoms else tokens[i]
            elif tokens[i].type == "expression":
                tokens[i] = tokens[i].calc(**atoms)
                if type(tokens[i]) == bool:
                    tokens[i] = Token.of(tokens[i])
        operators = self.operators.copy()
        for op in Token.operators:
            if op not in operators:
                continue
            for pos_ in operators[op]:
                if len(pos_) != 1:
                    continue
                pos = pos_[0]
                index = indices.index(pos)
                nxt = index + 1
                prv = index - 1
                if op == "~":
                    tokens[nxt] = -tokens[nxt]
                    indices.remove(pos)
                    tokens.pop(index)
                    continue
                if op == "&":
                    result = tokens[prv] & tokens[nxt]
                elif op == "|":
                    result = tokens[prv] | tokens[nxt]
                elif op == "->":
                    result = tokens[nxt] ** tokens[prv]
                elif op == "<-":
                    result = tokens[prv] ** tokens[nxt]
                elif op in ["<->", "^"]:
                    if all(type(x) is Expression for x in (tokens[prv], tokens[nxt])):
                        result = are_equal(tokens[prv], tokens[nxt])
                    else:
                        result = tokens[prv] == tokens[nxt]
                    if op == "^":
                        result = -result
                else:
                    raise ValueError(f"invalid operator {op}")
                indices.remove(pos)
                indices.remove(pos - 1)
                tokens[nxt] = result
                tokens.pop(index)
                tokens.pop(prv)
        if len(tokens) == 1:
            return tokens[0]
        return Expression.from_tokens(tokens)

    def copy(self):
        return Expression.from_tokens(self.tokens)

    def table(self, keep=None):
        exp = self.calc()
        if type(exp) == bool:
            return f"Expression \"{self.e}\" is always {exp}"
        has_results = False
        results = []
        d = []
        if exp.identifier != self.identifier:
            results.append(f"Expression converted.\nBefore: {self.identifier}\nAfter:  {exp.identifier}")
        atoms = sorted(list(exp.atoms.keys()))
        results.append("  ".join(atoms + [f"Expression({exp.identifier})"]))
        for dataset in exp.combinations:
            r = exp.calc(**dataset)
            result = int(exp.calc(**dataset))
            d.append(result)
            if keep is not None and int(keep) != result:
                continue
            has_results = True
            results.append("  ".join([str(x) for x in dataset.values()] + [str(result)] * (len(exp.tokens) != 1)))
        if not has_results:
            return f"Expression \"{self.e}\" is always {bool(1 - keep)}"
        s = sum(d)
        if s in [0, len(d)]:
            return f"Expression \"{self.e}\" is always {bool(s)}"
        return "\n".join(results)

    def __eq__(self, other):
        if type(other) != Expression:
            return other == self
        return are_equal(self, other)

    @property
    def combinations(self):
        return combinations(sorted(list(self.atoms.keys())))

    @property
    def atoms(self):
        result = {}
        for token in range(len(self.tokens)):
            identifier = self.tokens[token].identifier
            if self.tokens[token].type == "atom":
                result[identifier] = result.get(identifier, [])
                result[identifier].append((token, ))
            elif self.tokens[token].type == "expression":
                inner_atoms = self.tokens[token].atoms
                for key in inner_atoms:
                    result[key] = result.get(key, [])
                    result[key].extend([(token, ) + k for k in inner_atoms[key]])
        return result

    @property
    def operators(self):
        result = {}
        for token in range(len(self.tokens)):
            identifier = self.tokens[token].identifier
            if self.tokens[token].type == "operator":
                result[identifier] = result.get(identifier, [])
                result[identifier].append((token, ))
            elif self.tokens[token].type == "expression":
                inner_operators = self.tokens[token].operators
                for key in inner_operators:
                    result[key] = result.get(key, [])
                    result[key].extend([(token, ) + k for k in inner_operators[key]])
        return result

    def __getitem__(self, item):
        return self.tokens[item]


class Parser:
    def __init__(self, e):
        self.e = e
        self.i = 0

    @property
    def cur(self):
        return self.e[self.i]

    def parse(self):
        if not self.e:
            raise ParseError("Empty expression")
        state = "atom"
        tokens = []
        while True:
            if self.i >= len(self.e):
                break
            level = 0
            ch = self.cur
            if ch == " ":
                self.i += 1
                continue
            if ch in "01":
                self.i += 1
                tokens.append(Token.of(bool(int(ch))))
                state = "operator"
                continue
            if ch == "-":
                self.i += 1
                if self.cur != ">":
                    raise ParseError(f"Unexpected token: {self.cur}")
                ch += self.cur
            elif ch == "<":
                self.i += 2
                if self.e[self.i - 1: self.i + 1] != "->":
                    if self.e[self.i - 1] == "-":
                        ch += "-"
                    else:
                        raise ParseError(f"Unexpected token: {self.cur}")
                else:
                    ch += "->"

            if ch == "~" or is_letter(ord(ch[0])):
                token_type = "atom"
            elif ch in Token.operators[1:]:
                token_type = "operator"
            elif ch == "(":
                token_type = "expression"
                level += 1
            elif ch == ")":
                raise ParseError("Unmatched \")\"")
            else:
                raise ParseError("Unknown token: " + ch)
            if level:
                si = self.i
                while level:
                    self.i += 1
                    if self.i >= len(self.e):
                        raise ParseError("Unmatched \"(\"")
                    if self.cur == "(":
                        level += 1
                    elif self.cur == ")":
                        level -= 1
                tokens.append(Expression(self.e[si + 1: self.i]))
            else:
                if state != token_type:
                    raise ParseError(f"Unexpected token: {token_type}: {ch}")
                token = Token(ch)
                tokens.append(token)
            state = "operator" if token_type in ["atom", "expression"] and ch != "~" else "atom"
            self.i += 1
        return tokens


class ParseError(Exception):
    pass

class EvaluateError(Exception):
    pass


def proposition(*args):
    premises = args[:-1]
    conclusion = args[-1]
    and_token = Token("&")
    impl_token = Token("->")
    data = []
    for i in range(len(premises)):
        data.append(premises[i])
        data.append(and_token) if i != len(premises) - 1 else None
    data.append(impl_token)
    data.append(conclusion)
    e = Expression.from_tokens(data)
    for ds in e.combinations:
        if not e.calc(**ds):
            return False
    return True 

def are_equal(f1, f2):
    atoms = set(f1.atoms) | set(f2.atoms)
    return all(f1.calc(**ds) == f2.calc(**ds) for ds in combinations(list(atoms)))

def combinations(atoms):
    res = []
    for i in range(2 ** len(atoms)):
        data = {}
        for k in range(len(atoms)):
            mask = 2 ** (len(atoms) - k - 1)
            data[atoms[k]] = int(bool(i & mask))
        res.append(data)
    return res