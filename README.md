# matlog

**This module is well-documented with docstrings and type hints. Use `help(matlog.tokens)` or your editor hints to get familiar with it.**

Basic usage:    
```python
from matlog import Expression

expr = Expression("A & B")
print(expr.solve(A=1, B=0)) # False
print(expr.solve(B=0)) # False
print(expr.table()) # prints the truth table of expression
```

## Contents
+ [Expression string](#expression-string)
+ [Using Expression object](#expression)
  + [Truth tables](#truth-tables)
  + [Evaluating expression](#evaluating-expression)
  + [Partial evaluation](#partial-evaluation)
  + [Equality check](equality-check)
  + [proposition](proposition)

## Expression string
An expression string is a bunch of tokens (atoms, expressions, operators and literals) in a specific order.
There's a big example of expression string:    
`a | (b -> c) == (~a ^ 1) <- ~z`

Let's review in detail:    

### Atoms

Any letter is considered as an atom. Atoms can be only one letter long and must be divided by operators.

### Literals

A literal is `1` or `0` (True and False respectively). Literals are treated like atoms.

### Expressions

Any expression can contain nested expression strings in brackets. Nested expressions are treated like atoms.

### Operators

*matlog* expression syntax supports 6 binary operators and one unary (listed in order of priority, from high to low):
+ **~** (not / negation) *unary*: turns **True** to **False** and vice versa. 
+ **&** (and / conjunction): **True** if (and only if) both operands are **True**
+ **|** (or / disjunction): **True** if (and only if) any operand is **True**.
+ **->** (implication): **False** if (and only if) the left operand is **True** but the right is **False**
+ **==** (equality / biconditional): **True** if (and only if) operands are equal to each other
+ **^** (xor / exclusive disjunction): **False** if (and only if) operands are equal to each other
+ **<-** (converse implication): **->** with reversed operands order

## Expression

Expression class is the main class of the module.
There are three ways to create an Expression object:
+ from an [expression string](#expression-string): `Expression(expr_str)`
+ copying an existing Expression: `expr.copy()` (same as `Expression(expr.tokens)`)
+ deep-copying an existing Expression object: `expr.deep_copy()`
+ constructing an Expression from tokens (**module won't check if it is valid in this case**): `Expression([token1, token2...])`

### Truth tables 

You can use `.table(keep=None)` method to build a truth table for an Expression object.
`keep` parameter can be either `1` or `0` (to filter rows with corresponding values) or `None` if you need a full table    

### Evaluating expression

If you need a value for a specific set of values, you can use `.solve()` method like this:

```python
from matlog import Expression
expr = Expression("A & B | C")
print(expr.solve(A=1, B=0, C=1)) # prints (1)
print(expr.solve({"A": 1, "B": 0, "C": 1})) # you can pass a dictionary too
```

if you need a result value (release version would provide more convenient ways, of course):

```python
expr.value(A=1, B=0, C=1) # 1 (raises an exception if there's not enough data to solve expression)
```

### Partial evaluation

If you know some (but not all) values, you can also use `.solve()`, providing some values:

```python
from matlog import Expression
expr = Expression("A & B | C")
print(expr.solve(B=0)) # prints (C)
print(expr.solve({"B": 0})) # prints (C) too
``` 

### Equality check

If you're wondering if expressions are equal (producing the same results with any set of values), you can use `Expression.equals()` method:

```python
from matlog import Expression, are_equal
expr1 = Expression("A & B")
expr2 = Expression("B & A")
print(expr1.equals(expr2)) # True
```

### proposition [unavailable as of 2.0pre3]

Module can solve propositions:
```
Example:
Premise 1: P → Q 
Premise 2: P 
Conclusion: Q 
```
Checking that with module:
```python
from matlog import Expression, proposition

premise1 = Expression("P -> Q")
premise2 = Expression("P") # or Token("P")
conclusion = Expression("Q") # or Token("Q")

proposition(premise1, premise2, conclusion) # True
```