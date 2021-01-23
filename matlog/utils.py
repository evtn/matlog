from typing import List, Any, Union



def cut_literal(value: Union[int, bool]) -> int:
    return int(bool(value))


def combinations(letters: List[Any]) -> None:
    count = len(letters)
    for i in range(2 ** count):
        yield {
            letters[x]: cut((2 ** x) & i)
            for x in range(count)
        }