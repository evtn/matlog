from typing import List, Any, Union



def cut_literal(value: Union[int, bool]) -> int:
    return int(bool(value))


def combinations(letters: List[Any]) -> None:
    count = len(letters)
    for i in range(2 ** count):
        yield {
            letters[x]: cut_literal((2 ** x) & i)
            for x in range(count)
        }


class Table:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return "\n".join([
            " ".join(self.data["identifiers"]),
            *map(
                lambda row: " ".join(
                    map(
                        lambda i: str(row[i]),
                        self.data["identifiers"]
                    )
                ),
                self.data["values"]
            )
        ])

    def __repr__(self):
        return str(self)