from abc import abstractmethod, ABC
import typing as tp

import heapq
from itertools import groupby
import numpy as np
from math import radians, cos, sin, asin, sqrt
import datetime

TRow = tp.Dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


# Operations


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: tp.Sequence[str], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        previous_key: tp.Any = None
        group_block: tp.List[tp.Any] = []
        for row in rows:
            current_key = {k: row[k] for k in self.keys}
            if previous_key is None or current_key == previous_key:
                group_block.append(row)
            else:
                yield from self.reducer(self.keys, group_block)
                group_block = [row]
            previous_key = current_key
        if len(group_block) > 0:
            yield from self.reducer(self.keys, group_block)


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        empty_tuple: tp.Tuple[tp.Any, ...] = tuple()
        empty_iter: tp.Iterator[tp.Any] = iter([])
        empty_gen: tp.Tuple[tp.Any, tp.Iterator[tp.Any]] = (empty_tuple, empty_iter)

        gen_a = groupby(rows, lambda x: [x[k] for k in self.keys])
        gen_b = groupby(args[0], lambda x: [x[k] for k in self.keys])
        key_a, group_a = next(gen_a, empty_gen)
        key_b, group_b = next(gen_b, empty_gen)

        while group_a != empty_iter and group_b != empty_iter:
            if key_a < key_b:
                yield from self.joiner(self.keys, group_a, empty_iter)
                key_a, group_a = next(gen_a, empty_gen)
            elif key_a == key_b:
                yield from self.joiner(self.keys, group_a, group_b)
                key_a, group_a = next(gen_a, empty_gen)
                key_b, group_b = next(gen_b, empty_gen)
            else:
                yield from self.joiner(self.keys, empty_iter, group_b)
                key_b, group_b = next(gen_b, empty_gen)
        while group_a != empty_iter:
            yield from self.joiner(self.keys, group_a, empty_iter)
            key_a, group_a = next(gen_a, empty_gen)
        while group_b != empty_iter:
            yield from self.joiner(self.keys, empty_iter, group_b)
            key_b, group_b = next(gen_b, empty_gen)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: tp.Sequence[tp.Any], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {key: self._lower_case(value) if key == self.column else value for key, value in row.items()}


# Reducers


class Sum(Reducer):
    """Sum values in column passed and yield single row as a result"""
    def __init__(self, column: str) -> None:
        """
        :param column: name of column to sum
        """
        self.column = column

    def __call__(self, group_key: tp.Sequence[str], rows: TRowsIterable) -> TRowsGenerator:
        sum_score = 0
        for row in rows:
            sum_score += row[self.column]
        dict_ = {key: row[key] for key in row if key in group_key}
        dict_[self.column] = sum_score
        yield dict_


# Joiners


class LeftJoiner(Joiner):
    """Join with left strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        new_rows_a = list(rows_a)
        new_rows_b = list(rows_b)
        join_dict: tp.Dict[str, tp.Any] = {}
        for row_a in new_rows_a:
            if len(new_rows_b) == 0:
                yield row_a
            for row_b in new_rows_b:
                join_dict = row_a.copy()
                join_dict.update(row_b)
                yield join_dict

