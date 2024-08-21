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


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {key: ''.join([symbol for word in value for symbol in word if symbol.isalpha() or symbol == " "])
               if key == self.column else value for key, value in row.items()}


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


class Split(Mapper):
    """Split row on multiple rows by separator"""
    def __init__(self, column: str, separator: tp.Optional[str] = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        values_in_row = [v.split() for k, v in row.items() if k == self.column][0]
        for val in values_in_row:
            yield {key: val if key == self.column else value for key, value in row.items()}


class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: tp.Sequence[str], result_column: str) -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        product = 1
        for key in self.columns:
            product *= row[key]
        row[self.result_column] = product
        yield {k: v for k, v in row.items()}


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield {key: value for key, value in row.items()}
        else:
            pass


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {key: value for key, value in row.items() if key in self.columns}


class TfIdfMapper(Mapper):
    """Count tf-idf using three columns"""
    def __init__(self, frequency: str,
                 total_doc_count: str,
                 word_doc_count: str,
                 result: str):
        self.frequency = frequency
        self.total_doc_count = total_doc_count
        self.word_doc_count = word_doc_count
        self.result = result

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result] = row[self.frequency] * np.log(row[self.total_doc_count] / row[self.word_doc_count])
        yield row


class PMIMapper(Mapper):
    """Count pmi using two columns"""
    def __init__(self, doc_freq: str, total_freq: str, result: str):
        self.doc_freq = doc_freq
        self.total_freq = total_freq
        self.result = result

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result] = np.log(row[self.doc_freq] / row[self.total_freq])
        yield row


def at_least_two(row: TRow) -> bool:
    """filter for pmi task"""
    return row["count"] > 1


def long_word(row: TRow) -> bool:
    """filter for pmi task"""
    return len(row['text']) > 4


# function from https://stackoverflow.com/questions/4913349/
# /haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6373
    return c * r


class HaversineMapper(Mapper):
    """Counts haversine distance in kilometers between two points using their coordinates"""

    def __init__(self, start_point: str, end_point: str, result_length: str):
        self.start_point = start_point
        self.end_point = end_point
        self.result_length = result_length

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_length] = haversine(row[self.start_point][0], row[self.start_point][1],
                                            row[self.end_point][0], row[self.end_point][1])
        yield row


class TimeProcessMapper(Mapper):
    """Counts duration of trip in hours, its weekday and daytime"""

    def __init__(self, enter_time: str, leave_time: str, duration: str,
                 daytime: str, weekday: str):
        self.enter_time = enter_time
        self.leave_time = leave_time
        self.duration = duration
        self.daytime = daytime
        self.weekday = weekday

    def __call__(self, row: TRow) -> TRowsGenerator:
        weekday_array: tp.List[str] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        entry_str = row[self.enter_time].replace('T', '.')
        if len(entry_str) == 15:
            entry_str += '.000000'
        entry = datetime.datetime.strptime(entry_str, '%Y%m%d.%H%M%S.%f')
        row[self.weekday] = weekday_array[entry.weekday()]
        row[self.daytime] = entry.hour
        quit_str = row[self.leave_time].replace('T', '.')
        if len(quit_str) == 15:
            quit_str += '.000000'
        quit = datetime.datetime.strptime(quit_str, '%Y%m%d.%H%M%S.%f')
        timedelta = quit - entry
        row[self.duration] = timedelta.days * 24 + (timedelta.seconds + timedelta.microseconds / 1000000) / 3600
        yield row


class SpeedMapper(Mapper):
    """Counts speed using distance and duration"""

    def __init__(self, distance: str, duration: str, speed: str):
        self.distance = distance
        self.duration = duration
        self.speed = speed

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.speed] = row[self.distance] / row[self.duration]
        yield row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""
    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tp.Sequence[str], rows: TRowsIterable) -> TRowsGenerator:
        """to construct heap with rows sorted by row[self.column_max] with maximum value
         in the top we have to push 3-tuple:
         -row[self.column_max] as a value we compare rows by,
         np.random.uniform() as a value which is unique for each element of heap
         to prevent comparing dicts with < or >,
         row to use
         """
        heap: tp.Any = []
        for row in rows:
            heap.append((-row[self.column_max], np.random.uniform(), row))
        heapq.heapify(heap)
        for i in range(min(self.n, len(heap))):
            yield heapq.heappop(heap)[2]


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str) -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tp.Sequence[str], rows: TRowsIterable) -> TRowsGenerator:
        number_in_group = 0
        dict_: tp.Dict[str, tp.Any] = {}
        for row in rows:
            if row[self.words_column] not in dict_:
                dict_[row[self.words_column]] = 1
            else:
                dict_[row[self.words_column]] += 1
            number_in_group += 1
        for k, v in dict_.items():
            answer_dict = {key: row[key] for key in group_key}
            answer_dict[self.words_column] = k
            answer_dict[self.result_column] = v / number_in_group
            yield answer_dict


class Count(Reducer):
    """Count rows passed and yield single row as a result"""
    def __init__(self, column: str) -> None:
        """
        :param column: name of column to count
        """
        self.column = column

    def __call__(self, group_key: tp.Sequence[str], rows: TRowsIterable) -> TRowsGenerator:
        counter = 0
        for row in rows:
            counter += 1
        dict_ = {key: row[key] for key in row if key in group_key}
        dict_[self.column] = counter
        yield dict_


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


class MultiSum(Reducer):
    """Sum values in multiple columns passed and yield single row as a result"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param column: name of column to sum
        """
        self.columns = columns

    def __call__(self, group_key: tp.Sequence[str], rows: TRowsIterable) -> TRowsGenerator:
        sum_score = {key: 0 for key in self.columns}
        for row in rows:
            for key in self.columns:
                sum_score[key] += row[key]
        dict_ = {key: row[key] for key in row if key in group_key}
        for key in self.columns:
            dict_[key] = sum_score[key]
        yield dict_


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        new_rows_b = list(rows_b)
        for row_a in rows_a:
            for row_b in new_rows_b:
                join_dict: tp.Dict[str, tp.Any] = {}
                intersection = set(row_a.keys()).intersection(set(row_b.keys()))
                for key in set(row_a.keys()).difference(intersection):
                    join_dict[key] = row_a[key]
                for key in set(row_b.keys()).difference(intersection):
                    join_dict[key] = row_b[key]
                for key in keys:
                    join_dict[key] = row_a[key]
                for key in intersection.difference(set(keys)):
                    join_dict[key + self._a_suffix] = row_a[key]
                    join_dict[key + self._b_suffix] = row_b[key]
                yield join_dict


class OuterJoiner(Joiner):
    """Join with outer strategy"""
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
        if len(new_rows_a) == 0:
            for row_b in new_rows_b:
                yield row_b


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


class RightJoiner(Joiner):
    """Join with right strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        new_rows_a = list(rows_a)
        new_rows_b = list(rows_b)
        join_dict: tp.Dict[str, tp.Any] = {}
        for row_b in new_rows_b:
            if len(new_rows_a) == 0:
                yield row_b
            for row_a in new_rows_a:
                join_dict = row_b.copy()
                join_dict.update(row_a)
                yield join_dict
