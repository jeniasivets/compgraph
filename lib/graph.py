import typing as tp
from . import operations as ops

from . import external_sort as sort_


import copy


class Queue:
    def __init__(self, sequence: tp.List[tp.Any]) -> None:
        self.back_sequence: tp.List[tp.Any] = []
        self.front_sequence: tp.List[tp.Any] = list(sequence[::-1])

    def push(self, element: tp.Any) -> None:
        self.back_sequence.append(element)

    def pop(self) -> tp.Any:
        if len(self.front_sequence) == 0:
            self.front_sequence = list(self.back_sequence[::-1])
            self.back_sequence = []
        return self.front_sequence.pop()

    def size(self) -> int:
        return len(self.back_sequence) + len(self.front_sequence)


class Graph:
    """Computational graph implementation"""

    def __init__(self) -> None:
        self.queue = Queue([])
        self.kwarg_name: str
        self.parser: tp.Callable[[str], ops.TRow]

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        :param name: name of kwarg to use as data source
        """
        graph = Graph()
        graph.kwarg_name = name
        return graph

    @staticmethod
    def read_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> ops.TRowsGenerator:
        for line in open(filename):
            yield parser(line)

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        graph = Graph()
        graph.parser = parser
        graph.kwarg_name = filename
        return graph

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        graph = copy.deepcopy(self)
        graph.queue.push([ops.Map(mapper)])
        return graph

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        graph = copy.deepcopy(self)
        graph.queue.push([ops.Reduce(reducer, keys)])
        return graph

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        graph = copy.deepcopy(self)
        graph.queue.push([sort_.ExternalSort(keys)])
        return graph

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        graph = copy.deepcopy(self)
        graph.queue.push([ops.Join(joiner, keys), join_graph])
        return graph

    def gen_run(self, **kwargs: tp.Any) -> ops.TRowsGenerator:
        graph = copy.deepcopy(self)
        if graph.kwarg_name[-4:] == '.txt':
            object_ = Graph.read_file(graph.kwarg_name, self.parser)
        else:
            object_ = kwargs[graph.kwarg_name]()
        while graph.queue.size() > 0:
            op = graph.queue.pop()
            if len(op) == 1:
                object_ = op[0](object_)
            else:
                another_object = op[1].gen_run(**kwargs)
                object_ = op[0](object_, another_object)
        return object_

    def run(self, **kwargs: tp.Any) -> tp.List[ops.TRow]:
        """Single method to start execution; data sources passed as kwargs"""
        return list(self.gen_run(**kwargs))
