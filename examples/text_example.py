from compgraph.lib.graph import Graph
from compgraph.lib import operations as ops
import typing as tp


def very_long_word(row: ops.TRow) -> bool:
    return len(row['text']) > 7

def long_word_count_graph(filename: str, parser: tp.Callable[[str], ops.TRow],
                          text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which returns 10 most popular long words in text_column of all rows passed"""
    return Graph.graph_from_file(filename, parser) \
        .map(ops.FilterPunctuation(text_column)) \
        .map(ops.LowerCase(text_column)) \
        .map(ops.Split(text_column)) \
        .sort([text_column]) \
        .reduce(ops.Count(count_column), [text_column]) \
        .sort([count_column, text_column]) \
        .map(ops.Filter(very_long_word)).reduce(ops.TopN(count_column, 10), [])

