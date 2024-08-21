from .lib import Graph, operations


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf') -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    graph1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([doc_column, text_column])

    graph2 = Graph.graph_from_iter(input_stream_name)\
        .sort([doc_column])\
        .reduce(operations.Count('docs_number'), [])

    graph3 = graph1.reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count('words_number'), [text_column])

    graph4 = graph1.reduce(operations.TermFrequency(text_column, 'tf'), [doc_column]) \
        .sort([text_column]) \
        .join(operations.InnerJoiner(), graph3, [text_column])\
        .join(operations.InnerJoiner(), graph2, []) \
        .map(operations.TfIdfMapper('tf', 'docs_number', 'words_number', result_column)) \
        .sort([text_column]) \
        .reduce(operations.TopN(result_column, 3), [text_column]) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([doc_column, text_column])

    return graph4


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi') -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    graph1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)).sort([doc_column, text_column]) \
        .map(operations.Filter(operations.long_word))

    graph2 = graph1.reduce(operations.Count('count'), [doc_column, text_column]) \
        .map(operations.Filter(operations.at_least_two)) \
        .join(operations.InnerJoiner(), graph1, [doc_column, text_column]) \
        .sort([doc_column, text_column])\
        .reduce(operations.TermFrequency(text_column, 'doc_freq'), [doc_column]) \
        .join(operations.InnerJoiner(), graph1, [doc_column, text_column]).sort([text_column])

    graph3 = graph2.reduce(operations.TermFrequency(text_column, 'total_freq'), []) \
        .join(operations.InnerJoiner(), graph2, [text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([doc_column, text_column]) \
        .map(operations.PMIMapper('doc_freq', 'total_freq', result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .reduce(operations.TopN(result_column, 10), [doc_column])

    return graph3


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    dist_graph = Graph.graph_from_iter(input_stream_name_length) \
        .map(operations.HaversineMapper(start_coord_column, end_coord_column, 'distance')) \
        .map(operations.Project([edge_id_column, 'distance'])) \
        .sort([edge_id_column])

    time_graph = Graph.graph_from_iter(input_stream_name_time) \
        .map(operations.TimeProcessMapper(enter_time_column, leave_time_column,
                                          'duration', hour_result_column, weekday_result_column)) \
        .map(operations.Project([edge_id_column, 'duration', hour_result_column, weekday_result_column])) \
        .sort(['edge_id'])

    result_graph = time_graph.join(operations.InnerJoiner(), dist_graph, [edge_id_column]) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.MultiSum(['duration', 'distance']), [weekday_result_column, hour_result_column]) \
        .map(operations.SpeedMapper('distance', 'duration', speed_result_column)) \
        .map(operations.Project([weekday_result_column, hour_result_column, speed_result_column]))

    return result_graph
