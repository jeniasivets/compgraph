from compgraph.lib.graph import Graph
from compgraph.lib import operations as ops
import typing as tp


def big_yandex_maps_graph(filename_time: str, filename_length: str, parser: tp.Callable[[str], ops.TRow],
                           enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                           edge_id_column: str = 'edge_id', start_coord_column: str = 'start',
                           end_coord_column: str = 'end', weekday_result_column: str = 'weekday',
                           hour_result_column: str = 'hour', speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    dist_graph = Graph.graph_from_file(filename_length, parser) \
        .map(ops.HaversineMapper(start_coord_column, end_coord_column, 'distance')) \
        .map(ops.Project([edge_id_column, 'distance'])) \
        .sort([edge_id_column])

    time_graph = Graph.graph_from_file(filename_time, parser) \
        .map(ops.TimeProcessMapper(enter_time_column, leave_time_column,
                                   'duration', hour_result_column, weekday_result_column)) \
        .map(ops.Project([edge_id_column, 'duration', hour_result_column, weekday_result_column])) \
        .sort(['edge_id'])

    result_graph = time_graph.join(ops.InnerJoiner(), dist_graph, [edge_id_column]) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(ops.MultiSum(['duration', 'distance']), [weekday_result_column, hour_result_column]) \
        .map(ops.SpeedMapper('distance', 'duration', speed_result_column)) \
        .map(ops.Project([weekday_result_column, hour_result_column, speed_result_column]))

    return result_graph


