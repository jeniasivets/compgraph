from ..examples.text_example import long_word_count_graph
from ..examples.yandex_maps import big_yandex_maps_graph
from operator import itemgetter
import json


def test_corpus() -> None:
    graph = long_word_count_graph('../resource/extract_me/text_corpus.txt', json.loads, 'text', 'count')
    answer = graph.run()
    etalon = [{'text': 'something', 'count': 1396},
              {'text': 'anything', 'count': 869},
              {'text': 'everything', 'count': 849},
              {'text': 'princess', 'count': 684},
              {'text': 'suddenly', 'count': 649},
              {'text': 'gutenbergtm', 'count': 604},
              {'text': 'understand', 'count': 588},
              {'text': 'themselves', 'count': 581},
              {'text': 'together', 'count': 549},
              {'text': 'sometimes', 'count': 549}]

    assert sorted(answer, key=itemgetter('text')) == sorted(etalon, key=itemgetter('text'))


def test_graph() -> None:
    """tests for graph correctness and check we do not miss the processed data,
    added visualization in file average_speed.png"""
    graph = big_yandex_maps_graph('../resource/extract_me/travel_times.txt',
                                  '../resource/extract_me/road_graph_data.txt', json.loads)
    answer = graph.run()

    assert len(answer) == 168

    for i in range(168):
        assert answer[i]['weekday'] in {'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'}
        assert answer[i]['hour'] in set(range(24))
        for j in range(i):
            assert answer[i]['hour'] != answer[j]['hour'] or answer[i]['weekday'] != answer[j]['weekday']


test_corpus()
test_graph()
