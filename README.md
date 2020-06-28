## Computational graph
 
 - Вычислительный граф - заранее заданная последовательность операций, которую можно применять к различным наборам данных.
 
 - Пример графа вычислений, который подсчитывает количество слов в документах, можно увидеть в файле `graphs.py`. В директории `lib` реализован интерфейс графа вычислений и операции типа `map`,` reduce` и `join`. В директории `examples` есть примеры использования графа с текстом, в директории `test_examples` можно найти тесты, написанные под эти примеры.
 
 - A python lib for compgraph, wrote tests, map/reduce operations to work with large scale databases
 - The model is used to solve tasks on text, e.g. tf-idf, pmi, and tasks on Yandex.Map data, counting the average car speed in the city
