create:
	python3 -W ignore ./create_graph.py
create_copy:
	python3 -W ignore ./create_graph_copy.py
train: create
	python3 -W ignore ./train.py
train_copy: create_copy
	python3 -W ignore ./train_copy.py
test:
	python3 ./test.py -i data/dog.jpg
clean:
	rm -r graph
	rm -r train_graph
