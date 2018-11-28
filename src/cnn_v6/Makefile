create:
	python3 -W ignore ./create_graph.py
train_cnn:
	python3 -W ignore ./train_test_cnn.py --train
test_cnn:
	python3 -W ignore ./train_test_cnn.py --test
train_all:
	python3 -W ignore ./train_test_all.py --train
test_all: 
	python3 -W ignore ./train_test_all.py --test
clean:
	rm -f -r graph
	rm -f -r model_saver_cnn
	rm -f -r model_saver_cnn_lstm
