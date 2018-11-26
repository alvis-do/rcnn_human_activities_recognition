create: 
	python3 -W ignore ./create_graph.py
train_cnn: create
	python3 -W ignore ./train_cnn.py
train_all: create
	python3 -W ignore ./train_all.py
clean:
	rm -r graph
	rm -r model_saver_cnn
	rm -r model_saver_cnn_lstm
