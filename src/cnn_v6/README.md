
We use the floydhub cloud for building and running this project

## Run project by a command:
$floyd run --gpu --env tensorflow --data quachthanhthoi/datasets/kth/1:kth 'python3 cnn.py 0;python3 cnn.py 1 -m ./model_saver_cnn/model_10.ckpt;python3 lrcn.py 0 -c ./model_saver_cnn/model_10.ckpt;python3 lrcn.py 1 -l ./model_saver_cnn_lstm/model_10.ckpt'

## Descript above command:
- First command will train the CNN model
- Second command will be using for testing the CNN model on dataset by loading the weights from saved model.
- Third command will train LRCN model combining with pretrain CNN model.
- Fourth command will test the LRCN on dataset with the pretrain LRCN model.