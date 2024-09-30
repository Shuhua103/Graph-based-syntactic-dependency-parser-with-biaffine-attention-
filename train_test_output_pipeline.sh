#!/bin/bash
#Execute this pipeline when your current directory is Project_SDP

#Give the path to the train/dev/test dataset, and you can customize the training settings
TRAIN_PATH="sdp_2015/en.dm.sdp.train"
DEV_PATH="sdp_2015/en.dm.sdp.dev"
LEARNING_RATE=0.0005 #We suggest 0.0005 when you freeze the Bert while 0.00002 when you tune the Bert
IF_FREEZE_BERT=true
MAX_EPOCHS=1
MODEL_SAVE_PATH="your_model_trained.pth"
TEST_FILE_1="sdp_2015/en.id.dm.sdp"
TEST_FILE_2="sdp_2015/en.ood.dm.sdp"

#Command line for training
#By executing this, you will get your model and training log saved into the directory
#This will also print the train_loss/dev_loss/F1_arc/F1_label at each epoch for you to observe training situation 
python train_ARC_LABEL.py \
    --path_to_corpus $TRAIN_PATH \
    --path_to_corpus_dev $DEV_PATH \
    --learning_rate $LEARNING_RATE \
    --max_epochs $MAX_EPOCHS \
    --if_freeze_bert $IF_FREEZE_BERT

#Command line to test your trained model's performance on test dataset and output graph
python predict_eval_output_graph.py \
    --model_save_path $MODEL_SAVE_PATH \
    --test_file_1 $TEST_FILE_1 \
    --test_file_2 $TEST_FILE_2

