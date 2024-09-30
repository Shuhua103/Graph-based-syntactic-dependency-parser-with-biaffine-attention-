#!/bin/bash
#Execute this test-and-output script when your current directory is Project_SDP

#Set the path of your saved model(e.g. our best model) and test dataset you want to test and get graph output on
#We suggest you to test your model on our in-domain and out-of-domain dataset
MODEL_SAVE_PATH="model_state_dict_our_best_model.pth"
TEST_FILE_1="sdp_2015/en.id.dm.sdp"
TEST_FILE_2="sdp_2015/en.ood.dm.sdp"

#This command line will give you the performance of your model and generates a file(conllu) of graphs
python predict_eval_output_graph.py \
    --model_save_path $MODEL_SAVE_PATH \
    --test_file_1 $TEST_FILE_1 \
    --test_file_2 $TEST_FILE_2

