
import os
import json
import sys
import numpy as np
import pandas as pd
import pathlib
import tarfile
import tensorflow as tf


if __name__ == "__main__":

    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, "./model")

    model = tf.saved_model.load("./model/01")
    
    test_path = "/opt/ml/processing/test/"
    
    test_df = pd.read_csv(test_path+"/test.csv")
    
    # there is probably a better way to do this -
    users = test_df['username'].unique()
    accuracy = []
    
    for i in range(0, len(users)):
        # Get predictions (as tensors)
        _, beers = model(tf.constant([users[i]]))
        # Convert tensors to numpy array
        beer_list = [x.decode('UTF-8') for x in beers[0].numpy()]
        # Get the users selected beers
        true_list = test_df[test_df['username']==users[i]]['beer_name'].values
        user_accuracy = np.isin(true_list, beer_list)
        accuracy.append(np.count_nonzero(user_accuracy)/len(user_accuracy))
        
    top_500_accuracy = np.array(accuracy).mean()
    print("Top 500 Accuracy: ", top_500_accuracy)

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": top_500_accuracy, "standard_deviation": "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
