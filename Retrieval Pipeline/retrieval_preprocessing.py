
import argparse
import os
import warnings

import pandas as pd
import numpy as np


def users_test_train_split(df, split_ratio=0.15):
    split_len = int(len(df)*split_ratio)
    test_df = df[:split_len]
    train_df = df[split_len:]
    
    return test_df, train_df


if __name__ == "__main__":
    input_data_path_reviews = os.path.join("/opt/ml/processing/input/reviews", "final_reviews.csv")
    
    train_output_path = os.path.join("/opt/ml/processing/train", "train.csv")
    test_output_path = os.path.join("/opt/ml/processing/test", "test.csv")
    
    print("Reading review input data from {}".format(input_data_path_reviews))
    review_df = pd.read_csv(input_data_path_reviews, index_col="Unnamed: 0")
    
    # Shuffle dataframe
    review_df = review_df.sample(frac=1)
    
    # only get users with at least 5 reviews
    users_with_favorable_ratings = (review_df['username'].value_counts()
                                .loc[lambda x: x>10]
                                .loc[lambda x: x<100]
                                .index.values)
    
    review_df = review_df[review_df['username'].isin(users_with_favorable_ratings)]
    
    #Generate test train split
    test_df, train_df = users_test_train_split(review_df)
    
    
    print("Saving Train Data {}".format(train_output_path))
    train_df.to_csv(train_output_path, header=True, index=True)
    
    print("Saving Test Data {}".format(test_output_path))
    test_df.to_csv(test_output_path, header=True, index=True)
    
