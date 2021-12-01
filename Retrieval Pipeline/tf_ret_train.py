

import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text
import argparse
import numpy as np
import json
import os
import pandas as pd



# disable tf logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # Hyperparameters- sent by clientt passed as command line args to script
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--returned_recommendations', type=int, default=500)
    
    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    
    # model directory - /opt/ml/model   by default for sagemaker
    #parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    
    return parser.parse_known_args()


def get_train_data(train_dir):
    print("training_directory", train_dir)
    df_train = pd.read_csv(os.path.join(train_dir, 'train.csv'), index_col="Unnamed: 0")

    print('x train: ', np.shape(df_train))
    return df_train



def df_to_tensor(df):
    
    df_beer = df['beer_name'].unique()
    df_beer = pd.DataFrame(df_beer, columns = ['beer_name'])
    
    df_ratings = df[['username', 'beer_name']]
    df_ratings = df_ratings.dropna()
    
    # convert dataframes to tensors
    tf_beer_dict = tf.data.Dataset.from_tensor_slices(dict(df_beer))
    tf_ratings_dict = tf.data.Dataset.from_tensor_slices(dict(df_ratings))
    
    # map rows to a dictionary
    ratings = tf_ratings_dict.map(lambda x: {
        "beer_name": x["beer_name"],
        "username": x["username"]
    })
    beer_list = tf_beer_dict.map(lambda x: x['beer_name'])
    
    print('converted df to tensors')
    return ratings, beer_list


def get_unique_beers_and_users(ratings, beer_list):
    usernames = ratings.map(lambda x: x['username'])
    unique_users = np.unique(np.concatenate(list(usernames.batch(1000))))
    unique_beers = np.unique(np.concatenate(list(beer_list.batch(1000))))

    print("unique users: ", len(unique_users), "unique_beers: ", len(unique_beers))
    return unique_users, unique_beers

    
def test_train_split(ratings, df):
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(len(df), seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(int(len(df)*0.8))
    test = shuffled.skip(int(len(df)*0.8)).take(int(len(df)*0.2))
    print("test data len: ", len(test), "train data len: ", len(train))
    return test, train
    
    
# extend the tfrs class
class BeerRetreival(tfrs.Model):
    def __init__(self):
        super().__init__()
        
        embedding_dims = 32
        self.user_model =  tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary= unique_users, mask_token=None),
            tf.keras.layers.Embedding(len(unique_users)+1, embedding_dims)
        ])

        self.beer_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_beers, mask_token=None),
            tf.keras.layers.Embedding(len(unique_beers)+1, embedding_dims)
        ])

        self.task = tfrs.tasks.Retrieval(
                        metrics=tfrs.metrics.FactorizedTopK(
                        candidates=beer_list.batch(128).cache().map(self.beer_model)
                        ))
        
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features['username'])
        beer_embeddings = self.beer_model(features['beer_name'])
        return self.task(user_embeddings, beer_embeddings)
        
        

if __name__ == "__main__":

    args, _ = parse_args()
    

    print('Training data location: {}'.format(args.train))
    
    df_train = get_train_data(args.train)
    
    ratings, beer_list = df_to_tensor(df_train)
    unique_users, unique_beers = get_unique_beers_and_users(ratings, beer_list)
    test, train = test_train_split(ratings, df_train)

    
    returned_recommendations = args.returned_recommendations
    epochs = args.epochs
    learning_rate = args.learning_rate
    #returned_recommendations = 500
    #epochs = 4
    #learning_rate = 0.5
    print('returned reccomendations = {}, epochs = {}, learning rate = {}'.format(returned_recommendations, epochs, learning_rate))
    
    # create + train model
    model = BeerRetreival()
    optimizer = tf.keras.optimizers.Adagrad(learning_rate)
    model.compile(optimizer)
    model.fit(train.batch(8192),
             validation_data = test.batch(512),
             validation_freq = 2,
             epochs = epochs,
             verbose = 0)

    # Eval model
    scores = model.evaluate(test.batch(8192), return_dict=True, verbose=0)

    print("top 10 score: ", scores['factorized_top_k/top_10_categorical_accuracy'])
    print("top 50 score: ", scores['factorized_top_k/top_50_categorical_accuracy'])
    print("top 100 score: ", scores['factorized_top_k/top_100_categorical_accuracy'])

    #save model - need to call first
    brute_force = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=500)
    brute_force.index_from_dataset(
        beer_list.batch(128).map(lambda beer_name: (beer_name, model.beer_model(beer_name)))
    )

    _ = brute_force(np.array(["pblackburn"]))
    
    
    if args.current_host == args.hosts[0]:
        
        print("Host arg:", args.hosts[0])
        # save model to an S3 directory with version number '/1' in Tensorflow SavedModel Format
        print("Saving Model to: ", args.sm_model_dir)
        tf.saved_model.save(
          brute_force,
          os.path.join(args.sm_model_dir, "01"))
        print("Saved Model to: ", args.sm_model_dir)
        
