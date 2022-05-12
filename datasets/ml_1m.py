from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Office_Products.csv')
        df = pd.read_csv(file_path, usecols=[
                         'rating', 'reviewerID', 'product_id', 'date'])
        df = df.iloc[:500000, :]
        df.rename(columns={'reviewerID': 'user_id'}, inplace=True)
        df.loc[:, 'rating'] = df.loc[:, 'rating'].apply(lambda x: float(x))
        df.columns = ['rating', 'uid', 'sid', 'timestamp']
        return df
