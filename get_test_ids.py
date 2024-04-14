import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    main = pd.read_csv('data/train.csv')
    train = main[main.target != -1]
    train_, test_ = train_test_split(train, random_state=42, test_size=0.2)
    test_[['user_id']].to_csv('data/test_ids.csv', index=False)
