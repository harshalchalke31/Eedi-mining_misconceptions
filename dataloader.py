import pandas as pd

def load_and_merge_data(train_path, test_path, misconception_path):
    # Load the data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    misconception_df = pd.read_csv(misconception_path)
    
    # Merge misconception descriptions with train and test data
    for col_suffix in ['A', 'B', 'C', 'D']:
        col_name = f'Misconception{col_suffix}Id'
        misconception_df.rename(columns={'MisconceptionId': col_name}, inplace=True)
        train_df = train_df.merge(misconception_df, how='left', on=col_name, suffixes=('', f'_{col_suffix}'))
        test_df = test_df.merge(misconception_df, how='left', on=col_name, suffixes=('', f'_{col_suffix}'))
    
    return train_df, test_df
