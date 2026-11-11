"""Small, dependency-free checks shared by dataset consumers."""


def validate_split(split):
    train = split['train_episode_indices']
    test = split['test_episode_indices']
    total = split['total_episodes']
    if (split.get('status') != 'prepared' or not train or not test
            or any(type(i) is not int for i in train + test)
            or len(train) != len(set(train)) or len(test) != len(set(test))
            or set(train) & set(test)
            or set(train) | set(test) != set(range(total))
            or len(train) != split['train_episode_count']
            or len(test) != split['test_episode_count']
            or split['train_frames'] + split['test_frames'] != split['total_frames']):
        raise ValueError('Invalid episode split: train/test must be nonempty, disjoint and cover the dataset')
    if split.get('normalization_scope') != 'training episodes only':
        raise ValueError('Normalization must use training episodes only')
    return train, test
