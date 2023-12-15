def convert_to_one_hot(y):
    max_value = max(y)
    return [1 if y[i] == max_value else 0 for i in range(len(y))]


def get_one_hot(keys):
    length = len(keys)
    keys_in_one_hot = {}
    for (index, key) in enumerate(keys):
        one_hot = [0. for _ in range(length)]
        one_hot[index] = 1.
        keys_in_one_hot[key] = one_hot
    return keys_in_one_hot


def one_hot_to_string(keys):
    for i in range(len(keys)):
        if keys[i] == 1:
            return i
