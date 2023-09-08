

feature_combinations = {}

def combinations(feature_data):
    combinations = {
        'Combination 1': [0, 1, 2],  # Features 1, 2, 3
        'Combination 2': [2, 3, 4],  # Features 3, 4, 5
        'Combination 3': [2, 1, 3],  # Features 3, 2, 4
        'Combination 4': [0, 1, 2, 3, 4]  # All Features
    }

    # Create the feature combinations for each gesture and each frame
    for gesture, frames in feature_data.items():
        feature_combinations[gesture] = {}
        for comb_name, comb_indices in combinations.items():
            feature_combinations[gesture][comb_name] = []
            for frame in frames:
                combined_features = [frame[i] for i in comb_indices]
                feature_combinations[gesture][comb_name].append(combined_features)
    return feature_combinations