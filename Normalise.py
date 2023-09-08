import numpy as np

def flatten_features(frame):
    flat_frame = []
    for feature in frame:
        if isinstance(feature, list):
            flat_frame.extend(feature)
        else:
            flat_frame.append(feature)
    return flat_frame

separated_data_non_normalized = []
separated_data_normalized = []
labels = []

def norm(feature_combinations):
    # Separate and normalize the features
    for gesture, combinations in feature_combinations.items():
        for combination_name, frames in combinations.items():
            for frame in frames:
                # Separate 'Hand Orientation' from numerical features
                hand_orientation = frame[0]
                numerical_features = flatten_features(frame[1:])
                
                # Normalize the numerical features
                numerical_features_normalized = (numerical_features - np.mean(numerical_features)) / np.std(numerical_features) if np.std(numerical_features) != 0 else np.zeros_like(numerical_features)
                
                # Re-combine the 'Hand Orientation' with the non-normalized and normalized numerical features
                separated_data_non_normalized.append([hand_orientation] + numerical_features)
                separated_data_normalized.append([hand_orientation] + numerical_features_normalized.tolist())
                
                labels.append(gesture)
                
    return [separated_data_non_normalized,separated_data_normalized,labels]