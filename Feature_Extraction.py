from math import degrees, sqrt, acos


# Function to calculate angle between two vectors with safety checks, considering hand as origin
def angle_between_fingers(finger1, finger2, hand_position):
    vector1 = [b - a for a, b in zip(hand_position, finger1)]
    vector2 = [b - a for a, b in zip(hand_position, finger2)]
    
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = sqrt(sum(a * a for a in vector1))
    magnitude2 = sqrt(sum(b * b for b in vector2))
    
    # Safety check to ensure the value is within the domain for acos
    value = dot_product / (magnitude1 * magnitude2)
    value = max(min(value, 1.0), -1.0)
    
    angle = acos(value)
    return degrees(angle)

# Function to calculate Euclidean distance between two 3D points
def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Initialize a dictionary to store the corrected features
feature_data = {}

def extractFeatures(gesture_data):
    # Loop through each gesture and frame to extract corrected features for angles between adjacent fingers
    for gesture, frames in gesture_data.items():
        feature_frames = []
        for frame in frames:
            hand_orientation, hand_position, hand_rotation, finger_positions = frame
            
            # Existing features
            feature1 = hand_orientation  # Hand orientation
            feature2 = hand_rotation  # Hand rotation (Euler angles)
            feature3 = [euclidean_distance(hand_position, fingertip) for fingertip in finger_positions]  # Distances from hand to fingertips
            
            #Feature 4: Distance between each fingertip and all other fingertips
            feature4 = []
            for i in range(len(finger_positions)):
                for j in range(i+1, len(finger_positions)):
                    feature4.append(euclidean_distance(finger_positions[i], finger_positions[j]))        
            
            #Feature 5: Angles between adjacent fingers considering hand as origin
            feature5 = []
            for i in range(len(finger_positions) - 1):
                feature5.append(angle_between_fingers(finger_positions[i], finger_positions[i + 1], hand_position))
            
            # Combine all features for this frame
            features = [feature1, feature2, feature3, feature4, feature5]
            feature_frames.append(features)
        
        # Store the corrected features for this gesture
        feature_data[gesture] = feature_frames
    return feature_data

# Show a sample from the extracted features to verify
# print(feature_data['0'][:1][0][4])