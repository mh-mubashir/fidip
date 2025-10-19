# Convert SMIL pose prior to GMM format
import pickle
import numpy as np

# Define the Mahalanobis class (needed to load the pickle file)
class Mahalanobis(object):
    def __init__(self, mean, prec, prefix):
        self.mean = mean
        self.prec = prec
        self.prefix = prefix

    def __call__(self, pose):
        if len(pose.shape) == 1:
            return (pose[self.prefix:]-self.mean).reshape(1, -1).dot(self.prec)
        else:
            return (pose[:, self.prefix:]-self.mean).dot(self.prec)

# Load the SMIL pose prior with proper encoding
with open('priors/smil_pose_prior.pkl', 'rb') as f:
    smil_prior = pickle.load(f, encoding='latin1')

# Extract the Mahalanobis object (it's directly the object, not in a dictionary)
mahal_obj = smil_prior

# Create GMM format (single component)
gmm_data = {
    'means': mahal_obj.mean[np.newaxis, :],  # Add batch dimension
    'covars': np.linalg.inv(mahal_obj.prec)[np.newaxis, :, :],  # Convert precision to covariance
    'weights': np.array([1.0])  # Single weight
}

# Save as GMM prior
with open('priors/gmm_01.pkl', 'wb') as f:
    pickle.dump(gmm_data, f)

print("Successfully converted SMIL pose prior to GMM format!")
