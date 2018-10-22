import numpy as np
from collections import Counter

#we are using pixel difference for getting similarities and not semantic similarity

#L1 and L2 distances (or equivalently the L1/L2 
# norms of the differences between a pair of images) are the most commonly used special cases of a p-norm.

# Clearly, the pixel-wise distance does not correspond at all to perceptual or semantic similarity. 

# Summary

# In summary:

#     We introduced the problem of Image Classification, in which we are given a set of images that are all labeled with a single category. We are then asked to predict these categories for a novel set of test images and measure the accuracy of the predictions.
#     We introduced a simple classifier called the Nearest Neighbor classifier. We saw that there are multiple hyper-parameters (such as value of k, or the type of distance used to compare examples) that are associated with this classifier and that there was no obvious way of choosing them.
#     We saw that the correct way to set these hyperparameters is to split your training data into two: a training set and a fake test set, which we call validation set. We try different hyperparameter values and keep the values that lead to the best performance on the validation set.
#     If the lac 	k of training data is a concern, we discussed a procedure called cross-validation, which can help reduce noise in estimating which hyperparameters work best.
#     Once the best hyperparameters are found, we fix them and perform a single evaluation on the actual test set.
#     We saw that Nearest Neighbor can get us about 40% accuracy on CIFAR-10. It is simple to implement but requires us to store the entire training set and it is expensive to evaluate on a test image.
#     Finally, we saw that the use of L1 or L2 distances on raw pixel values is not adequate since the distances correlate more strongly with backgrounds and color distributions of images than with their semantic content.

# In next lectures we will embark on addressing these challenges and eventually arrive at solutions that give 90% accuracies, allow us to completely discard the training set once learning is complete, and they will allow us to evaluate a test image in less than a millisecond.

# Summary: Applying kNN in practice

# If you wish to apply kNN in practice (hopefully not on images, or perhaps as only a baseline) proceed as follows:

#     Preprocess your data: Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit variance. We will cover this in more detail in later sections, and chose not to cover data normalization in this section because pixels in images are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.
#     If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA (wiki ref, CS229ref, blog ref) or even Random Projections.
#     Split your training data randomly into train/val splits. As a rule of thumb, between 70-90% of your data usually goes to the train split. This setting depends on how many hyperparameters you have and how much of an influence you expect them to have. If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. If you are concerned about the size of your validation data, it is best to split the training data into folds and perform cross-validation. If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).
#     Train and evaluate the kNN classifier on the validation data (for all folds, if doing cross-validation) for many choices of k (e.g. the more the better) and across different distance types (L1 and L2 are good candidates)
#     If your kNN classifier is running too long, consider using an Approximate Nearest Neighbor library (e.g. FLANN) to accelerate the retrieval (at cost of some accuracy).
#     Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be burned on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.


class NearestNeighbour(object):
	def __init__(self):
		pass

	def train(self,X,y):
		self.Xtr = X
		self.ytr = y

	def predict(self,X):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test)
		for i in range(0,num_test):
			distance = np.sum(np.abs(self.Xtr - X[i,:]),axis=1)  #L1 norm
			#distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))  # L2 norm
			min_index = np.argmin(distance)  #only the first occurence is returned in case of multiple values
			Ypred[i] = self.ytr[min_index]
			print("Ypred[{}] = ".format(i),Ypred[i])
		return Ypred

class KNearestNeighbour(object):
	def __init__(self):
		pass

	def train(self,X,y):
		self.Xtr = X
		self.ytr = y

	def predict(self,X,k):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test)
		for i in range(0,num_test):
			distance = np.sum(np.abs(self.Xtr - X[i,:]),axis=1)  #L1 norm
			#distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))  # L2 norm
			k_min_indexes = np.argpartition(distance,k)  
			print("k_min_indexes = ", k_min_indexes)
			kNearNeighbour = [self.ytr[i] for i in k_min_indexes[:k]]
			kNearNeighbourWithCount = Counter(kNearNeighbour)
			Ypred[i] = kNearNeighbourWithCount.most_common(1)[0][0]
			print("Ypred[{}] = ".format(i),Ypred[i])
		return Ypred