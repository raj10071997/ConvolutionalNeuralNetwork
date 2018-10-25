import numpy as np


#function for one example score
def L_i(x,y,W):   #unvectorized version
	delta = 1.0
	scores = W.dot(x)
	correct_class_score = scores[y]
	loss = 0;
	t = len(scores)
	for i in range(t):
		if (i!=y):
			loss += max(0,scores[i]-correct_class_score+delta)

	return loss

#function for one example score
def L_i_Vectorized(x,y,W):
	delta = 1.0
	scores = W.dot(x)
	margins = np.maximum(0,scores-scores[y]+delta)
	margins[y]=0
	loss = np.sum(margins)
	return loss


#function for complete trainning data loss
def L(X,Y,W):
	delta = 1.0
	scores = W.dot(X)
	scores.transpose()
	Y.transpose() # if Y's dimension is 500000*1, otherwise comment this line
	num_train = len(Y)
	loss = np.zeors(num_train)
	for i in range(num_train):
		train_example_scores_for_each_image = scores[i]
		margins = np.maximum(0,train_example_scores_for_each_image-
			train_example_scores_for_each_image[Y[i]]+delta)
		margins[Y[i]] = 0
		loss[i] = np.sum(margins)

	return loss	





	