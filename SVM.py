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


def CIFAR10_loss_fun(W):
	return L(X_train,Y_train,W)


# instead of evaluating on the whole dataset for updating one 
#parameter use mini batch gradient descent and use calculus instead of numerical approach
def eval_numerical_gradient(f,W):
	fx = f(W)
	W1 = W
	W2 = W
	h = 0.0001
	grad = np.zeros(W.shape)
	it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		old_value = W[idx]
		W1[idx] += h
		W2[idx] -= h
		fx_new1 = f(W1)
		fx_new2 = f(W2)
		grad[idx] = (fx_new1-fx_new2)/(2*h)
		W1[idx] = old_value
		W2[idx] = old_value
		it.iternext()

	return grad

# W = (10,3073) X_test = (3073,10000), 3073 not 3072 because of the bias
def predict(W,X_test):
	scores = W.dot(X_test)
	scores.transpose()
	Y_pred = np.zeros(X_test.shape[1])
	for i in range(X_test.shape[1]):
		test_scores_for_each_image = scores[i]
		Y_pred[i] = np.argmax(test_scores_for_each_image)

	return Y_pred

W = np.random.rand(10, 3073) * 0.001
df = eval_numerical_gradient(CIFAR10_loss_fun, W)

best_loss = CIFAR10_loss_fun(W)
best_weights = W

for steps in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
	step_size = 10 ** steps
	W_new = W - step_size*df
	new_loss = CIFAR10_loss_fun(W_new)
	if new_loss < best_loss:
		best_loss = new_loss
		best_weights = W



predictions = predict(best_weights, X_test)

print(np.mean(predictions == Y_test))










	