import math
import numpy as np
import json

class Evaluation:

	def compute_rank1(self, Y, y):

		print("Y")
		print(len(Y))
		print(len(Y[1]))
		print(Y)
		print("y")
		print(y)
		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0]
				imin = idx1[smin]
				count_all += 1
				if imin:
					count_correct += 1
		return count_correct/count_all*100

	def compute_rank1_nn(self, Y, y):
		count_all = 0
		count_correct = 0
		for idx, preds in enumerate(Y):
			maxIx = np.argmax(preds,axis=0) + 1
			print("maxIx -> ", maxIx)
			print("detected -> ", y[idx])
			if (maxIx == y[idx]):
				count_correct += 1
			count_all += 1

	
		rank1 = count_correct/count_all
		

		print("correctly indentified -> ", count_correct)
		return rank1*100

	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

		# def compute_rank5(self, Y, y):
	# 	# First loop over classes in order to select the closest for each class.
	# 	classes = np.unique(sorted(y))
		
	# 	sentinel = 0
	# 	for cla1 in classes:
	# 		idx1 = y==cla1
	# 		if (list(idx1).count(True)) <= 1:
	# 			continue
	# 		Y1 = Y[idx1==True, :]

	# 		for cla2 in classes:
	# 			# Select the closest that is higher than zero:
	# 			idx2 = y==cla2
	# 			if (list(idx2).count(True)) <= 1:
	# 				continue
	# 			Y2 = Y1[:, idx1==True]
	# 			Y2[Y2==0] = math.inf
	# 			min_val = np.min(np.array(Y2))
	# 			# ...