import math
import numpy as np

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
		for ix, predictions in enumerate(Y):
			maxIx = np.argmax(predictions,axis=0) + 1
			print("maxIx -> ", maxIx)
			print("detected -> ", y[ix])
			if (maxIx == y[ix]):
				count_correct += 1
			count_all += 1

		rank1 = count_correct/count_all
		
		print("correctly indentified -> ", count_correct)
		return rank1*100

	def compute_rankN_nn(self, Y, y, rank):
		count_all = 0
		count_correct = 0
		for i in range(len(Y)):
			predictions = Y[i]
			top_n =  np.argsort(predictions)[::-1][:rank]
			top_n += 1
			if y[i] in top_n:
				count_correct += 1
			count_all += 1

		print("count all -> ", count_all)
		score = count_correct/count_all
		
		return score*100

	def compute_CMC_ranks_nn(self, Y, y, max_rank):
		acc_by_rank_list = []
		for rank in range(1, max_rank + 1):
			acc_by_rank_list.append(self.compute_rankN_nn(Y, y, rank))

		return acc_by_rank_list