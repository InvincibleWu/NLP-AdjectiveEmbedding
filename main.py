#!/usr/bin/env python

import submission as submission
import gensim




input_dir = './BBC_Data.zip'
data_file = submission.process_data(input_dir)

## Output file name to store the final trained embeddings.
embedding_file_name = 'Test5.txt'
# embedding_file_name = 'Test1_128,2,3,12500,0.001,64_.txt'
results_file_name = 'Test5_results.txt'



## Fixed parameters
num_steps = 100001
embedding_dim = 200



## Train Embeddings, and write embeddings in "adjective_embeddings.txt"
submission.adjective_embeddings(data_file, embedding_file_name, num_steps, embedding_dim)



def hits(model_file):
	gt = open('ground_truth.txt', 'r')
	gt_dic = {}
	test_list = []

	parameters = ['120', '2', '2', '4000', '0.002', '64']
	output_file = open(results_file_name, 'w')

	p1 = 'batch_size:'.ljust(20) + parameters[0] + '\n'
	p2 = 'skip_window:'.ljust(20) + parameters[1] + '\n'
	p3 = 'num_samples:'.ljust(20) + parameters[2] + '\n'
	p4 = 'vocabulary_size:'.ljust(20) + parameters[3] + '\n'
	p5 = 'learning_rate:'.ljust(20) + parameters[4] + '\n'
	p6 = 'num_sampled:'.ljust(20) + parameters[5] + '\n'
	output_file.write(p1)
	output_file.write(p2)
	output_file.write(p3)
	output_file.write(p4)
	output_file.write(p5)
	output_file.write(p6)
	output_file.write('\n\n')
	for line in gt.readlines():
		temp = line.split()
		key = temp[0]
		test_list.append(key)
		value = temp[1:]
		gt_dic[key] = value
	total_tests = len(test_list)
	total_hits = 0
	for adj in test_list:
		ground_truth = gt_dic[adj]
		# print(ground_truth)
		test = submission.Compute_topk(model_file, adj, 100)
		# print(test)
		adj_word = adj.ljust(15) + '---->  '
		print(adj.ljust(15), end = '---->  ')
		for word in ground_truth:
			if word in test:
				adj_word += ' ' + word
				print(word, end = ' ')
				total_hits += 1
		adj_word += '\n'
		output_file.write(adj_word)
		print()


	average_hits = '\n\nAverage hits: ' + str(total_hits/total_tests)
	output_file.write(average_hits)
	output_file.close()
	print(average_hits)

hits(embedding_file_name)

















