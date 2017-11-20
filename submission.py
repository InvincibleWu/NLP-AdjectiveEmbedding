#!/usr/bin/env python


## Submission.py for COMP6714-Project2
###################################################################################################################





import os
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import collections
import random
import math
import gensim
import re


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):


	# Specification of Training data:
	batch_size = 120      # Size of mini-batch for skip-gram model.
	skip_window = 2       # How many words to consider left and right of the target word.
	num_samples = 2         # How many times to reuse an input to generate a label.
	vocabulary_size = 4000
	l_rate = 0.002
	num_sampled = 64      # Sample size for negative examples.

	logs_path = './log/'

	# Specification of test Sample:
	sample_size = 20       # Random sample of words to evaluate similarity.
	sample_window = 100    # Only pick samples in the head of the distribution.
	sample_examples = np.random.choice(sample_window, sample_size, replace=False) # Randomly pick a sample of size 16




	global data_index

	data_index = 0
	embedding_size = embedding_dim  # Dimension of the embedding vector.
	num_iterations = num_steps



	df = open(data_file, 'r')
	dl = []
	for line in df.readlines():
		dl.append(line.strip())


	data, count, dictionary, reverse_dictionary = build_dataset(dl, vocabulary_size)

	graph = tf.Graph()

	with graph.as_default():
	    
	    with tf.device('/cpu:0'):
	        # Placeholders to read input data.
	        with tf.name_scope('Inputs'):
	            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	            
	        # Look up embeddings for inputs.
	        with tf.name_scope('Embeddings'):            
	            sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
	            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
	            
	            # Construct the variables for the NCE loss
	            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
	                                                      stddev=1.0 / math.sqrt(embedding_size)))
	            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
	        
	        # Compute the average NCE loss for the batch.
	        # tf.nce_loss automatically draws a new sample of the negative labels each
	        # time we evaluate the loss.
	        with tf.name_scope('Loss'):
	            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, 
	                                             labels=train_labels, inputs=embed, 
	                                             num_sampled=num_sampled, num_classes=vocabulary_size))
	        
	        # Construct the Gradient Descent optimizer using a learning rate of 0.01.
	        with tf.name_scope('Gradient_Descent'):
	            optimizer = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(loss)

	        # Normalize the embeddings to avoid overfitting.
	        with tf.name_scope('Normalization'):
	            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	            normalized_embeddings = embeddings / norm
	            
	        sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
	        similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)
	        
	        # Add variable initializer.
	        init = tf.global_variables_initializer()
	        
	        
	        # Create a summary to monitor cost tensor
	        tf.summary.scalar("cost", loss)
	        # Merge all summary variables.
	        merged_summary_op = tf.summary.merge_all()


	with tf.Session(graph=graph) as session:
	    # We must initialize all variables before we use them.
	    session.run(init)
	    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	    
	    # print('Initializing the model')
	    
	    average_loss = 0
	    for step in range(num_steps):
	        batch_inputs, batch_labels = generate_batch(count, data, reverse_dictionary, batch_size, num_samples, skip_window)
	        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
	        
	        # We perform one update step by evaluating the optimizer op using session.run()
	        _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
	        
	        summary_writer.add_summary(summary, step )
	        average_loss += loss_val

	        # if step % 5000 == 0:
	        #     if step > 0:
	        #         average_loss /= 5000
	            
	        #         # The average loss is an estimate of the loss over the last 5000 batches.
	        #         print('Average loss at step ', step, ': ', average_loss)
	        #         average_loss = 0

	        # Evaluate similarity after every 10000 iterations.
	        # if step % 10000 == 0:
	        #     sim = similarity.eval() #
	        #     for i in range(sample_size):
	        #         sample_word = reverse_dictionary[sample_examples[i]]
	        #         top_k = 10  # Look for top-10 neighbours for words in sample set.
	        #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
	        #         log_str = 'Nearest to %s:' % sample_word
	        #         for k in range(top_k):
	        #             close_word = reverse_dictionary[nearest[k]]
	        #             log_str = '%s %s,' % (log_str, close_word)
	        #         print(log_str)
	        #     print()
	    final_embeddings = normalized_embeddings.eval()
	
	final_embeddings_file = open(embeddings_file_name, 'w')
	final_embeddings_file.write(str(len(final_embeddings)) + ' ' + str(embedding_size) + '\n')
	
	for row in range(vocabulary_size):
		vector = reverse_dictionary[row]
		for v in range(embedding_size):
			num = str(round(final_embeddings[row][v], 6))
			vector += ' ' + num
		vector += '\n'
		final_embeddings_file.write(vector)
	final_embeddings_file.close()




def process_data(input_data):
	z = zipfile.ZipFile(input_data, "r")
	combined_txt = ''
	for file in z.namelist():
		if file[-3:] == "txt":
			combined_txt += str(z.read(file))
	nlp = spacy.load("en")
	document = nlp(combined_txt)


	data_list = list()

	skip_list = ['SPACE', 'PUNCT', 'PART']
	
	change_list = {'NUM':'-NUM-', 'SYM':'-SYM-', 'PROPN':'-PROPN-', 'ADP':'-ADP-', 'DET':'-DET-', 'CCONJ':'-CCONJ-'}

	ent_type_list = {'DATE':'-DATE-', 'PERSON':'-PERSON-', 'ORG':'-ORG-', 'MONEY':'-MONEY-', 'QUANTITY':'-QUANTITY-', 'GPE':'-GPE-', 'ORDINAL':'-ORDINAL-', 'PERCENT':'-PERCENT-'}


	file_name = 'processed_data_file.txt'
	file = open(file_name, 'w')
	for sent in document.sents:
	    for word in sent:
	        if word.pos_ in skip_list:
	            continue
	        elif word.pos_ == 'ADJ':
	            data_list.append(word.lower_ + '*' + word.pos_)
	        elif word.pos_ == 'VERB':
	            data_list.append(word.lemma_ + '*' + word.pos_)
	        elif word.ent_type_ in ent_type_list:
	            data_list.append(ent_type_list[word.ent_type_] + '*' + word.pos_)
	        elif word.pos_ in change_list:
	            data_list.append(change_list[word.pos_] + '*' + word.pos_)
	        elif word.pos_ == 'NOUN':
	        	data_list.append(word.lemma_ + '*' + word.pos_)
	        else:
	            data_list.append(word.lower_ + '*' + word.pos_)
	    data_list.append('-EOS-*END')
	for w in data_list:
		t = w + '\n'
		file.write(t)
	file.close()

	return file_name




def Compute_topk(model_file, input_adjective, top_k):
	model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
	tn = len(open(model_file, 'r').readlines()) - 1
	adj = input_adjective + '*ADJ'

	m = open(model_file, 'r')
	adj_list = []
	for line in m:
		word = line.split(' ')[0]
		if word[-4:] == '*ADJ':
			adj_list.append(word[:-4])

	try:
		topk = model.wv.most_similar([adj], topn = tn)
	except:
		output = random.sample(adj_list, 100)
		return output
	else:
		word_num = 0
		word_index = 0
		output = []
		while word_num < top_k:
			if topk[word_index][0][-4:] == '*ADJ':
				word_num += 1
				output.append(topk[word_index][0][:-4])
			word_index += 1
			if word_index == len(topk):
				break
		return output



def build_dataset(words, n_words):
	count = [['UNK*UNK', -1]]
	count.extend(collections.Counter(words).most_common(n_words - 1))
	total_num = len(words)
	dictionary = dict()
	for word, c in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		index = dictionary.get(word, 0)
		if index == 0:  # i.e., one of the 'UNK' words
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reversed_dictionary



def weighted_choice(weights, buffer, skip_window, num_samples, reverse_dictionary):
	output_list = []
	center_word = reverse_dictionary[buffer[skip_window]]
	first_verb_before_index = -1
	first_noun_after_index = -1
	if center_word[-4:] == '*ADJ':
		for i in range(0, skip_window):
			if reverse_dictionary[buffer[skip_window - 1 - i]][-5:] == '*VERB':
				output_list.append(i)
				first_verb_before_index = i
				break
		for j in range(skip_window + 1, skip_window*2 + 1):
			if reverse_dictionary[buffer[i]][-5:] == '*NOUN':
				output_list.append(j)
				first_noun_after_index = j
				break
	skip_index = [first_noun_after_index, first_verb_before_index, skip_window]
	sw = sum(weights)
	new_weights = []
	for i in range(len(weights)):
		new_weights.append(sw - weights[i])
	totals = []
	running_total = 0
	for w in new_weights:
		running_total += w
		totals.append(running_total)

	while len(output_list) < num_samples:
		rnd = random.random() * running_total
		for i, total in enumerate(totals):
			if rnd < total:
				if not i in skip_index:
					skip_index.append(i)
					output_list.append(i)
	return output_list





def generate_batch(count, data, reverse_dictionary, batch_size, num_samples, skip_window):
    global data_index   
    
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window
    
    data_index += span
    for i in range(batch_size // num_samples):
        weights = []
        for index in range(span):
        	if index == skip_window:
        		weights.append(0)
        	else:
        		weights.append(count[buffer[index]][1])


        words_to_use = weighted_choice(weights, buffer, skip_window, num_samples, reverse_dictionary)
        for j in range(num_samples): # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop(0)
            labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
        
        # slide the window to the next position    
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else: 
            buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1
        
    # end-of-for
    data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
    return batch, labels





