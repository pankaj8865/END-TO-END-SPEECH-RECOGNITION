DATADIR = './audio' # unzipped train and test data
OUTDIR = './MaDaR4' # just a random name

print('Importing Files...')

import os
import re
from glob import glob
import numpy as np
from scipy.io import wavfile
from functions.gen import data_generator
import tensorflow as tf
from tensorflow.contrib import layers
from functions.bl import baseline
from tensorflow.contrib import signal
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tqdm import tqdm
from functions.record import recorder

print('Declaring Globals...')

POSSIBLE_LABELS = 'yes no up down left right on off stop go zero one two three four five six seven eight nine bed bird cat dog happy house marvin sheila tree wow silence unknown'.split()	#list containing all the possible commands
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}	#dictionary used to represent each command with a unique number
name2id = {name: i for i, name in id2name.items()}	#dictionary acting as reverse of the previous one


params=dict(seed=2018, batch_size=64, keep_prob=0.5, learning_rate=1e-3, clip_gradients=15.0, use_batch_norm=True, num_classes=len(POSSIBLE_LABELS))	#important parameters of this net

hparams = tf.contrib.training.HParams(**params)	#gettting the hyper parameters using the parameters
os.makedirs(os.path.join(OUTDIR, 'eval'), exist_ok=True)	#making the directory which will save the data of the trained model
model_dir = OUTDIR	#the directory to be used for saving the model

def model_handler(features, labels, mode, params, config):
	# Im really like to use make_template instead of variable_scopes and re-usage
	extractor = tf.make_template('extractor', baseline, create_scope_now_=True)
	# wav is a waveform signal with shape (16000, )
	wav = features['wav']
	# we want to compute spectograms by means of short time fourier transform:
	specgram = signal.stft(
		wav,
		400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
		160,  # 16000 * 0.010 -- default stride
		)
	# specgram is a complex tensor, so split it into abs and phase parts:
	phase = tf.angle(specgram) / np.pi
	# log(1 + abs) is a default transformation for energy units
	amp = tf.log1p(tf.abs(specgram))

	x = tf.stack([amp, phase], axis=3) # shape is [bs, time, freq_bins, 2]
	x = tf.to_float(x)  # we want to have float32, not float64

	logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

	if mode == tf.estimator.ModeKeys.TRAIN:
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
		# some lr tuner, you could use move interesting functions
		def learning_rate_decay_fn(learning_rate, global_step):
			return tf.train.exponential_decay(learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

		train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(), learning_rate=params.learning_rate, optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True), learning_rate_decay_fn=learning_rate_decay_fn, clip_gradients=params.clip_gradients, variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))	#this will be called only in the train mode which inturn is using tensorflow's algo to minimise the loss(cost) of the network
		
		specs = dict(mode=mode, loss=loss, train_op=train_op)

	if mode == tf.estimator.ModeKeys.EVAL:
		prediction = tf.argmax(logits, axis=-1)
		acc, acc_op = tf.metrics.mean_per_class_accuracy(labels, prediction, params.num_classes)
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
		specs = dict(mode=mode, loss=loss, eval_metric_ops=dict(acc=(acc, acc_op)))

	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
			'sample': features['sample'], # it's a hack for simplicity
		}
		specs = dict(mode=mode, predictions=predictions)
	return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
	return tf.estimator.Estimator(model_fn=model_handler, config=config, params=hparams)	#the function which would be called by '_create_my_experiment()'


print('Rebuilding Network from previous CheckPoint...')

run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)	#Loading the model from the directory ./MaDaR4

def _create_my_experiment(run_config, hparams):
	exp = tf.contrib.learn.Experiment(estimator=create_model(config=run_config, hparams=hparams), train_input_fn=train_input_fn, eval_input_fn=val_input_fn, train_steps=10000, eval_steps=200, train_steps_per_iteration=1000)	#the main classifier which will be trained and be used for predictions in the end
	return exp
	
def test_data_generator(data):
	def generator():
		for path in data:
			_, wav = wavfile.read(path)
			wav = wav.astype(np.float32) / np.iinfo(np.int16).max
			fname = os.path.basename(path)
			yield dict(sample=np.string_(fname), wav=wav)
	return generator

def predictor():
	# now we want to predict!
	submission = dict()
	paths = ['./audio/recording.wav']
	
	test_input_fn = generator_input_fn(x=test_data_generator(paths), batch_size=hparams.batch_size, shuffle=False, num_epochs=1, queue_capacity= 10 * hparams.batch_size, num_threads=1)	#the predict function being called

	model = create_model(config=run_config, hparams=hparams)
	it = model.predict(input_fn=test_input_fn)

	# last batch will contain padding, so remove duplicates
	for t in tqdm(it):
		fname, label = t['sample'].decode(), id2name[t['label']]
		submission[fname] = label
	os.system('clear')
	for fname, label in submission.items():
		if label == 'stop':
			print('Exitting...')
			os.system('rm -rf ./audio/*')
			return True
		print('You said : {}\n'.format(label))
		return False

print('Ready')

while True:
	_ = input('Press Enter to continue...')
	os.system('rm -rf ./audio/*')
	recorder()
	if predictor():
		break
print('Thank You')
