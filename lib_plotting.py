import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import json
import pandas as pd
import sys
from tqdm import tqdm 

def plotOutputScore(score, labels, output_dir='output'):
	''' Plot output scores of the NN modeli, and purity stuff '''
	outputScore = np.array(score)
	labels = np.array(labels)
	scanvalue = np.linspace(0., 1.0, num=100)
	cut = []
	sig_eff = []
	purity = []
	signal = labels[labels==1]
	signal_score = score[labels==1]
	significance = []
	N_signal = []

	for i in tqdm(range(len(scanvalue)), desc="========== Caluting ROC curve..."):
		signal_tagged = signal[signal_score>scanvalue[i]]
		all_tagged = labels[score>scanvalue[i]]
		if len(all_tagged)==0:
			break
		cut.append(scanvalue[i])
		N_signal.append(len(signal_tagged))
		significance.append(len(signal_tagged)/np.sqrt(len(all_tagged)))
		sig_eff.append(len(signal_tagged)/len(signal))
		purity.append(len(signal_tagged)/len(all_tagged))
	
	print("========== Plotting testing performance...")
	pdf = matplotlib.backends.backend_pdf.PdfPages("{}/testing_perm.pdf".format(output_dir))
	## first is the simple output distribution for signal and background
	fig,ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	nbins=200
	ax.hist(outputScore[labels==1], bins=nbins, range=[0,1], density=True, label = 'signal', histtype='step')
	ax.hist(outputScore[labels==0], bins=nbins, range=[0,1], density=True, label = 'background', histtype='step')
	ax.set_ylabel('density')
	ax.set_xlabel('sigmoid output')
	#ax.set_yscale('log')
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	## plot significance as a funciton of cut
	fig,ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	color = 'tab:red'
	ax.plot(cut, significance, 'o', color=color)
	ax.set_ylabel('significance', color=color)
	ax.set_xlabel('cut')
	ax.tick_params(axis='y', labelcolor = color)

	ax2 = ax.twinx()
	color = 'tab:green'
	ax2.set_ylabel('N. of tagged signal events', color=color)
	ax2.plot(cut, N_signal, color = color)
	ax2.tick_params(axis='y', labelcolor = color)

	pdf.savefig()
	fig.clear()
	plt.close(fig)

	## plot eff versus purity
	fig,ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(cut, sig_eff, 'o', label="efficiency", color = 'blue')
	ax.plot(cut, purity, 'o', label="purity", color='orange')
	ax.set_xlabel('cut')
	ax.set_ylim(0, 1.2)
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig,ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(sig_eff, purity, 'o', color = 'blue')
	ax.set_xlabel('efficiency')
	ax.set_ylabel('purity')
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	pdf.close()

def plotAccLoss(trainInput, testInput, putVar, output_dir='plots'):
	'''comparison between train and testing loss and accuracy'''

	epochs = np.arange(1, len(trainInput) + 1)

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.plot(epochs, trainInput, 'o')
	plt.plot(epochs, testInput, 'o')

	if putVar == 'Loss':
		plt.legend(['training loss', 'testing loss'], loc='upper right')
		plt.ylabel('loss')
		plt.yscale('log')

	elif putVar == 'Acc':
		plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
		plt.ylabel('accuracy (%)')

	else:
		print('Hello, you are putting some varible I do not know. Please check...')
		sys.exit()
	
	plt.xlabel('epoch')
	plt.savefig('{}/{}_compare.pdf'.format(output_dir, putVar))

def variable_plotting(signal, bkg, variables="S2Only_input_var.json", outputFile="output/inputVar.pdf"):
	'''used to plot input distributions between signal and background'''

	nbins = 50
	with open('{}'.format(variables)) as vardict:
		variablelist = json.load(vardict)[:]


	varcounter = -1

	fig, ax = plt.subplots(3,4, figsize=(16, 12))
	for i, axobjlist in enumerate(ax):
		for j, axobj in enumerate(axobjlist):
			varcounter += 1
			if varcounter < len(variablelist):
				var = variablelist[varcounter]
				
				b = bkg[var]
				s = signal[var]
				b.replace([np.inf, -np.inf], np.nan, inplace=True)
				s.replace([np.inf, -np.inf], np.nan, inplace=True)

				b = b.dropna()
				s = s.dropna()

				minval = min([np.amin(s), np.amin(b)])
				maxval = max([np.amax(s), np.amax(b)])*1.4
				binning = np.linspace(minval, maxval, nbins)

				axobj.hist(b, binning, histtype=u'step', color='orange', label='background', density=1)
				axobj.hist(s, binning, histtype=u'step', color='g', label='signal', density=1)

				axobj.legend()
				axobj.set_yscale('log')
				axobj.set_title(variablelist[varcounter])

			else:
				axobj.axis('off')
	del signal, bkg

	plt.tight_layout()
	plt.savefig(outputFile, transparent=True)
