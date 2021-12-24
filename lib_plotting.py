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
	N_bkg = []
	#sWeight = 1.95/(73010*0.7)
	#bWeight = 456.9/(946892*0.7)
	#### For 1 hit:
	sWeight = 2.7889/(52004)
	bWeight = 3554.7073/(552226-52004)

	for i in tqdm(range(len(scanvalue)), desc="========== Caluting ROC curve..."):
		signal_tagged = signal[signal_score>scanvalue[i]]
		all_tagged = labels[score>scanvalue[i]]
		if len(all_tagged)==0:
			break
		tmp_sig = len(signal_tagged)*sWeight/np.sqrt((len(all_tagged)-len(signal_tagged))*bWeight+len(signal_tagged)*sWeight)
		if tmp_sig>1:
			#print('for cut = {}'.format(scanvalue[i]))
			print("sig : Nsig : Nbkg = {} : {} : {}".format(tmp_sig, len(signal_tagged)*sWeight, (len(all_tagged)-len(signal_tagged))*bWeight))
		cut.append(scanvalue[i])
		N_signal.append(len(signal_tagged)*sWeight)
		N_bkg.append((len(all_tagged) - len(signal_tagged))*bWeight)
		significance.append(tmp_sig)
		sig_eff.append(len(signal_tagged)/len(signal))
		purity.append(len(signal_tagged)*sWeight/((len(all_tagged)-len(signal_tagged))*bWeight+len(signal_tagged)*sWeight))
	
	print("========== Plotting testing performance...")
	pdf = matplotlib.backends.backend_pdf.PdfPages("{}/testing_perm.pdf".format(output_dir))
	## first is the simple output distribution for signal and background
	fig,ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	nbins=200
	ax.hist(outputScore[labels==1], bins=nbins, range=[0,1], density=True, label = 'signal', histtype='step')
	ax.hist(outputScore[labels==0], bins=nbins, range=[0,1], density=True, label = 'background', histtype='step')
	ax.set_ylabel('density')
	ax.set_xlabel('sigmoid output')
	ax.set_yscale('log')
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
	ax2.set_ylabel('N. of tagged events', color=color)
	ax2.set_ylim(0,5)
	ax2.plot(cut, N_signal, color = color, label='signal')
	ax2.plot(cut, N_bkg, color='blue', label='background')
	ax2.tick_params(axis='y', labelcolor = 'black')
	ax2.legend()

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

def variable_plotting(signal, bkg, sig2=None, noname=False, variables="S2Only_input_var.json", outputFile="output/inputVar.pdf"):
	'''used to plot input distributions between signal and background'''

	nbins = 50
	with open('{}'.format(variables)) as vardict:
		variablelist = json.load(vardict)[:]


	varcounter = -1

	fig, ax = plt.subplots(5,5, figsize=(16, 12))
	for i, axobjlist in enumerate(ax):
		for j, axobj in enumerate(axobjlist):
			varcounter += 1
			if varcounter < len(variablelist):
				
				if noname:
					b = bkg[varcounter]
					s = signal[varcounter]
					if sig2 is not None:
						s2 = sig2[varcounter]
				else:
					var = variablelist[varcounter]
					b = bkg[var]
					s = signal[var]
					if sig2 is not None:
						s2 = sig2[var]
				b.replace([np.inf, -np.inf], np.nan, inplace=True)
				b = b.dropna()
				s.replace([np.inf, -np.inf], np.nan, inplace=True)
				s = s.dropna()
				if sig2 is not None:
					s2.replace([np.inf, -np.inf], np.nan, inplace=True)
					s2 = s2.dropna()

				minval = min([np.amin(s), np.amin(b)])
				maxval = min([np.amax(s), np.amax(b)])*1.4
				binning = np.linspace(minval, maxval, nbins)

				axobj.hist(b, binning, histtype=u'step', color='orange', label='background', density=1)
				axobj.hist(s, binning, histtype=u'step', color='green', label='signal', density=1)
				if sig2 is not None:
					axobj.hist(s2, binning, histtype=u'step', color='blue',label='new sig', density=1)

				axobj.legend()
				axobj.set_yscale('log')
				axobj.set_title(variablelist[varcounter])

			else:
				axobj.axis('off')
	del signal, bkg

	plt.tight_layout()
	plt.savefig(outputFile, transparent=True)


def compare2Sig(sig1_score, sig2_score, outputDir):
	'''Compare the efficiency and other performance between two signals.
	sig1_score: sigmoid output score for signal 1
	sig2_score: sigmoid output score for signal 2
	outputfile: name of output file'''

	scanvalue = np.linspace(0., 1.0, num=100)
	sigeff_1 = []
	sigeff_2 = []
	cut = []

	for i in tqdm(range(len(scanvalue)), desc="========== Caluting ROC curve..."):
		sigeff_1.append(len(sig1_score[sig1_score>scanvalue[i]])/len(sig1_score))
		sigeff_2.append(len(sig2_score[sig2_score>scanvalue[i]])/len(sig2_score))
		cut.append(scanvalue[i])
	
	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.plot(cut, sigeff_1, 'o')
	plt.plot(cut, sigeff_2, 'o')
	plt.legend(['trained signal', 'new signals'], loc='upper right')
	plt.ylabel('efficiency')
	plt.xlabel('output score')
	plt.savefig('{}/eff.pdf'.format(outputDir))	

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	nbins=200
	plt.hist(sig1_score, bins=nbins, range=[0,1], density=True, label='trained signal', histtype='step')
	plt.hist(sig2_score, bins=nbins, range=[0,1], density=True, label='new signal', histtype='step')
	plt.legend(['trained signal', 'new signals'], loc='upper right')
	plt.ylabel('density')
	plt.xlabel('output score')
	plt.savefig('{}/score.pdf'.format(outputDir))