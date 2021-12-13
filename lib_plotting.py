import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import sys

def plotAccLoss(trainInput, testInput, putVar, output_dir='plots'):
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
	plt.savefig('{}/_compare.pdf'.format(output_dir, putVar))

def variable_plotting(signal, bkg, outputFile="output/inputVar.pdf"):

	nbins = 50
	with open("input_var.json") as vardict:
		variablelist = json.load(vardict)[:]


	varcounter = -1

	fig, ax = plt.subplots(3,4, figsize=(25,35))
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
				axobj.set_yscale('log',nonposty='clip')
				axobj.set_title(variablelist[varcounter])

			else:
				axobj.axis('off')
	del signal, bkg

	plt.tight_layout()
	plt.savefig(outputFile, transparent=True)
