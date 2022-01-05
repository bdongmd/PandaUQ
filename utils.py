import numpy as np
from typing import Any, Tuple

from numpy import lib
import lib_plotting

lbound = 0.158655524
ubound = 0.841344746

def evaluation(model : str, input_data : np.array, n_predictions : int, cut : float, title: str, label: int, sanityCheck: Any = False) -> Tuple[float, float, float, float, float, list]:
	""" DUQ main function
	Args:
	    model (string): NN model
	    input_data (np.array/list): object input
	    cut (float): selected model cut
	    saveOutScore (boolen): to save the whole output score or not for each object
	"""

	test_data = np.vstack([input_data]*n_predictions)
	outScore = np.array(model.predict(test_data, verbose=0))
	#badValueLeft = tmp_outScore>0.04365354
	#badValueRight = tmp_outScore<0.04365356
	#badValue = np.logical_and(badValueLeft, badValueRight)
	#outScore = tmp_outScore[~badValue]
	CI = np.quantile(outScore, [lbound, ubound], axis=0)
	outMedian = np.median(outScore)
	objAcc = np.array(outScore)[np.array(outScore)>cut].size / n_predictions

	if cut < outMedian:
		usedCI = CI[0]
		uncer = np.sqrt((outMedian - CI[0])**2) 
		significance = (outMedian - cut) / uncer
	else:
		usedCI = CI[1]
		uncer = np.sqrt((outMedian - CI[1])**2)
		significance = (outMedian - cut) / uncer

	del test_data
	if uncer>0.7:
		lib_plotting.plot_large_uncer(outScore, outMedian, uncer[0], outFile='output/large_uncer_{}.png'.format(title))

	if sanityCheck:
		lib_plotting.SanityCheck(var=outScore, CI=CI, title=title, label=label)

	return(objAcc, significance, usedCI, uncer, outMedian, outScore.tolist())