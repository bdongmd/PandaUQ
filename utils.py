import numpy as np
from typing import Any, Tuple
import lib_plotting

lbound = 0.158655524
ubound = 0.841344746

def evaluation(model : str, input_data : np.array, n_predictions : int, cut : float, title: str, sanityCheck: Any = False) -> Tuple[float, float, float, float, list]:
	""" DUQ main function
	Args:
	    model (string): NN model
	    input_data (np.array/list): object input
	    cut (float): selected model cut
	    saveOutScore (boolen): to save the whole output score or not for each object
	"""

	test_data = np.vstack([input_data]*n_predictions)
	tmp_outScore = np.array(model.predict(test_data, verbose=0))
	badValueLeft = tmp_outScore>0.04365354
	badValueRight = tmp_outScore<0.04365356
	badValue = np.logical_and(badValueLeft, badValueRight)
	outScore = tmp_outScore[~badValue]
	CI = np.quantile(outScore, [lbound, ubound], axis=0)
	outMedian = np.median(outScore)
	objAcc = np.array(outScore)[np.array(outScore)>cut].size / n_predictions

	if cut < outMedian:
		significance = (outMedian - cut) / np.sqrt((outMedian - CI[0])**2)
		usedCI = CI[0]
	else:
		significance = (outMedian - cut) / np.sqrt((outMedian - CI[1])**2)
		usedCI = CI[1]

	del test_data
	if sanityCheck:
		lib_plotting.SanityCheck(var=outScore, CI=CI, title=title)

	return(objAcc, significance, usedCI, outMedian, outScore.tolist())