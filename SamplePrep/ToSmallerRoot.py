from numpy import dtype
import uproot3 as up
import argparse

parser = argparse.ArgumentParser(description='Add input and output files.')
parser.add_argument('-i', '--inputfile', type=str,
	default='/Users/binbindong/Desktop/work/PandaX/PandaUQ/input/simu_2hit_bdt.root')
parser.add_argument('-s', '--isSignal', type=bool)
parser.add_argument('-o', '--outputfile', type=str)
args = parser.parse_args()

#### well, they have different tree name for signal and background sample : )
if args.isSignal:
	treeName = 'b8_tree'
else:
	treeName = 'acc_tree'

input = up.open(args.inputfile)[treeName]
#input.show()

##### newly defined varialbes....
##### they are not directly defined in the input file
wcdf = input.array('wS1CDF_max')
qElse = input.array('qElseBeforeS1max')
qNear = input.array('qS1_max')/input.array('qNearS1max')
TBA = (input.array('qS2T_max')-input.array('qS2B_max'))/input.array('qS2_max')
PMT_q = input.array('qS2maxChannelCharge_max')/input.array('qS2_max')
hit_rms = input.array('qS2hitStdev_max')
top_rms = input.array('rmsMaxQPMTPosS2T_max')
wcdf = 4*0.001*input.array('wS2CDF_max')
w2575 = 4*0.001*(input.array('wS2CDF75_max') - input.array('wS2CDF25_max'))
w50 = 4*0.001*input.array('wS2CDF50_max')

output = up.recreate(args.outputfile)
output["miniTree"] = up.newtree({"wcdf": float, "qElse": float, "qNear" : float, "TBA" : float, "PMT_q" :  float, "hit_rms" :  float, "top_rms" : float, "wcdf" : float, "w2575" : float, "w50" : float})
output["miniTree"].extend({
	"wcdf" : wcdf,
	"qElse" : qElse,
	"qNear" :  qNear,
	"TBA" : TBA,
	"PMT_q" : PMT_q,
	"hit_rms" : hit_rms,
	"top_rms" : top_rms,
	"wcdf" : wcdf,
	"w2575" : w2575,
	"w50" : w50
})

output.close()