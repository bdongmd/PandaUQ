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

################### Add your cut here
######### define your cut
#cut = input.array('wS1CDF_max')>1
######### If no cut needed
cut = [True] * numentries

##### newly defined varialbes....
##### they are not directly defined in the input file
wcdf = input.array('wS1CDF_max')[cut]
qElse = input.array('qElseBeforeS1max')[cut]
qNear = input.array('qS1_max')/input.array('qNearS1max')[cut]
TBA = (input.array('qS2T_max')-input.array('qS2B_max'))/input.array('qS2_max')[cut]
PMT_q = input.array('qS2maxChannelCharge_max')/input.array('qS2_max')[cut]
hit_rms = input.array('qS2hitStdev_max')[cut]
top_rms = input.array('rmsMaxQPMTPosS2T_max')[cut]
wcdf_S2 = 4*0.001*input.array('wS2CDF_max')[cut]
w2575 = 4*0.001*(input.array('wS2CDF75_max') - input.array('wS2CDF25_max'))[cut]
w50 = 4*0.001*input.array('wS2CDF50_max')[cut]

output = up.recreate(args.outputfile)
output["miniTree"] = up.newtree({"wcdf": float, "qElse": float, "qNear" : float, "TBA" : float, "PMT_q" :  float, "hit_rms" :  float, "top_rms" : float, "wcdf_S2" : float, "w2575" : float, "w50" : float})
output["miniTree"].extend({
	"wcdf" : wcdf,
	"qElse" : qElse,
	"qNear" :  qNear,
	"TBA" : TBA,
	"PMT_q" : PMT_q,
	"hit_rms" : hit_rms,
	"top_rms" : top_rms,
	"wcdf_S2" : wcdf_S2,
	"w2575" : w2575,
	"w50" : w50
})

output.close()
