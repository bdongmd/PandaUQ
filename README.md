# PandaUQ
Uncertainty quantification apply in Panda analysis

## Content
* [Package dependency](#package-dependency)
* [Instruction](#instruction)

## Package dependency
- tensorflow
- kera
- uproot3 (uproot3 not uproot(4) !!!!!)
- pandas
- numpy
- matplotlib
- tqdm

## Instruction
- First convert the ntuples to a root sample includes all required input varaibles for NN training
```
cd SamplePrep
# example of running S2 samples
python3 ToSmallerRoot.py -i ../input/sig_total.root -s S2Only -o ../input/bdong_sig.root 
```

- Convert data format from root to hdf5
For S2only training, can directly use the ```conver.py``` to get the hdf5. So far, listed 23 variables for training. Command:
```
python3 convert.py -b ../input/bkg_total.root -s ../input/original/rn_wall_30_200_reclu_pe.root --s2 ../input/original/tot_ds_sample_reclu_pe.root --saveScale -c input_Variables.json -o ../input/total_shuffled_3class.h5
```
In order to save scaling info for training variables, use flag ```--saveScale```. Otherwise, the file containing scaling info will be loaded. \\
(In the above case, 3 classes instead of 2 are used. Label 0: background, 1: signal from wall, 2: signal from ds)

- Training:
```
python3 train.py -i input/total_shuffled.h5 -o output/var23_epoch80_Dorp0p2
```

- Evaluation (with DUQ method include):
```
python3 calibrate.py -i input/total_shuffled.h5 -o output/var23_epoch80_Dorp0p2 -m output/var23_epoch80_Dorp0p2/training_e40.h5
```
Manipulate the flags in the ```calibrate.py``` to select the type of plots you want to have.
