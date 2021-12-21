# PandaUQ
Uncertainty quantification apply in Panda analysis

## Content
* [Package dependency](#package-dependency)
* [Instruction](#instruction)

## Package dependency
- tensorflow
- kera
- tqdm
- pandas
- uproot3 (uproot3 not uproot(4) !!!!!)
- numpy
- matplotlib

## Instruction
- First convert the ntuples to a root sample includes all required input varaibles for NN training
```
cd SamplePrep
# example of running S2 samples
python3 ToSmallerRoot.py -i ../input/sig_total.root -s S2Only -o ../input/bdong_sig.root 
```

- Convert data format from root to hdf5
```
python3 convert.py -s ../input/bdong_sig.root -b ../input/bdong_bkg.root -o ../input/bdong_total_shuffled.h5 -c S2Only_input_var.json
```



