# r2c
Recognition to Cognition Networks


## Environment Creation

* conda create -n r2c10 python=3.6

* interactive -N 1 -n 20 -p cidsegpu1 -q cidsegpu1_other_res --gres=gpu:4 -t 0-7:00:00

* source activate r2c10


* module load cuda/9.0.176  
* export LD_LIBRARY_PATH=/packages/7x/cuda/9.0.176:$LD_LIBRARY_PATH

* conda install numpy setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg

* pip install -r allennlp-requirements.txt

* conda uninstall PyYAML
* pip install -r allennlp-requirements.txt

* pip install --no-deps allennlp==0.8.0
* python -m spacy download en_core_web_sm

* conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

## Checks 
* >>> torch.__version__  
'1.1.0'  
* >>> torchvision.__version__   
'0.3.0'  
* Currently Loaded Modulefiles:  
  1) cudnn/7.0         2) anaconda3/5.3.0   3) cuda/9.0.176  
## Adding other file (Follow rowan's data folder README)

* Image and annots files addition
1. Get the *.h5 files into the data folder 
2. Get the vcr1images into the data folder
3. Get the 3 jsonl files into the data folder


## Execution

* module load cuda/9.0.176  
* export LD_LIBRARY_PATH=/packages/7x/cuda/9.0.176:$LD_LIBRARY_PATH
* export PYTHONPATH=~/r2c-latest/r2c
* cd r2c/models/ 
* python train.py -params multiatt/default.json -folder saves/flagship_answer 


## Current Issue/Status :
* Running for 1 hr and then giving memory error  
Warning ⚠️ (Need to solve this issue)

/home/kkpal/.conda/envs/r2cw/lib/python3.6/site-packages/torch/nn/_reduction.py:46: UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
  warnings.warn(warning.format(ret))
  
/home/kkpal/.conda/envs/r2cw/lib/python3.6/site-packages/torch/nn/modules/rnn.py:525: RuntimeWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
  self.num_layers, self.dropout, self.training, self.bidirectional)
