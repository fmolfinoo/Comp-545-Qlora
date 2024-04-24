All the changes are limited to the modeling/llama/ folder where we modified eval and train to be able to train and evaluate Qlora models. We created 2 files one called base_eval to evaluate base models without any modifications and input_records_trunc.json(is not included in the repo but you can create your own by commenting the loading code and uncommenting the usual way to build input records) that saves a dictionary created during training that takes around 30 minutes to create to be able to quickly train and not have to create it from scratch every time.

We also modified the modeling/llama/conf/config.yaml to be able to manually set paths to models and affects other variables in the program like which set to use during evaluation.Some of the paths in the original config file where not compatible with windows paths, so we have to adapt them.


Note we ran all the code using Pycharm IDE not the command line, so we changed how we imported some files to make it compatible with this.


The input record file with the other adapters is in the https://huggingface.co/molfi/QloraCodeLLamaAdapters 