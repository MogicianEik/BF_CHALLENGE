# BF CHALLENGE PROJECT USAGE

1.  update the environment via conda/anaconda use: `conda env create -f environment.yml`
2.  check data in the data file:
  1. provide a labeled table SNP file for taining, here `Training_data.tbx` is provided
  2. provide a labeled table SNP file for separate validation, here `Validation_data.tbx` is provided
  3. provide an unlabled table SNP file for SV prediction
  4. provide a `labels.yml` file after seeing the label-code mapping during the training, used for validation and prediction
3.  activate the conda environment at the working directory
4.  To run the training process, `sh train.sh`, feel free the change the parameters. to check parameters, `python src/main.py -h`
5.  To run the validation process, `sh test.sh` and check the arguments of that script.
6.  To run the prediction, `sh predict.sh`
