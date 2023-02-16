export CUDA_VISIBLE_DEVICES=0
python -W ignore src/main.py \
--n_class 3 \
--eval_file "data/Validation_data_updated.tbx" \
--model_path "outputs/saved_models/" \
--log_path "outputs/runs/" \
--task_name "SV_SD_predict_updated" \
--batch_size 1 \
--evaluation
