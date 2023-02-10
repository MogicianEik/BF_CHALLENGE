export CUDA_VISIBLE_DEVICES=0
python -W ignore src/main.py \
--n_class 9 \
--eval_file "data/Validation_data_old.tbx" \
--model_path "outputs/saved_models/" \
--log_path "outputs/runs/" \
--task_name "SV_predict_old" \
--batch_size 1 \
--evaluation
