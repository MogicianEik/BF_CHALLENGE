export CUDA_VISIBLE_DEVICES=0
python -W ignore src/main.py \
--n_class 9 \
--pred_file "data/Prediction_data_batch1.tbx" \
--model_path "outputs/saved_models/old_set_best0.67_fold3/" \
--log_path "outputs/runs/" \
--task_name "SV_predict_old" \
--batch_size 1 \
--prediction \
--pred_rotation 3
