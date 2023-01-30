export CUDA_VISIBLE_DEVICES=0
python -W ignore src/main.py \
--n_class 9 \
--pred_file "data/Prediction_data_updated.tbx" \
--model_path "outputs/saved_models/" \
--log_path "outputs/runs/" \
--task_name "SV_predict" \
--batch_size 1 \
--prediction \
--pred_rotation 4
