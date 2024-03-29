export CUDA_VISIBLE_DEVICES=0
python -W ignore src/main.py \
--n_class 9 \
--train_file "data/Training_data_updated.tbx" \
--model_path "outputs/saved_models/" \
--log_path "outputs/runs/" \
--task_name "SV_predict_updated" \
--batch_size 44 \
--lr 1e-4 \
--num_epochs 20
