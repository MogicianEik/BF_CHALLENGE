export CUDA_VISIBLE_DEVICES=0
python -W ignore src/main.py \
--n_class 9 \
--train_file "data/Training_data.tbx" \
--model_path "outputs/saved_models/" \
--log_path "outputs/runs/" \
--task_name "SV_predict" \
--batch_size 32 \
--lr 5e-4 \
--num_epochs 50
