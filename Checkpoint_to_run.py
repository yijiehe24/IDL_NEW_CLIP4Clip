#Use checkpoint to run eval
#Paste this into terminal


python -m torch.distributed.launch --nproc_per_node=4 \
/content/CLIP4Clip/main_task_retrieval.py \
--do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv /content/CLIP4Clip/msrvtt_data/MSRVTT_train.9k.csv \
--val_csv /content/CLIP4Clip/msrvtt_data/MSRVTT_JSFUSION_test.csv \
--data_path /content/CLIP4Clip/msrvtt_data/MSRVTT_data.json \
--features_path /content/CLIP4Clip/MSRVTT/videos/all \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-3 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 12  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model /content/CLIP4Clip/pytorch_model.bin \
--resume_model /content/CLIP4Clip/pytorch_opt.bin  \
--pretrained_clip_name ViT-B/32

