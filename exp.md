

tmux new -s exp01 
tmux attach -t exp01

exp_4
source /home/estudiante1/venvs/bd_env/bin/activate
python run_model.py --gpus 0 \
            --exp_name exp_4 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 70 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 501 \
            --checkpoint_interval 50 \
exp_3
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name exp_3 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 401 \
            --checkpoint_interval 50 \

exp_2
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name exp_2 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 301 \
            --checkpoint_interval 50 \

UNet_exp_1(sigmoide)
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name UNet_exp_1 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 301 \
            --checkpoint_interval 50 \

UNet_exp_2(sigmoide)
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name UNet_exp_2 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 401 \
            --checkpoint_interval 50 \

UNet_exp_3
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name UNet_exp_3 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 301 \
            --checkpoint_interval 50 \

UNet_exp_4
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name UNet_exp_4 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 401 \
            --checkpoint_interval 50 \

UNet_exp_5
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name UNet_exp_5 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model UNet \
            --num_epochs 501 \
            --checkpoint_interval 50 \


Pix_exp_1
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name Pix_exp_1 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Pix2Pix \
            --num_epochs 301 \
            --checkpoint_interval 50 \

Pix_exp_2
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name Pix_exp_2 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Pix2Pix \
            --num_epochs 401 \
            --checkpoint_interval 50 \

Pix_exp_3
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name Pix_exp_3 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Pix2Pix \
            --num_epochs 501 \
            --checkpoint_interval 50 \

UNetR-exp_4
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name UNetR-exp_4 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Mask_UNet \
            --num_epochs 501 \
            --checkpoint_interval 50 \

PixR-exp_1
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name PixR-exp_1 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Mask_Pix2Pix \
            --num_epochs 301 \
            --checkpoint_interval 50 \

PixR-exp_2
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name PixR-exp_2 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Mask_Pix2Pix \
            --num_epochs 401 \
            --checkpoint_interval 50 \

PixR-exp_3
source /home/estudiante1/venvs/bd_env/bin/activate           
python run_model.py --gpus 0 \
            --exp_name PixR-exp_3 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/fram_template/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Mask_Pix2Pix \
            --num_epochs 501 \
            --checkpoint_interval 50 \


Pix_Mask_R_Epx_1
source /home/estudiante1/venvs/bd_env/bin/activate           
python3 run_model.py --gpus 0 \
            --exp_name Pix_Mask_R_Epx_1 \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path /home/estudiante1/MRI_pj/segm_mri/BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 50 \
            --normalization min_max \
            --num_workers 4 \
            --model Mask_R_Pix2Pix \
            --num_epochs 501 \
            --checkpoint_interval 50 \