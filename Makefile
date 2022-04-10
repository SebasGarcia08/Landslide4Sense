# Setup colab
setup-colab:
	cd /content/Landslide4Sense && git checkout develop && git pull origin develop

	mkdir /content/Landslide4Sense/data /content/Landslide4Sense/models /content/Landslide4Sense/submissions
	
	unzip /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/TrainData.zip -d /content/Landslide4Sense/data/TrainData 
	
	unzip /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/ValidData.zip -d /content/Landslide4Sense/data/ValidData
	
	cp /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/*.txt /content/Landslide4Sense/data/ 
	
	cp -r /content/drive/MyDrive/data_science_projects/landslide4sense-2022/models/* /content/Landslide4Sense/models/ 
	
	pip install gpustat
	
	pip install -e "/content/Landslide4Sense[dev]"

# Compile and install exact python packages
poetry:
	pip install poetry
	poetry install

train:
	python scripts/train.py \
	    --data_dir ./data/ \
	    --gpu_id 0\
		--num_workers 2\
		--bacth_size 64\
	    --snapshot_dir /content/drive/MyDrive/data_science_projects/landslide4sense-2022/models/

predict:
	python scripts/predict.py\
		--data_dir ./data/ \
		--gpu_id 0 \
		--test_list ./data/valid.txt \
		--snapshot_dir /content/Landslide4Sense/submissions \
		--restore_from /content/drive/MyDrive/data_science_projects/landslide4sense-2022/models/baseline/epoch7.pth