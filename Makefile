# Setup colab
colab:
	cd /content/Landslide4Sense && git checkout develop && git pull origin develop
	unzip /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/TrainData.zip -d /content/Landslide4Sense/data/TrainData 
	unzip /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/ValidData.zip -d /content/Landslide4Sense/data/ValidData
	pip install gpustat

# Compile and install exact python packages
poetry:
	pip install poetry
	poetry install

