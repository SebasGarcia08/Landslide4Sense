# Setup colab
setup-colab:
	cd /content/Landslide4Sense && git checkout develop && git pull origin develop
	mkdir /content/Landslide4Sense/data /content/Landslide4Sense/models /content/Landslide4Sense/submissions
	unzip /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/TrainData.zip -d /content/Landslide4Sense/data/TrainData 
	unzip /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/ValidData.zip -d /content/Landslide4Sense/data/ValidData
	cp /content/drive/MyDrive/data_science_projects/landslide4sense-2022/data/*.txt /content/Landslide4Sense/data/ 
	cp -r /content/drive/MyDrive/data_science_projects/landslide4sense-2022/models/* /content/Landslide4Sense/models/ 
	pip install gpustat
	pip install -e /content/Landslide4Sense 

# Compile and install exact python packages
poetry:
	pip install poetry
	poetry install

