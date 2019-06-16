# Download dataset from Dropbox
wget -O raw_classical_music_data.zip https://www.dropbox.com/s/vvfo9dq400kw9og/raw_classical_music_data.zip?dl=1

# Unzip the downloaded zip file
unzip ./raw_classical_music_data.zip -d raw_classical_music_data

# Remove the downloaded zip file
rm ./raw_classical_music_data.zip

rm ./
mv ./raw_classical_music_data/raw_classical_music_data ./
# Install the dependencies
pip3 install --upgrade pip3

pip3 install pypianoroll
