echo 'All these steps may take a while. So grab a cup of coffee.';
sleep 3;
sudo apt -y install python3;
sudo apt-get -y install python3-pyaudio;
sudo python3 -m pip install numpy;
sudo python3 -m pip install scipy;
sudo python3 -m pip install tensorflow-gpu==1.11 || sudo python3 -m pip install tensorflow==1.5;
