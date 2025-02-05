## Helmet Detection
This programme is using openncc for video capturing and nvidia jetson orin nano for inferencing. It is also compatible for other platforms for inferencing. The model can be any architectures but needs to be exported by torch.jit.save.

## installation
install openncc lib:
git clone https://github.com/EyecloudAi/opennccframe.git
sudo apt-get install libusb-dev libusb-1.0-0-dev
cd opennccframe
sudo ./install.sh
sudo apt-get install libopencv-dev
sudo ln -s /usr/include/opencv4/opencv2/ /usr/include/opencv2

install pytorch for jetson:
wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'
pip3 install numpy torch-2.3.0-cp310-cp310-linux_aarch64.whl

build executable file
mkdir build && cd build
(if cmake cant find nvcc): export PATH=/usr/local/cuda/bin:$PATH
cmake ..
make
sudo ./HD /dev/openncc ../HD.torchscript
