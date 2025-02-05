#include <includes.h>
#include <native_vpu_api.h>

#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

typedef struct
{
    void *start;
    size_t length;
} VideoSpec_t;

int  v4l2_device_open(char *node);
int  v4l2_device_start(int fd);
int  v4l2_device_fill(int fd, struct v4l2_buffer *src);
int  v4l2_device_queue(int fd, struct v4l2_buffer *src);
void v4l2_device_query(int fd, int *imageWidth, int *imageHeight);
VideoSpec_t* v4l2_device_allocate(int fd, struct v4l2_buffer *src);
void showimg(float* data, float scale_w, float scale_h, float confidenceThreshold, Mat imageBGR, vector<string> labels, vector<Scalar> colors, float scale);
torch::Tensor matToTensor(Mat imageBGR, int inputWidth, int inputHeight);

int main(int argc, char* argv[])
{
    if(argc <3)
    {
        cout << "Please input sudo ./HD [<node(/dev/video*)>] ./path/to/model.torchscript" << endl;
        return 0;
    }

    int imageWidth, imageHeight;
    /* input size of model */
    int inputWidth  = 640;
    int inputHeight = 640;
    
    /* labels of classification */
    vector<string> labels = {"With Helmet", "Without Helmet"};
    vector<Scalar> colors = {Scalar(255,0,0), Scalar(0,0,255)};
    
    float confidenceThreshold = 0.5f; 
    cv::Mat imageBGR;
    
    int ret = 0;
    /* detect device */
    int devNum = ncc_dev_number_get();
    printf("ncc_dev_number_get find devs %d\n", devNum);
    if(devNum<=0)
    {
        printf("ncc_dev_number_get error, the device was not found\n");
        return -1;
    }

    /* initialize device and load firmware */
    ret = ncc_dev_init("/usr/lib/openncc/OpenNcc.mvcmd", devNum);
    if(ret<0)
    {
        printf("ncc_dev_init error\n");
        return -1;
    }
    else
    {
        printf("ncc_dev_init success num %d\n",ret);
    }

    /* start device */
    ret = ncc_dev_start(0);
    if(ret>0)
    {
        printf("ncc_dev_start error %d !\n", ret);
        return -1;
    }
    
    /* check for CUDA availability and select device */
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        cout << "CUDA is available! Using GPU." << endl;
        device_type = torch::kCUDA;
    } else {
        cout << "CUDA is not available. Using CPU." << endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    /* load TorchScript model */
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[2], device);
    } catch (const c10::Error& e) {
        cerr << "Error loading the model from " << argv[2] << endl;
        return -1;
    }
    model.to(device);
    model.eval();

    /* open UVC camera */
    int fd;
    fd = v4l2_device_open(argv[1]);

    /* query camera video format */
    v4l2_device_query(fd, &imageWidth, &imageHeight);

    /* allocate memory for frames */
    struct v4l2_buffer src;
    VideoSpec_t *frame = v4l2_device_allocate(fd, &src);
    if(frame == 0)
        exit(0);

    /* start capturing images */
    ret = v4l2_device_start(fd);
    if(ret == -1)
    {
        perror("VIDIOC_STREAMON failed!\n");
    }
    
    float scale_w = imageWidth  / float(inputWidth);
    float scale_h = imageHeight / float(inputHeight);
    float scale = 1.0*960/imageWidth;

    while(1)
    {
        /* read video frame */
        ret = v4l2_device_fill(fd, &src);
        if (ret == -1)
        {
            perror("VIDIOC_DQBUF failed!\n");
            usleep(10000);
            continue;
        }
        
        /* read YUV image from frame */
    	imageBGR.create(imageHeight * 3 / 2, imageWidth, CV_8UC1);
    	imageBGR.data = (unsigned char*)frame[src.index].start;
    	
    	/* convert YUV image to BGR */
    	cvtColor(imageBGR, imageBGR, CV_YUV2BGR_I420);
    
	/* convert BGR image to Tensor */
    	torch::Tensor tensorImage;
    	tensorImage = matToTensor(imageBGR, inputWidth, inputHeight);
    	
    	/* inference */
    	tensorImage = tensorImage.to(device);
    	torch::NoGradGuard noGrad;
    	vector<torch::jit::IValue> inputs;
    	inputs.push_back(tensorImage);
    	torch::Tensor output = model.forward(inputs).toTensor();
    	
    	/* process output data */
    	output = output.reshape({300, 6});
        output = output.cpu();
    	float* data = output.data_ptr<float>();

	/* display outcome */
    	showimg(data, scale_w, scale_h, confidenceThreshold, imageBGR, labels, colors, scale);

        /* re-queued frame buffer */
        ret = v4l2_device_queue(fd, &src);
        if(ret == -1)
        {
            perror("VIDIOC_QBUF failed!\n");
            continue;
        }
    }
}

torch::Tensor matToTensor(Mat imageBGR, int inputWidth, int inputHeight)
{
    Mat imageRGB;
    cvtColor(imageBGR, imageRGB, COLOR_BGR2RGB);
    resize(imageRGB, imageRGB, Size(inputWidth, inputHeight));
    torch::Tensor tensorImage = torch::from_blob(imageRGB.data, {imageRGB.rows, imageRGB.cols, 3}, torch::kByte);
    tensorImage = tensorImage.permute({2, 0, 1});
    tensorImage = tensorImage.toType(torch::kFloat);
    tensorImage = tensorImage.div(255.0);
    tensorImage = tensorImage.unsqueeze(0);
    return tensorImage;
}

void showimg(float* data, float scale_w, float scale_h, float confidenceThreshold, Mat imageBGR, vector<string> labels, vector<Scalar> colors, float scale)
{
    for (int i = 0; i < 300; ++i)
    {
     
        if (data[i * 6 + 4] > confidenceThreshold)
        {
            int bx1 = static_cast<int>(data[i * 6 + 0] * scale_w);
            int by1 = static_cast<int>(data[i * 6 + 1] * scale_h);
            int bx2 = static_cast<int>(data[i * 6 + 2] * scale_w);
            int by2 = static_cast<int>(data[i * 6 + 3] * scale_h);
            	
       	    int classIndex = static_cast<int>(data[i * 6 + 5]);
                
            rectangle(imageBGR, Point(bx1, by1), Point(bx2, by2), colors[classIndex], 2);
            string text = labels[classIndex] + " Conf: " + to_string(data[i * 6 + 4]).substr(0,4);
            putText(imageBGR, text, Point(bx1, max(by1-5, 20)), FONT_HERSHEY_SIMPLEX, 0.5, colors[classIndex], 1);
        }

    }
    resize(imageBGR,imageBGR,Size(imageBGR.cols*scale,imageBGR.rows*scale),0,0,INTER_LINEAR);
    imshow("video_capture", imageBGR);
    waitKey(1);
}


int v4l2_device_open(char *node)
{
    int fd;
    fd = open(node,O_RDWR);

    return fd;
}

void v4l2_device_query(int fd, int *imageWidth, int *imageHeight)
{
    struct v4l2_format fmt;
    fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_G_FMT, &fmt);

    *imageWidth = fmt.fmt.pix.width;
    *imageHeight = fmt.fmt.pix.height;
    printf("Current data format information:\n\twidth:%d\n\theight:%d\n",fmt.fmt.pix.width,fmt.fmt.pix.height);

    struct v4l2_fmtdesc fmtdesc;
    fmtdesc.index = 0;
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    printf("Support format:\n");
    while(ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc)!=-1)
    {
        printf("\t%d.%c%c%c%c\t%s\n",fmtdesc.index+1,fmtdesc.pixelformat & 0xFF,\
                (fmtdesc.pixelformat >> 8) & 0xFF,(fmtdesc.pixelformat >> 16) & 0xFF, (fmtdesc.pixelformat >> 24) & 0xFF,fmtdesc.description);
        fmtdesc.index++;
    }
}

VideoSpec_t* v4l2_device_allocate(int fd, struct v4l2_buffer *src)
{
    struct v4l2_requestbuffers reqbuf;
    reqbuf.count = 4;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;

    if(ioctl(fd, VIDIOC_REQBUFS, &reqbuf) == -1)
    {
        perror("VIDIOC_REQBUFS failed!\n");
        return (VideoSpec_t*)0;
    }

    VideoSpec_t *frame;
    frame = (VideoSpec_t*)calloc(reqbuf.count,sizeof(VideoSpec_t));

    for(int i=0;i<reqbuf.count;i++)
    {
        src->index = i;
        src->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        src->memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_QUERYBUF, src) == -1)
        {
            perror("VIDIOC_QUERYBUF failed!\n");
            return (VideoSpec_t*)0;
        }

        frame[i].length = src->length;
        frame[i].start = mmap(NULL, src->length,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd, src->m.offset);

        if (frame[i].start == MAP_FAILED)
        {
            perror("mmap failed!\n");
            return (VideoSpec_t*)0;
        }

        if (ioctl(fd, VIDIOC_QBUF, src) == -1)
        {
            perror("VIDIOC_QBUF failed!\n");
            return (VideoSpec_t*)0;
        }
    }

    return frame;
}

int v4l2_device_start(int fd)
{
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int ret = ioctl(fd, VIDIOC_STREAMON, &type);

    return ret;
}

int v4l2_device_fill(int fd, struct v4l2_buffer *src)
{
    src->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    src->memory = V4L2_MEMORY_MMAP;
    int ret = ioctl(fd, VIDIOC_DQBUF, src);

    return ret;
}

int v4l2_device_queue(int fd, struct v4l2_buffer *src)
{
    int ret = ioctl(fd, VIDIOC_QBUF, src);
    return ret;
}
