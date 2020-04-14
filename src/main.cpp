#if SNIPPET001

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>


using namespace cv;
using namespace std;




Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;


//--------------------------------【onMouse( )回调函数】------------------------------------
//      描述：鼠标操作回调
//-------------------------------------------------------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject) {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch (event) {
    //此句代码的OpenCV2版为：
    //case CV_EVENT_LBUTTONDOWN:
    //此句代码的OpenCV3版为：
    case EVENT_LBUTTONDOWN:
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        break;

    //此句代码的OpenCV2版为：
    //case CV_EVENT_LBUTTONUP:
    //此句代码的OpenCV3版为：
    case EVENT_LBUTTONUP:
        selectObject = false;

        if (selection.width > 0 && selection.height > 0)
            trackObject = -1;

        break;
    }
}

//--------------------------------【help( )函数】----------------------------------------------
//      描述：输出帮助信息
//-------------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    cout << "\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n"
         << "\n\n\t\t\t此为本书OpenCV3版的第8个配套示例程序\n"
         << "\n\n\t\t\t   当前使用的OpenCV版本为：" << CV_VERSION
         << "\n\n  ----------------------------------------------------------------------------";
    cout << "\n\n\t此Demo显示了基于均值漂移的追踪（tracking）技术\n"
         "\t请用鼠标框选一个有颜色的物体，对它进行追踪操作\n";
    cout << "\n\n\t操作说明： \n"
         "\t\t用鼠标框选对象来初始化跟踪\n"
         "\t\tESC - 退出程序\n"
         "\t\tc - 停止追踪\n"
         "\t\tb - 开/关-投影视图\n"
         "\t\th - 显示/隐藏-对象直方图\n"
         "\t\tp - 暂停视频\n";
}

const char* keys = {
    "{1|  | 0 | camera number}"
};



//-----------------------------------【main( )函数】--------------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
    {
        ShowHelpText();
        //VideoCapture cap;
        Rect trackWindow;
        int hsize = 16;
        float hranges[] = { 0, 180 };
        const float* phranges = hranges;
        /*cap.open(0);

        if (!cap.isOpened())
        {
            cout << "不能初始化摄像头\n";
        }*/
        namedWindow("Histogram", 0);
        namedWindow("CamShift Demo", 0);
        setMouseCallback("CamShift Demo", onMouse, 0);
        createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
        createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0);
        createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);
        Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
        bool paused = false;

        for (;;) {
            if (!paused) {
                //cap >> frame;
                if (frame.empty())
                    break;
            }

            frame.copyTo(image);

            if (!paused) {
                cvtColor(image, hsv, COLOR_BGR2HSV);

                if (trackObject) {
                    int _vmin = vmin, _vmax = vmax;
                    inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
                            Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                    int ch[] = { 0, 0 };
                    hue.create(hsv.size(), hsv.depth());
                    mixChannels(&hsv, 1, &hue, 1, ch, 1);

                    if (trackObject < 0) {
                        Mat roi(hue, selection), maskroi(mask, selection);
                        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                        //此句代码的OpenCV3版为：
                        normalize(hist, hist, 0, 255, NORM_MINMAX);
                        //此句代码的OpenCV2版为：
                        //normalize(hist, hist, 0, 255, CV_MINMAX);
                        trackWindow = selection;
                        trackObject = 1;
                        histimg = Scalar::all(0);
                        int binW = histimg.cols / hsize;
                        Mat buf(1, hsize, CV_8UC3);

                        for (int i = 0; i < hsize; i++)
                            buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);

                        //此句代码的OpenCV3版为：
                        cvtColor(buf, buf, COLOR_HSV2BGR);
                        //此句代码的OpenCV2版为：
                        //cvtColor(buf, buf, CV_HSV2BGR);

                        for (int i = 0; i < hsize; i++) {
                            int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
                            rectangle(histimg, Point(i * binW, histimg.rows),
                                      Point((i + 1)*binW, histimg.rows - val),
                                      Scalar(buf.at<Vec3b>(i)), -1, 8);
                        }
                    }

                    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                    backproj &= mask;
                    RotatedRect trackBox = CamShift(backproj, trackWindow,
                                                    //此句代码的OpenCV3版为：
                                                    TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
                    //此句代码的OpenCV2版为：
                    //TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                    if (trackWindow.area() <= 1) {
                        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                           trackWindow.x + r, trackWindow.y + r) &
                                      Rect(0, 0, cols, rows);
                    }

                    if (backprojMode)
                        cvtColor(backproj, image, COLOR_GRAY2BGR);

                    //此句代码的OpenCV3版为：
                    ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
                    //此句代码的OpenCV2版为：
                    //ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
                }
            } else if (trackObject < 0)
                paused = false;

            if (selectObject && selection.width > 0 && selection.height > 0) {
                Mat roi(image, selection);
                bitwise_not(roi, roi);
            }

            imshow("CamShift Demo", image);
            imshow("Histogram", histimg);
            char c = (char)waitKey(10);

            if (c == 27)
                break;

            switch (c) {
            case 'b':
                backprojMode = !backprojMode;
                break;

            case 'c':
                trackObject = 0;
                histimg = Scalar::all(0);
                break;

            case 'h':
                showHist = !showHist;

                if (!showHist)
                    destroyWindow("Histogram");
                else
                    namedWindow("Histogram", 1);

                break;

            case 'p':
                paused = !paused;
                break;

            default:
                ;
            }
        }
    }
    {
        // 载入原图
        Mat image = imread("1.jpg");
        //创建窗口
        namedWindow("方框滤波【原图】");
        namedWindow("方框滤波【效果图】");
        //显示原图
        imshow("方框滤波【原图】", image);
        //进行方框滤波操作
        Mat out;
        boxFilter(image, out, -1, Size(5, 5));
        //显示效果图
        imshow("方框滤波【效果图】", out);
        waitKey(0);
    }
    {
        //【1】载入原始图
        Mat srcImage = imread("1.jpg");
        //【2】显示原始图
        imshow("均值滤波【原图】", srcImage);
        //【3】进行均值滤波操作
        Mat dstImage;
        blur(srcImage, dstImage, Size(7, 7));
        //【4】显示效果图
        imshow("均值滤波【效果图】", dstImage);
        waitKey(0);
    }
    {
        // 载入原图
        Mat image = imread("1.jpg");
        //创建窗口
        namedWindow("高斯滤波【原图】");
        namedWindow("高斯滤波【效果图】");
        //显示原图
        imshow("高斯滤波【原图】", image);
        //进行高斯滤波操作
        Mat out;
        GaussianBlur(image, out, Size(5, 5), 0, 0);
        //显示效果图
        imshow("高斯滤波【效果图】", out);
        waitKey(0);
    }
    {
        // 载入原图
        Mat image = imread("1.jpg");
        //创建窗口
        namedWindow("中值滤波【原图】");
        namedWindow("中值滤波【效果图】");
        //显示原图
        imshow("中值滤波【原图】", image);
        //进行中值滤波操作
        Mat out;
        medianBlur(image, out, 7);
        //显示效果图
        imshow("中值滤波【效果图】", out);
        waitKey(0);
    }
    {
        // 载入原图
        Mat image = imread("1.jpg");
        //创建窗口
        namedWindow("双边滤波【原图】");
        namedWindow("双边滤波【效果图】");
        //显示原图
        imshow("双边滤波【原图】", image);
        //进行双边滤波操作
        Mat out;
        bilateralFilter(image, out, 25, 25 * 2, 25 / 2);
        //显示效果图
        imshow("双边滤波【效果图】", out);
        waitKey(0);
    }
    {
        // 【1】加载源图像
        Mat srcImage, dstImage;
        srcImage = imread("1.jpg", 1);

        if (!srcImage.data) {
            printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~！ \n");
            return false;
        }

        // 【2】转为灰度图并显示出来
        cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
        imshow("原始图", srcImage);
        // 【3】进行直方图均衡化
        equalizeHist(srcImage, dstImage);
        // 【4】显示结果
        imshow("经过直方图均衡化后的图", dstImage);
        // 等待用户按键退出程序
        waitKey(0);
    }
    {
        //载入原始图
        Mat image = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
        //创建窗口
        namedWindow("【原始图】膨胀");
        namedWindow("【效果图】膨胀");
        //显示原始图
        imshow("【原始图】膨胀", image);
        //定义核
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //进行形态学操作
        morphologyEx(image, image, MORPH_DILATE, element);
        //显示效果图
        imshow("【效果图】膨胀", image);
        waitKey(0);
    }
    {
        //载入原始图
        Mat image = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
        //创建窗口
        namedWindow("【原始图】腐蚀");
        namedWindow("【效果图】腐蚀");
        //显示原始图
        imshow("【原始图】腐蚀", image);
        //定义核
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //进行形态学操作
        morphologyEx(image, image, MORPH_ERODE, element);
        //显示效果图
        imshow("【效果图】腐蚀", image);
        waitKey(0);
    }
    {
        //载入原始图
        Mat image = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
        //创建窗口
        namedWindow("【原始图】开运算");
        namedWindow("【效果图】开运算");
        //显示原始图
        imshow("【原始图】开运算", image);
        //定义核
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //进行形态学操作
        morphologyEx(image, image, MORPH_OPEN, element);
        //显示效果图
        imshow("【效果图】开运算", image);
        waitKey(0);
    }
    {
        //载入原始图
        Mat image = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
        //创建窗口
        namedWindow("【原始图】闭运算");
        namedWindow("【效果图】闭运算");
        //显示原始图
        imshow("【原始图】闭运算", image);
        //定义核
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //进行形态学操作
        morphologyEx(image, image, MORPH_CLOSE, element);
        //显示效果图
        imshow("【效果图】闭运算", image);
        waitKey(0);
    }
    {
        //载入原始图
        Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
        Mat srcImage1 = srcImage.clone();
        //显示原始图
        imshow("【原始图】Canny边缘检测", srcImage);
        //----------------------------------------------------------------------------------
        //  一、最简单的canny用法，拿到原图后直接用。
        //  注意：此方法在OpenCV2中可用，在OpenCV3中已失效
        //----------------------------------------------------------------------------------
        //Canny( srcImage, srcImage, 150, 100,3 );
        //imshow("【效果图】Canny边缘检测", srcImage);
        //----------------------------------------------------------------------------------
        //  二、高阶的canny用法，转成灰度图，降噪，用canny，最后将得到的边缘作为掩码，拷贝原图到效果图上，得到彩色的边缘图
        //----------------------------------------------------------------------------------
        Mat dstImage, edge, grayImage;
        // 【1】创建与src同类型和大小的矩阵(dst)
        dstImage.create(srcImage1.size(), srcImage1.type());
        // 【2】将原图像转换为灰度图像
        cvtColor(srcImage1, grayImage, COLOR_BGR2GRAY);
        // 【3】先用使用 3x3内核来降噪
        blur(grayImage, edge, Size(3, 3));
        // 【4】运行Canny算子
        Canny(edge, edge, 3, 9, 3);
        //【5】将g_dstImage内的所有元素设置为0
        dstImage = Scalar::all(0);
        //【6】使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中
        srcImage1.copyTo(dstImage, edge);
        //【7】显示效果图
        imshow("【效果图】Canny边缘检测2", dstImage);
        waitKey(0);
    }
    {
        // 【1】载入原始图，且必须以二值图模式载入
        Mat srcImage = imread("1.jpg", 0);
        imshow("原始图", srcImage);
        //【2】初始化结果图
        Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
        //【3】srcImage取大于阈值119的那部分
        srcImage = srcImage > 119;
        imshow("取阈值后的原始图", srcImage);
        //【4】定义轮廓和层次结构
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        //【5】查找轮廓
        //此句代码的OpenCV2版为：
        //findContours( srcImage, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
        //此句代码的OpenCV3版为：
        findContours(srcImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        // 【6】遍历所有顶层的轮廓， 以随机颜色绘制出每个连接组件颜色
        int index = 0;

        for (; index >= 0; index = hierarchy[index][0]) {
            Scalar color(rand() & 255, rand() & 255, rand() & 255);
            //此句代码的OpenCV2版为：
            //drawContours( dstImage, contours, index, color, CV_FILLED, 8, hierarchy );
            //此句代码的OpenCV3版为：
            drawContours(dstImage, contours, index, color, FILLED, 8, hierarchy);
        }

        //【7】显示最后的轮廓图
        imshow("轮廓图", dstImage);
        waitKey(0);
    }
    {
        //【0】载入原始图
        Mat srcImage = imread("1.jpg");  //工程目录下应该有一张名为1.jpg的素材图
        imshow("【原始图】Canny边缘检测", srcImage);    //显示原始图
        Mat dstImage, edge, grayImage;  //参数定义
        //【1】创建与src同类型和大小的矩阵(dst)
        dstImage.create(srcImage.size(), srcImage.type());
        //【2】将原图像转换为灰度图像
        //此句代码的OpenCV2版为：
        //cvtColor( srcImage, grayImage, CV_BGR2GRAY );
        //此句代码的OpenCV3版为：
        cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
        //【3】先用使用 3x3内核来降噪
        blur(grayImage, edge, Size(3, 3));
        //【4】运行Canny算子
        Canny(edge, edge, 3, 9, 3);
        //【5】显示效果图
        imshow("【效果图】Canny边缘检测", edge);
        waitKey(0);
    }
    return 0;
}

#endif


#if SNIPPET002

#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    // 【1】读入一张图片
    Mat img = imread("001.jpg");
    // 【2】在窗口中显示载入的图片
    imshow("【载入的图片】", img);
    // 【3】等待6000 ms后窗口自动关闭
    waitKey(6000);
}

#endif


#if SNIPPET003

#include <opencv2/opencv.hpp>
using namespace cv;

void main()
{
    // 【1】读入一张图片，载入图像
    Mat srcImage = imread("002.jpg");
    // 【2】显示载入的图片
    imshow("【原始图】", srcImage);
    // 【3】等待任意按键按下
    waitKey(0);
}

#endif


#if SNIPPET004

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;


int main()
{
    //载入原图
    Mat srcImage = imread("003.jpg");
    //显示原图
    imshow("【原图】腐蚀操作", srcImage);
    //进行腐蚀操作
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat dstImage;
    erode(srcImage, dstImage, element);
    //显示效果图
    imshow("【效果图】腐蚀操作", dstImage);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET005

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;


int main()
{
    //【1】载入原始图
    Mat srcImage = imread("005.jpg");
    //【2】显示原始图
    imshow("均值滤波【原图】", srcImage);
    //【3】进行均值滤波操作
    Mat dstImage;
    blur(srcImage, dstImage, Size(7, 7));
    //【4】显示效果图
    imshow("均值滤波【效果图】", dstImage);
    waitKey(0);
}

#endif


#if SNIPPET006

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


int main()
{
    //【0】载入原始图
    Mat srcImage = imread("006.jpg");  //工程目录下应该有一张名为1.jpg的素材图
    imshow("【原始图】Canny边缘检测", srcImage);    //显示原始图
    Mat dstImage, edge, grayImage;  //参数定义
    //【1】创建与src同类型和大小的矩阵(dst)
    dstImage.create(srcImage.size(), srcImage.type());
    //【2】将原图像转换为灰度图像
    //此句代码的OpenCV2版为：
    //cvtColor( srcImage, grayImage, CV_BGR2GRAY );
    //此句代码的OpenCV3版为：
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
    //【3】先用使用 3x3内核来降噪
    blur(grayImage, edge, Size(3, 3));
    //【4】运行Canny算子
    Canny(edge, edge, 3, 9, 3);
    //【5】显示效果图
    imshow("【效果图】Canny边缘检测", edge);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET007

#include <opencv2\opencv.hpp>

using namespace cv;

int main()
{
    //【1】读入视频
    VideoCapture capture("007.avi");

    //【2】循环显示每一帧
    while (1) {
        Mat frame;//定义一个Mat变量，用于存储每一帧的图像
        capture >> frame;  //读取当前帧

        //若视频播放完成，退出循环
        if (frame.empty()) {
            break;
        }

        imshow("读取视频", frame);  //显示当前帧
        waitKey(30);  //延时30ms
    }

    return 0;
}

#endif


#if SNIPPET008

#include <opencv2\opencv.hpp>
using namespace cv;


int main()
{
    //【1】从摄像头读入视频
    VideoCapture capture(0);

    //【2】循环显示每一帧
    while (1) {
        Mat frame;  //定义一个Mat变量，用于存储每一帧的图像
        capture >> frame;  //读取当前帧
        imshow("读取视频", frame);  //显示当前帧
        waitKey(30);  //延时30ms
    }

    return 0;
}

#endif


#if SNIPPET009

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;



//-----------------------------------【全局变量声明】-----------------------------------------
//      描述：声明全局变量
//-------------------------------------------------------------------------------------------------
Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;


//--------------------------------【onMouse( )回调函数】------------------------------------
//      描述：鼠标操作回调
//-------------------------------------------------------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject) {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch (event) {
    //此句代码的OpenCV2版为：
    //case CV_EVENT_LBUTTONDOWN:
    //此句代码的OpenCV3版为：
    case EVENT_LBUTTONDOWN:
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        break;

    //此句代码的OpenCV2版为：
    //case CV_EVENT_LBUTTONUP:
    //此句代码的OpenCV3版为：
    case EVENT_LBUTTONUP:
        selectObject = false;

        if (selection.width > 0 && selection.height > 0)
            trackObject = -1;

        break;
    }
}

//--------------------------------【help( )函数】----------------------------------------------
//      描述：输出帮助信息
//-------------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    cout << "\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n"
         << "\n\n\t\t\t此为本书OpenCV3版的第8个配套示例程序\n"
         << "\n\n\t\t\t   当前使用的OpenCV版本为：" << CV_VERSION
         << "\n\n  ----------------------------------------------------------------------------";
    cout << "\n\n\t此Demo显示了基于均值漂移的追踪（tracking）技术\n"
         "\t请用鼠标框选一个有颜色的物体，对它进行追踪操作\n";
    cout << "\n\n\t操作说明： \n"
         "\t\t用鼠标框选对象来初始化跟踪\n"
         "\t\tESC - 退出程序\n"
         "\t\tc - 停止追踪\n"
         "\t\tb - 开/关-投影视图\n"
         "\t\th - 显示/隐藏-对象直方图\n"
         "\t\tp - 暂停视频\n";
}

const char* keys = {
    "{1|  | 0 | camera number}"
};


//-----------------------------------【main( )函数】--------------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始
//-------------------------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    ShowHelpText();
    VideoCapture cap;
    Rect trackWindow;
    int hsize = 16;
    float hranges[] = { 0, 180 };
    const float* phranges = hranges;
    cap.open(0);

    if (!cap.isOpened()) {
        cout << "不能初始化摄像头\n";
    }

    namedWindow("Histogram", 0);
    namedWindow("CamShift Demo", 0);
    setMouseCallback("CamShift Demo", onMouse, 0);
    createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
    createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0);
    createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);
    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;

    for (;;) {
        if (!paused) {
            cap >> frame;

            if (frame.empty())
                break;
        }

        frame.copyTo(image);

        if (!paused) {
            cvtColor(image, hsv, COLOR_BGR2HSV);

            if (trackObject) {
                int _vmin = vmin, _vmax = vmax;
                inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                int ch[] = { 0, 0 };
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if (trackObject < 0) {
                    Mat roi(hue, selection), maskroi(mask, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    //此句代码的OpenCV3版为：
                    normalize(hist, hist, 0, 255, NORM_MINMAX);
                    //此句代码的OpenCV2版为：
                    //normalize(hist, hist, 0, 255, CV_MINMAX);
                    trackWindow = selection;
                    trackObject = 1;
                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);

                    for (int i = 0; i < hsize; i++)
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);

                    //此句代码的OpenCV3版为：
                    cvtColor(buf, buf, COLOR_HSV2BGR);
                    //此句代码的OpenCV2版为：
                    //cvtColor(buf, buf, CV_HSV2BGR);

                    for (int i = 0; i < hsize; i++) {
                        int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
                        rectangle(histimg, Point(i * binW, histimg.rows),
                                  Point((i + 1)*binW, histimg.rows - val),
                                  Scalar(buf.at<Vec3b>(i)), -1, 8);
                    }
                }

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                                //此句代码的OpenCV3版为：
                                                TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
                //此句代码的OpenCV2版为：
                //TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                if (trackWindow.area() <= 1) {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                if (backprojMode)
                    cvtColor(backproj, image, COLOR_GRAY2BGR);

                //此句代码的OpenCV3版为：
                ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
                //此句代码的OpenCV2版为：
                //ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
            }
        } else if (trackObject < 0)
            paused = false;

        if (selectObject && selection.width > 0 && selection.height > 0) {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }

        imshow("CamShift Demo", image);
        imshow("Histogram", histimg);
        char c = (char)waitKey(10);

        if (c == 27)
            break;

        switch (c) {
        case 'b':
            backprojMode = !backprojMode;
            break;

        case 'c':
            trackObject = 0;
            histimg = Scalar::all(0);
            break;

        case 'h':
            showHist = !showHist;

            if (!showHist)
                destroyWindow("Histogram");
            else
                namedWindow("Histogram", 1);

            break;

        case 'p':
            paused = !paused;
            break;

        default:
            ;
        }
    }

    return 0;
}

#endif


#if SNIPPET010

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;





//-----------------------------------【全局函数声明】-----------------------------------------
//      描述：声明全局函数
//-------------------------------------------------------------------------------------------------
void tracking(Mat &frame, Mat &output);
bool addNewPoints();
bool acceptTrackedPoint(int i);

//-----------------------------------【全局变量声明】-----------------------------------------
//      描述：声明全局变量
//-------------------------------------------------------------------------------------------------
string window_name = "optical flow tracking";
Mat gray;   // 当前图片
Mat gray_prev;  // 预测图片
vector<Point2f> points[2];  // point0为特征点的原来位置，point1为特征点的新位置
vector<Point2f> initial;    // 初始化跟踪点的位置
vector<Point2f> features;   // 检测的特征
int maxCount = 500; // 检测的最大特征数
double qLevel = 0.01;   // 特征检测的等级
double minDist = 10.0;  // 两特征点之间的最小距离
vector<uchar> status;   // 跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;


//--------------------------------【help( )函数】----------------------------------------------
//      描述：输出帮助信息
//-------------------------------------------------------------------------------------------------
static void help()
{
    //输出欢迎信息和OpenCV版本
    cout << "\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n"
         << "\n\n\t\t\t此为本书OpenCV3版的第9个配套示例程序\n"
         << "\n\n\t\t\t   当前使用的OpenCV版本为：" << CV_VERSION
         << "\n\n  ----------------------------------------------------------------------------";
}


//-----------------------------------【main( )函数】--------------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始
//-------------------------------------------------------------------------------------------------
int main()
{
    Mat frame;
    Mat result;
    VideoCapture capture("007.avi");
    help();

    if (capture.isOpened()) { // 摄像头读取文件开关
        while (true) {
            capture >> frame;

            if (!frame.empty()) {
                tracking(frame, result);
            } else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            int c = waitKey(50);

            if ((char)c == 27) {
                break;
            }
        }
    }

    return 0;
}

//-------------------------------------------------------------------------------------------------
// function: tracking
// brief: 跟踪
// parameter: frame 输入的视频帧
//            output 有跟踪结果的视频帧
// return: void
//-------------------------------------------------------------------------------------------------
void tracking(Mat &frame, Mat &output)
{
    //此句代码的OpenCV3版为：
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    //此句代码的OpenCV2版为：
    //cvtColor(frame, gray, CV_BGR2GRAY);
    frame.copyTo(output);

    // 添加特征点
    if (addNewPoints()) {
        goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);
        points[0].insert(points[0].end(), features.begin(), features.end());
        initial.insert(initial.end(), features.begin(), features.end());
    }

    if (gray_prev.empty()) {
        gray.copyTo(gray_prev);
    }

    // l-k光流法运动估计
    calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);
    // 去掉一些不好的特征点
    int k = 0;

    for (size_t i = 0; i < points[1].size(); i++) {
        if (acceptTrackedPoint(i)) {
            initial[k] = initial[i];
            points[1][k++] = points[1][i];
        }
    }

    points[1].resize(k);
    initial.resize(k);

    // 显示特征点和运动轨迹
    for (size_t i = 0; i < points[1].size(); i++) {
        line(output, initial[i], points[1][i], Scalar(0, 0, 255));
        circle(output, points[1][i], 3, Scalar(0, 255, 0), -1);
    }

    // 把当前跟踪结果作为下一此参考
    swap(points[1], points[0]);
    swap(gray_prev, gray);
    imshow(window_name, output);
}

//-------------------------------------------------------------------------------------------------
// function: addNewPoints
// brief: 检测新点是否应该被添加
// parameter:
// return: 是否被添加标志
//-------------------------------------------------------------------------------------------------
bool addNewPoints()
{
    return points[0].size() <= 10;
}

//-------------------------------------------------------------------------------------------------
// function: acceptTrackedPoint
// brief: 决定哪些跟踪点被接受
// parameter:
// return:
//-------------------------------------------------------------------------------------------------
bool acceptTrackedPoint(int i)
{
    return status[i] && ((abs(points[0][i].x - points[1][i].x) + abs(points[0][i].y - points[1][i].y)) > 2);
}

#endif


#if SNIPPET011


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

//-----------------------------------【宏定义部分】--------------------------------------------
//  描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "【滑动条的创建&线性混合示例】"        //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//      描述：全局变量声明
//-----------------------------------------------------------------------------------------------
const int g_nMaxAlphaValue = 100;//Alpha值的最大值
int g_nAlphaValueSlider;//滑动条对应的变量
double g_dAlphaValue;
double g_dBetaValue;

//声明存储图像的变量
Mat g_srcImage1;
Mat g_srcImage2;
Mat g_dstImage;


//-----------------------------------【on_Trackbar( )函数】--------------------------------
//      描述：响应滑动条的回调函数
//------------------------------------------------------------------------------------------
void on_Trackbar(int, void*)
{
    //求出当前alpha值相对于最大值的比例
    g_dAlphaValue = (double)g_nAlphaValueSlider / g_nMaxAlphaValue;
    //则beta值为1减去alpha值
    g_dBetaValue = (1.0 - g_dAlphaValue);
    //根据alpha和beta值进行线性混合
    addWeighted(g_srcImage1, g_dAlphaValue, g_srcImage2, g_dBetaValue, 0.0, g_dstImage);
    //显示效果图
    imshow(WINDOW_NAME, g_dstImage);
}


//-----------------------------【ShowHelpText( )函数】--------------------------------------
//      描述：输出帮助信息
//-------------------------------------------------------------------------------------------------
//-----------------------------------【ShowHelpText( )函数】----------------------------------
//      描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第17个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------【main( )函数】-----------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //显示帮助信息
    ShowHelpText();
    //加载图像 (两图像的尺寸需相同)
    g_srcImage1 = imread("011.jpg");
    g_srcImage2 = imread("012.jpg");

    if (!g_srcImage1.data) {
        printf("读取第一幅图片错误，请确定目录下是否有imread函数指定图片存在~！ \n");
        return -1;
    }

    if (!g_srcImage2.data) {
        printf("读取第二幅图片错误，请确定目录下是否有imread函数指定图片存在~！\n");
        return -1;
    }

    //设置滑动条初值为70
    g_nAlphaValueSlider = 70;
    //创建窗体
    namedWindow(WINDOW_NAME, 1);
    //在创建的窗体中创建一个滑动条控件
    char TrackbarName[50];
    sprintf(TrackbarName, "透明值 %d", g_nMaxAlphaValue);
    createTrackbar(TrackbarName, WINDOW_NAME, &g_nAlphaValueSlider, g_nMaxAlphaValue, on_Trackbar);
    //结果在回调函数中显示
    on_Trackbar(g_nAlphaValueSlider, 0);
    //按任意键退出
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET012

#include <opencv2/opencv.hpp>
using namespace cv;

//-----------------------------------【宏定义部分】--------------------------------------------
//  描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "【程序窗口】"        //为窗口标题定义的宏 


//-----------------------------------【全局函数声明部分】------------------------------------
//      描述：全局函数的声明
//------------------------------------------------------------------------------------------------
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawRectangle(cv::Mat& img, cv::Rect box);
void ShowHelpText();

//-----------------------------------【全局变量声明部分】-----------------------------------
//      描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Rect g_rectangle;
bool g_bDrawingBox = false;//是否进行绘制
RNG g_rng(12345);



//-----------------------------------【main( )函数】--------------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //【0】改变console字体颜色
    system("color 9F");
    //【0】显示欢迎和帮助文字
    ShowHelpText();
    //【1】准备参数
    g_rectangle = Rect(-1, -1, 0, 0);
    Mat srcImage(600, 800, CV_8UC3), tempImage;
    srcImage.copyTo(tempImage);
    g_rectangle = Rect(-1, -1, 0, 0);
    srcImage = Scalar::all(0);
    //【2】设置鼠标操作回调函数
    namedWindow(WINDOW_NAME);
    setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);

    //【3】程序主循环，当进行绘制的标识符为真时，进行绘制
    while (1) {
        srcImage.copyTo(tempImage);//拷贝源图到临时变量

        if (g_bDrawingBox) DrawRectangle(tempImage, g_rectangle);//当进行绘制的标识符为真，则进行绘制

        imshow(WINDOW_NAME, tempImage);

        if (waitKey(10) == 27) break;//按下ESC键，程序退出
    }

    return 0;
}



//--------------------------------【on_MouseHandle( )函数】-----------------------------
//      描述：鼠标回调函数，根据不同的鼠标事件进行不同的操作
//-----------------------------------------------------------------------------------------------
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    Mat& image = *(cv::Mat*) param;

    switch (event) {
    //鼠标移动消息
    case EVENT_MOUSEMOVE: {
        if (g_bDrawingBox) { //如果是否进行绘制的标识符为真，则记录下长和宽到RECT型变量中
            g_rectangle.width = x - g_rectangle.x;
            g_rectangle.height = y - g_rectangle.y;
        }
    }
    break;

    //左键按下消息
    case EVENT_LBUTTONDOWN: {
        g_bDrawingBox = true;
        g_rectangle = Rect(x, y, 0, 0);//记录起始点
    }
    break;

    //左键抬起消息
    case EVENT_LBUTTONUP: {
        g_bDrawingBox = false;//置标识符为false

        //对宽和高小于0的处理
        if (g_rectangle.width < 0) {
            g_rectangle.x += g_rectangle.width;
            g_rectangle.width *= -1;
        }

        if (g_rectangle.height < 0) {
            g_rectangle.y += g_rectangle.height;
            g_rectangle.height *= -1;
        }

        //调用函数进行绘制
        DrawRectangle(image, g_rectangle);
    }
    break;
    }
}



//-----------------------------------【DrawRectangle( )函数】------------------------------
//      描述：自定义的矩形绘制函数
//-----------------------------------------------------------------------------------------------
void DrawRectangle(cv::Mat& img, cv::Rect box)
{
    cv::rectangle(img, box.tl(), box.br(), cv::Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));//随机颜色
}


//-----------------------------------【ShowHelpText( )函数】-----------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第18个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\n\n\t欢迎来到【鼠标交互演示】示例程序\n");
    printf("\n\n\t请在窗口中点击鼠标左键并拖动以绘制矩形\n");
}

#endif


#if SNIPPET013

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;



//-----------------------------【ShowHelpText( )函数】--------------------------------------
//      描述：输出帮助信息
//-------------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第19个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\n\n\t欢迎来到【基本图像容器-Mat类】示例程序~\n\n");
    printf("\n\n\t程序说明：\n\n\t此示例程序用于演示Mat类的格式化输出功能，输出风格可为：");
    printf("\n\n\n\t【1】OpenCV默认风格");
    printf("\n\n\t【2】Python风格");
    printf("\n\n\t【3】逗号分隔风格");
    printf("\n\n\t【4】Numpy风格");
    printf("\n\n\t【5】C语言风格\n\n");
    printf("\n  --------------------------------------------------------------------------\n");
}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int, char**)
{
    //改变控制台的前景色和背景色
    system("color 8F");
    //显示帮助文字
    ShowHelpText();
    Mat I = Mat::eye(4, 4, CV_64F);
    I.at<double>(1, 1) = CV_PI;
    cout << "\nI = " << I << ";\n" << endl;
    Mat r = Mat(10, 3, CV_8UC3);
    randu(r, Scalar::all(0), Scalar::all(255));
    //此段代码的OpenCV2版为：
    //cout << "r (OpenCV默认风格) = " << r << ";" << endl << endl;
    //cout << "r (Python风格) = " << format(r,"python") << ";" << endl << endl;
    //cout << "r (Numpy风格) = " << format(r,"numpy") << ";" << endl << endl;
    //cout << "r (逗号分隔风格) = " << format(r,"csv") << ";" << endl<< endl;
    //cout << "r (C语言风格) = " << format(r,"C") << ";" << endl << endl;
    //此段代码的OpenCV3版为：
    cout << "r (OpenCV默认风格) = " << r << ";" << endl << endl;
    cout << "r (Python风格) = " << format(r, Formatter::FMT_PYTHON) << ";" << endl << endl;
    cout << "r (Numpy风格) = " << format(r, Formatter::FMT_NUMPY) << ";" << endl << endl;
    cout << "r (逗号分隔风格) = " << format(r, Formatter::FMT_CSV) << ";" << endl << endl;
    cout << "r (C语言风格) = " << format(r, Formatter::FMT_C) << ";" << endl << endl;
    Point2f p(6, 2);
    cout << "【2维点】p = " << p << ";\n" << endl;
    Point3f p3f(8, 2, 0);
    cout << "【3维点】p3f = " << p3f << ";\n" << endl;
    vector<float> v;
    v.push_back(3);
    v.push_back(5);
    v.push_back(7);
    cout << "【基于Mat的vector】shortvec = " << Mat(v) << ";\n" << endl;
    vector<Point2f> points(20);

    for (size_t i = 0; i < points.size(); ++i)
        points[i] = Point2f((float)(i * 5), (float)(i % 7));

    cout << "【二维点向量】points = " << points << ";";
    getchar();//按任意键退出
    return 0;
}


#endif


#if SNIPPET014

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

//此程序对于OpenCV3版需要额外包含头文件：
#include <opencv2/imgproc/imgproc.hpp>



//-----------------------------------【宏定义部分】--------------------------------------------
//      描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "【绘制图1】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【绘制图2】"        //为窗口标题定义的宏 
#define WINDOW_WIDTH 600//定义窗口大小的宏



//--------------------------------【全局函数声明部分】-------------------------------------
//      描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void DrawEllipse(Mat img, double angle);//绘制椭圆
void DrawFilledCircle(Mat img, Point center);//绘制圆
void DrawPolygon(Mat img);//绘制多边形
void DrawLine(Mat img, Point start, Point end);//绘制线段



//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第20个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}




//---------------------------------------【main( )函数】--------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(void)
{
    // 创建空白的Mat图像
    Mat atomImage = Mat::zeros(WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3);
    Mat rookImage = Mat::zeros(WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3);
    ShowHelpText();
    // ---------------------<1>绘制化学中的原子示例图------------------------
    //【1.1】先绘制出椭圆
    DrawEllipse(atomImage, 90);
    DrawEllipse(atomImage, 0);
    DrawEllipse(atomImage, 45);
    DrawEllipse(atomImage, -45);
    //【1.2】再绘制圆心
    DrawFilledCircle(atomImage, Point(WINDOW_WIDTH / 2, WINDOW_WIDTH / 2));
    // ----------------------------<2>绘制组合图-----------------------------
    //【2.1】先绘制出椭圆
    DrawPolygon(rookImage);
    // 【2.2】绘制矩形
    rectangle(rookImage,
              Point(0, 7 * WINDOW_WIDTH / 8),
              Point(WINDOW_WIDTH, WINDOW_WIDTH),
              Scalar(0, 255, 255),
              -1,
              8);
    // 【2.3】绘制一些线段
    DrawLine(rookImage, Point(0, 15 * WINDOW_WIDTH / 16), Point(WINDOW_WIDTH, 15 * WINDOW_WIDTH / 16));
    DrawLine(rookImage, Point(WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8), Point(WINDOW_WIDTH / 4, WINDOW_WIDTH));
    DrawLine(rookImage, Point(WINDOW_WIDTH / 2, 7 * WINDOW_WIDTH / 8), Point(WINDOW_WIDTH / 2, WINDOW_WIDTH));
    DrawLine(rookImage, Point(3 * WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8), Point(3 * WINDOW_WIDTH / 4, WINDOW_WIDTH));
    // ---------------------------<3>显示绘制出的图像------------------------
    imshow(WINDOW_NAME1, atomImage);
    moveWindow(WINDOW_NAME1, 0, 200);
    imshow(WINDOW_NAME2, rookImage);
    moveWindow(WINDOW_NAME2, WINDOW_WIDTH, 200);
    waitKey(0);
    return (0);
}



//-------------------------------【DrawEllipse( )函数】--------------------------------
//      描述：自定义的绘制函数，实现了绘制不同角度、相同尺寸的椭圆
//-----------------------------------------------------------------------------------------
void DrawEllipse(Mat img, double angle)
{
    int thickness = 2;
    int lineType = 8;
    ellipse(img,
            Point(WINDOW_WIDTH / 2, WINDOW_WIDTH / 2),
            Size(WINDOW_WIDTH / 4, WINDOW_WIDTH / 16),
            angle,
            0,
            360,
            Scalar(255, 129, 0),
            thickness,
            lineType);
}


//-----------------------------------【DrawFilledCircle( )函数】---------------------------
//      描述：自定义的绘制函数，实现了实心圆的绘制
//-----------------------------------------------------------------------------------------
void DrawFilledCircle(Mat img, Point center)
{
    int thickness = -1;
    int lineType = 8;
    circle(img,
           center,
           WINDOW_WIDTH / 32,
           Scalar(0, 0, 255),
           thickness,
           lineType);
}


//-----------------------------------【DrawPolygon( )函数】--------------------------
//      描述：自定义的绘制函数，实现了凹多边形的绘制
//--------------------------------------------------------------------------------------
void DrawPolygon(Mat img)
{
    int lineType = 8;
    //创建一些点
    Point rookPoints[1][20];
    rookPoints[0][0] = Point(WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8);
    rookPoints[0][1] = Point(3 * WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8);
    rookPoints[0][2] = Point(3 * WINDOW_WIDTH / 4, 13 * WINDOW_WIDTH / 16);
    rookPoints[0][3] = Point(11 * WINDOW_WIDTH / 16, 13 * WINDOW_WIDTH / 16);
    rookPoints[0][4] = Point(19 * WINDOW_WIDTH / 32, 3 * WINDOW_WIDTH / 8);
    rookPoints[0][5] = Point(3 * WINDOW_WIDTH / 4, 3 * WINDOW_WIDTH / 8);
    rookPoints[0][6] = Point(3 * WINDOW_WIDTH / 4, WINDOW_WIDTH / 8);
    rookPoints[0][7] = Point(26 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 8);
    rookPoints[0][8] = Point(26 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 4);
    rookPoints[0][9] = Point(22 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 4);
    rookPoints[0][10] = Point(22 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 8);
    rookPoints[0][11] = Point(18 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 8);
    rookPoints[0][12] = Point(18 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 4);
    rookPoints[0][13] = Point(14 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 4);
    rookPoints[0][14] = Point(14 * WINDOW_WIDTH / 40, WINDOW_WIDTH / 8);
    rookPoints[0][15] = Point(WINDOW_WIDTH / 4, WINDOW_WIDTH / 8);
    rookPoints[0][16] = Point(WINDOW_WIDTH / 4, 3 * WINDOW_WIDTH / 8);
    rookPoints[0][17] = Point(13 * WINDOW_WIDTH / 32, 3 * WINDOW_WIDTH / 8);
    rookPoints[0][18] = Point(5 * WINDOW_WIDTH / 16, 13 * WINDOW_WIDTH / 16);
    rookPoints[0][19] = Point(WINDOW_WIDTH / 4, 13 * WINDOW_WIDTH / 16);
    const Point* ppt[1] = { rookPoints[0] };
    int npt[] = { 20 };
    fillPoly(img,
             ppt,
             npt,
             1,
             Scalar(255, 255, 255),
             lineType);
}


//-----------------------------------【DrawLine( )函数】--------------------------
//      描述：自定义的绘制函数，实现了线的绘制
//---------------------------------------------------------------------------------
void DrawLine(Mat img, Point start, Point end)
{
    int thickness = 2;
    int lineType = 8;
    line(img,
         start,
         end,
         Scalar(0, 0, 0),
         thickness,
         lineType);
}

#endif


#if SNIPPET015

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

//-----------------------------------【全局函数声明部分】-----------------------------------
//          描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div);
void ShowHelpText();



//--------------------------------------【main( )函数】---------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    //【1】创建原始图并显示
    Mat srcImage = imread("001.jpg");
    imshow("原始图像", srcImage);
    //【2】按原始图的参数规格来创建创建效果图
    Mat dstImage;
    dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//效果图的大小、类型与原图片相同
    ShowHelpText();
    //【3】记录起始时间
    double time0 = static_cast<double>(getTickCount());
    //【4】调用颜色空间缩减函数
    colorReduce(srcImage, dstImage, 32);
    //【5】计算运行时间并输出
    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    cout << "\t此方法运行时间为： " << time0 << "秒" << endl;  //输出运行时间
    //【6】显示效果图
    imshow("效果图", dstImage);
    waitKey(0);
}


//---------------------------------【colorReduce( )函数】---------------------------------
//          描述：使用【指针访问：C操作符[ ]】方法版的颜色空间缩减函数
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
    //参数准备
    outputImage = inputImage.clone();  //拷贝实参到临时变量
    int rowNumber = outputImage.rows;  //行数
    int colNumber = outputImage.cols * outputImage.channels(); //列数 x 通道数=每一行元素的个数

    //双重循环，遍历所有的像素值
    for (int i = 0; i < rowNumber; i++) { //行循环
        uchar* data = outputImage.ptr<uchar>(i);  //获取第i行的首地址

        for (int j = 0; j < colNumber; j++) { //列循环
            // ---------【开始处理每个像素】-------------
            data[j] = data[j] / div * div + div / 2;
            // ----------【处理结束】---------------------
        }  //行处理结束
    }
}


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第21个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}

#endif



#if SNIPPET016

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;



//-----------------------------------【全局函数声明部分】-----------------------------------
//      描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div);
void ShowHelpText();



//--------------------------------------【main( )函数】--------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    //【1】创建原始图并显示
    Mat srcImage = imread("1.jpg");
    imshow("原始图像", srcImage);
    //【2】按原始图的参数规格来创建创建效果图
    Mat dstImage;
    dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//效果图的大小、类型与原图片相同
    ShowHelpText();
    //【3】记录起始时间
    double time0 = static_cast<double>(getTickCount());
    //【4】调用颜色空间缩减函数
    colorReduce(srcImage, dstImage, 32);
    //【5】计算运行时间并输出
    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    cout << "此方法运行时间为： " << time0 << "秒" << endl;  //输出运行时间
    //【6】显示效果图
    imshow("效果图", dstImage);
    waitKey(0);
}




//-------------------------------------【colorReduce( )函数】-----------------------------
//      描述：使用【迭代器】方法版的颜色空间缩减函数
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
    //参数准备
    outputImage = inputImage.clone();  //拷贝实参到临时变量
    //获取迭代器
    Mat_<Vec3b>::iterator it = outputImage.begin<Vec3b>();  //初始位置的迭代器
    Mat_<Vec3b>::iterator itend = outputImage.end<Vec3b>();  //终止位置的迭代器

    //存取彩色图像像素
    for (; it != itend; ++it) {
        // ------------------------【开始处理每个像素】--------------------
        (*it)[0] = (*it)[0] / div * div + div / 2;
        (*it)[1] = (*it)[1] / div * div + div / 2;
        (*it)[2] = (*it)[2] / div * div + div / 2;
        // ------------------------【处理结束】----------------------------
    }
}



//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第22个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}



#endif


#if SNIPPET017

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

//-----------------------------------【全局函数声明部分】-----------------------------------
//          描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div);
void ShowHelpText();


//--------------------------------------【main( )函数】---------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    system("color 9F");
    //【1】创建原始图并显示
    Mat srcImage = imread("1.jpg");
    imshow("原始图像", srcImage);
    //【2】按原始图的参数规格来创建创建效果图
    Mat dstImage;
    dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//效果图的大小、类型与原图片相同
    ShowHelpText();
    //【3】记录起始时间
    double time0 = static_cast<double>(getTickCount());
    //【4】调用颜色空间缩减函数
    colorReduce(srcImage, dstImage, 32);
    //【5】计算运行时间并输出
    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    cout << "此方法运行时间为： " << time0 << "秒" << endl;  //输出运行时间
    //【6】显示效果图
    imshow("效果图", dstImage);
    waitKey(0);
}


//----------------------------------【colorReduce( )函数】-------------------------------
//          描述：使用【动态地址运算配合at】方法版本的颜色空间缩减函数
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
    //参数准备
    outputImage = inputImage.clone();  //拷贝实参到临时变量
    int rowNumber = outputImage.rows;  //行数
    int colNumber = outputImage.cols;  //列数

    //存取彩色图像像素
    for (int i = 0; i < rowNumber; i++) {
        for (int j = 0; j < colNumber; j++) {
            // ------------------------【开始处理每个像素】--------------------
            outputImage.at<Vec3b>(i, j)[0] = outputImage.at<Vec3b>(i, j)[0] / div * div + div / 2;  //蓝色通道
            outputImage.at<Vec3b>(i, j)[1] = outputImage.at<Vec3b>(i, j)[1] / div * div + div / 2;  //绿色通道
            outputImage.at<Vec3b>(i, j)[2] = outputImage.at<Vec3b>(i, j)[2] / div * div + div / 2;  //红是通道
            // -------------------------【处理结束】----------------------------
        }  // 行处理结束
    }
}


//-------------------------------【ShowHelpText( )函数】--------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第23个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


#endif


#if SNIPPET018

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;



//---------------------------------【宏定义部分】---------------------------------------------
//      描述：包含程序所使用宏定义
//-------------------------------------------------------------------------------------------------
#define NTESTS 14
#define NITERATIONS 20



//----------------------------------------- 【方法一】-------------------------------------------
//      说明：利用.ptr 和 []
//-------------------------------------------------------------------------------------------------
void colorReduce0(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols * image.channels(); //每行元素的总元素数量

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            data[i] = data[i] / div * div + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}

//-----------------------------------【方法二】-------------------------------------------------
//      说明：利用 .ptr 和 * ++
//-------------------------------------------------------------------------------------------------
void colorReduce1(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols * image.channels(); //每行元素的总元素数量

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            *data++ = *data / div * div + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}

//-----------------------------------------【方法三】-------------------------------------------
//      说明：利用.ptr 和 * ++ 以及模操作
//-------------------------------------------------------------------------------------------------
void colorReduce2(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols * image.channels(); //每行元素的总元素数量

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            int v = *data;
            *data++ = v - v % div + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}

//----------------------------------------【方法四】---------------------------------------------
//      说明：利用.ptr 和 * ++ 以及位操作
//----------------------------------------------------------------------------------------------------
void colorReduce3(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols * image.channels(); //每行元素的总元素数量
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //掩码值
    uchar mask = 0xFF << n; // e.g. 对于 div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //------------开始处理每个像素-------------------
            *data++ = *data & mask + div / 2;
            //-------------结束像素处理------------------------
        }  //单行处理结束
    }
}


//----------------------------------------【方法五】----------------------------------------------
//      说明：利用指针算术运算
//---------------------------------------------------------------------------------------------------
void colorReduce4(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols * image.channels(); //每行元素的总元素数量
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    int step = image.step; //有效宽度
    //掩码值
    uchar mask = 0xFF << n; // e.g. 对于 div=16, mask= 0xF0
    //获取指向图像缓冲区的指针
    uchar *data = image.data;

    for (int j = 0; j < nl; j++) {
        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            *(data + i) = *data & mask + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束

        data += step;  // next line
    }
}

//---------------------------------------【方法六】----------------------------------------------
//      说明：利用 .ptr 和 * ++以及位运算、image.cols * image.channels()
//-------------------------------------------------------------------------------------------------
void colorReduce5(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //掩码值
    uchar mask = 0xFF << n; // e.g. 例如div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < image.cols * image.channels(); i++) {
            //-------------开始处理每个像素-------------------
            *data++ = *data & mask + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}

// -------------------------------------【方法七】----------------------------------------------
//      说明：利用.ptr 和 * ++ 以及位运算(continuous)
//-------------------------------------------------------------------------------------------------
void colorReduce6(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols * image.channels(); //每行元素的总元素数量

    if (image.isContinuous()) {
        //无填充像素
        nc = nc * nl;
        nl = 1;  // 为一维数列
    }

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //掩码值
    uchar mask = 0xFF << n; // e.g. 比如div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            *data++ = *data & mask + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}

//------------------------------------【方法八】------------------------------------------------
//      说明：利用 .ptr 和 * ++ 以及位运算 (continuous+channels)
//-------------------------------------------------------------------------------------------------
void colorReduce7(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols; //列数

    if (image.isContinuous()) {
        //无填充像素
        nc = nc * nl;
        nl = 1;  // 为一维数组
    }

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //掩码值
    uchar mask = 0xFF << n; // e.g. 比如div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            *data++ = *data & mask + div / 2;
            *data++ = *data & mask + div / 2;
            *data++ = *data & mask + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}


// -----------------------------------【方法九】 ------------------------------------------------
//      说明：利用Mat_ iterator
//-------------------------------------------------------------------------------------------------
void colorReduce8(Mat &image, int div = 64)
{
    //获取迭代器
    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    Mat_<Vec3b>::iterator itend = image.end<Vec3b>();

    for (; it != itend; ++it) {
        //-------------开始处理每个像素-------------------
        (*it)[0] = (*it)[0] / div * div + div / 2;
        (*it)[1] = (*it)[1] / div * div + div / 2;
        (*it)[2] = (*it)[2] / div * div + div / 2;
        //-------------结束像素处理------------------------
    }//单行处理结束
}

//-------------------------------------【方法十】-----------------------------------------------
//      说明：利用Mat_ iterator以及位运算
//-------------------------------------------------------------------------------------------------
void colorReduce9(Mat &image, int div = 64)
{
    // div必须是2的幂
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //掩码值
    uchar mask = 0xFF << n; // e.g. 比如 div=16, mask= 0xF0
    // 获取迭代器
    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    Mat_<Vec3b>::iterator itend = image.end<Vec3b>();

    //扫描所有元素
    for (; it != itend; ++it) {
        //-------------开始处理每个像素-------------------
        (*it)[0] = (*it)[0] & mask + div / 2;
        (*it)[1] = (*it)[1] & mask + div / 2;
        (*it)[2] = (*it)[2] & mask + div / 2;
        //-------------结束像素处理------------------------
    }//单行处理结束
}

//------------------------------------【方法十一】---------------------------------------------
//      说明：利用Mat Iterator_
//-------------------------------------------------------------------------------------------------
void colorReduce10(Mat &image, int div = 64)
{
    //获取迭代器
    Mat_<Vec3b> cimage = image;
    Mat_<Vec3b>::iterator it = cimage.begin();
    Mat_<Vec3b>::iterator itend = cimage.end();

    for (; it != itend; it++) {
        //-------------开始处理每个像素-------------------
        (*it)[0] = (*it)[0] / div * div + div / 2;
        (*it)[1] = (*it)[1] / div * div + div / 2;
        (*it)[2] = (*it)[2] / div * div + div / 2;
        //-------------结束像素处理------------------------
    }
}

//--------------------------------------【方法十二】--------------------------------------------
//      说明：利用动态地址计算配合at
//-------------------------------------------------------------------------------------------------
void colorReduce11(Mat &image, int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols; //列数

    for (int j = 0; j < nl; j++) {
        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            image.at<Vec3b>(j, i)[0] = image.at<Vec3b>(j, i)[0] / div * div + div / 2;
            image.at<Vec3b>(j, i)[1] = image.at<Vec3b>(j, i)[1] / div * div + div / 2;
            image.at<Vec3b>(j, i)[2] = image.at<Vec3b>(j, i)[2] / div * div + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}

//----------------------------------【方法十三】-----------------------------------------------
//      说明：利用图像的输入与输出
//-------------------------------------------------------------------------------------------------
void colorReduce12(const Mat &image, //输入图像
                   Mat &result,      // 输出图像
                   int div = 64)
{
    int nl = image.rows; //行数
    int nc = image.cols; //列数
    //准备好初始化后的Mat给输出图像
    result.create(image.rows, image.cols, image.type());
    //创建无像素填充的图像
    nc = nc * nl;
    nl = 1;  //单维数组
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //掩码值
    uchar mask = 0xFF << n; // e.g.比如div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = result.ptr<uchar>(j);
        const uchar* idata = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------开始处理每个像素-------------------
            *data++ = (*idata++)&mask + div / 2;
            *data++ = (*idata++)&mask + div / 2;
            *data++ = (*idata++)&mask + div / 2;
            //-------------结束像素处理------------------------
        } //单行处理结束
    }
}

//--------------------------------------【方法十四】-------------------------------------------
//      说明：利用操作符重载
//-------------------------------------------------------------------------------------------------
void colorReduce13(Mat &image, int div = 64)
{
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //掩码值
    uchar mask = 0xFF << n; // e.g. 比如div=16, mask= 0xF0
    //进行色彩还原
    image = (image & Scalar(mask, mask, mask)) + Scalar(div / 2, div / 2, div / 2);
}




//-----------------------------------【ShowHelpText( )函数】-----------------------------
//      描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第24个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    printf("\n\n正在进行存取操作，请稍等……\n\n");
}




//-----------------------------------【main( )函数】--------------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始
//-------------------------------------------------------------------------------------------------
int main()
{
    int64 t[NTESTS], tinit;
    Mat image0;
    Mat image1;
    Mat image2;
    system("color 4F");
    ShowHelpText();
    image0 = imread("1.png");

    if (!image0.data)
        return 0;

    //时间值设为0
    for (int i = 0; i < NTESTS; i++)
        t[i] = 0;

    // 多次重复测试
    int n = NITERATIONS;

    for (int k = 0; k < n; k++) {
        cout << k << " of " << n << endl;
        image1 = imread("1.png");
        //【方法一】利用.ptr 和 []
        tinit = getTickCount();
        colorReduce0(image1);
        t[0] += getTickCount() - tinit;
        //【方法二】利用 .ptr 和 * ++
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce1(image1);
        t[1] += getTickCount() - tinit;
        //【方法三】利用.ptr 和 * ++ 以及模操作
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce2(image1);
        t[2] += getTickCount() - tinit;
        //【方法四】 利用.ptr 和 * ++ 以及位操作
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce3(image1);
        t[3] += getTickCount() - tinit;
        //【方法五】 利用指针的算术运算
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce4(image1);
        t[4] += getTickCount() - tinit;
        //【方法六】利用 .ptr 和 * ++以及位运算、image.cols * image.channels()
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce5(image1);
        t[5] += getTickCount() - tinit;
        //【方法七】利用.ptr 和 * ++ 以及位运算(continuous)
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce6(image1);
        t[6] += getTickCount() - tinit;
        //【方法八】利用 .ptr 和 * ++ 以及位运算 (continuous+channels)
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce7(image1);
        t[7] += getTickCount() - tinit;
        //【方法九】 利用Mat_ iterator
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce8(image1);
        t[8] += getTickCount() - tinit;
        //【方法十】 利用Mat_ iterator以及位运算
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce9(image1);
        t[9] += getTickCount() - tinit;
        //【方法十一】利用Mat Iterator_
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce10(image1);
        t[10] += getTickCount() - tinit;
        //【方法十二】 利用动态地址计算配合at
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce11(image1);
        t[11] += getTickCount() - tinit;
        //【方法十三】 利用图像的输入与输出
        image1 = imread("1.png");
        tinit = getTickCount();
        Mat result;
        colorReduce12(image1, result);
        t[12] += getTickCount() - tinit;
        image2 = result;
        //【方法十四】 利用操作符重载
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce13(image1);
        t[13] += getTickCount() - tinit;
        //------------------------------
    }

    //输出图像
    imshow("原始图像", image0);
    imshow("结果", image2);
    imshow("图像结果", image1);
    // 输出平均执行时间
    cout << endl << "-------------------------------------------" << endl << endl;
    cout << "\n【方法一】利用.ptr 和 []的方法所用时间为 " << 1000.*t[0] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法二】利用 .ptr 和 * ++ 的方法所用时间为" << 1000.*t[1] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法三】利用.ptr 和 * ++ 以及模操作的方法所用时间为" << 1000.*t[2] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法四】利用.ptr 和 * ++ 以及位操作的方法所用时间为" << 1000.*t[3] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法五】利用指针算术运算的方法所用时间为" << 1000.*t[4] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法六】利用 .ptr 和 * ++以及位运算、channels()的方法所用时间为" << 1000.*t[5] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法七】利用.ptr 和 * ++ 以及位运算(continuous)的方法所用时间为" << 1000.*t[6] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法八】利用 .ptr 和 * ++ 以及位运算 (continuous+channels)的方法所用时间为" << 1000.*t[7] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法九】利用Mat_ iterator 的方法所用时间为" << 1000.*t[8] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法十】利用Mat_ iterator以及位运算的方法所用时间为" << 1000.*t[9] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法十一】利用Mat Iterator_的方法所用时间为" << 1000.*t[10] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法十二】利用动态地址计算配合at 的方法所用时间为" << 1000.*t[11] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法十三】利用图像的输入与输出的方法所用时间为" << 1000.*t[12] / getTickFrequency() / n << "ms" << endl;
    cout << "\n【方法十四】利用操作符重载的方法所用时间为" << 1000.*t[13] / getTickFrequency() / n << "ms" << endl;
    waitKey();
    return 0;
}



#endif


#if SNIPPET019

#include "opencv2/opencv.hpp"
#include <time.h>
using namespace cv;


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//       描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第29个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//-----------------------------------【main( )函数】--------------------------------------------
//  描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
    //改变console字体颜色
    system("color 5F");
    ShowHelpText();
    //初始化
    FileStorage fs("test.yaml", FileStorage::WRITE);
    //开始文件写入
    fs << "frameCount" << 5;
    time_t rawtime;
    time(&rawtime);
    fs << "calibrationDate" << asctime(localtime(&rawtime));
    Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
    Mat distCoeffs = (Mat_<double>(5, 1) << 0.1, 0.01, -0.001, 0, 0);
    fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
    fs << "features" << "[";

    for (int i = 0; i < 3; i++) {
        int x = rand() % 640;
        int y = rand() % 480;
        uchar lbp = rand() % 256;
        fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";

        for (int j = 0; j < 8; j++)
            fs << ((lbp >> j) & 1);

        fs << "]" << "}";
    }

    fs << "]";
    fs.release();
    printf("\n文件读写完毕，请在工程目录下查看生成的文件~");
    getchar();
    return 0;
}


#endif


#if SNIPPET020

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


int main()
{
    // 原图
    Mat img = imread("020.jpg");
    // 方框滤波
    Mat box_filter_img;
    boxFilter(img, box_filter_img, -1, Size(5, 5));
    // 均值滤波
    Mat mean_blur_img;
    blur(img, mean_blur_img, Size(7, 7));
    // 高斯滤波
    Mat gaussian_blur_img;
    GaussianBlur(img, gaussian_blur_img, Size(5, 5), 0, 0);
    // 中值滤波
    Mat media_blur_img;
    medianBlur(img, media_blur_img, 7);
    // 双边滤波
    Mat bilateral_filter_img;
    bilateralFilter(img, bilateral_filter_img, 25, 25 * 2, 25 / 2);
    imshow("img", img);
    imshow("box_filter", box_filter_img);
    imshow("mean_blur", mean_blur_img);
    imshow("gaussian_blur", gaussian_blur_img);
    imshow("media_blur", media_blur_img);
    imshow("bilateral_filter", bilateral_filter_img);
    waitKey(0);
}

#endif


#if SNIPPET021

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】--------------------------------------
//  描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage1, g_dstImage2, g_dstImage3;//存储图片的Mat类型
int g_nBoxFilterValue = 3;  //方框滤波参数值
int g_nMeanBlurValue = 3;  //均值滤波参数值
int g_nGaussianBlurValue = 3;  //高斯滤波参数值


//-----------------------------------【全局函数声明部分】--------------------------------------
//  描述：全局函数声明
//-----------------------------------------------------------------------------------------------
//四个轨迹条的回调函数
static void on_BoxFilter(int, void *);      //均值滤波
static void on_MeanBlur(int, void *);       //均值滤波
static void on_GaussianBlur(int, void *);           //高斯滤波
void ShowHelpText();


//-----------------------------------【main( )函数】--------------------------------------------
//  描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
    //改变console字体颜色
    system("color 5F");
    //输出帮助文字
    ShowHelpText();
    // 载入原图
    g_srcImage = imread("020.jpg", 1);

    if (!g_srcImage.data) {
        printf("Oh，no，读取srcImage错误~！ \n");
        return false;
    }

    //克隆原图到三个Mat类型中
    g_dstImage1 = g_srcImage.clone();
    g_dstImage2 = g_srcImage.clone();
    g_dstImage3 = g_srcImage.clone();
    //显示原图
    namedWindow("【<0>原图窗口】", 1);
    imshow("【<0>原图窗口】", g_srcImage);
    //=================【<1>方框滤波】==================
    //创建窗口
    namedWindow("【<1>方框滤波】", 1);
    //创建轨迹条
    createTrackbar("内核值：", "【<1>方框滤波】", &g_nBoxFilterValue, 40, on_BoxFilter);
    on_BoxFilter(g_nBoxFilterValue, 0);
    //================================================
    //=================【<2>均值滤波】==================
    //创建窗口
    namedWindow("【<2>均值滤波】", 1);
    //创建轨迹条
    createTrackbar("内核值：", "【<2>均值滤波】", &g_nMeanBlurValue, 40, on_MeanBlur);
    on_MeanBlur(g_nMeanBlurValue, 0);
    //================================================
    //=================【<3>高斯滤波】=====================
    //创建窗口
    namedWindow("【<3>高斯滤波】", 1);
    //创建轨迹条
    createTrackbar("内核值：", "【<3>高斯滤波】", &g_nGaussianBlurValue, 40, on_GaussianBlur);
    on_GaussianBlur(g_nGaussianBlurValue, 0);
    //================================================
    //输出一些帮助信息
    cout << endl << "\t运行成功，请调整滚动条观察图像效果~\n\n"
         << "\t按下“q”键时，程序退出。\n";

    //按下“q”键时，程序退出
    while (char(waitKey(1)) != 'q') {}

    return 0;
}


//-----------------------------【on_BoxFilter( )函数】------------------------------------
//  描述：方框滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_BoxFilter(int, void *)
{
    //方框滤波操作
    boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
    //显示窗口
    imshow("【<1>方框滤波】", g_dstImage1);
}


//-----------------------------【on_MeanBlur( )函数】------------------------------------
//  描述：均值滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_MeanBlur(int, void *)
{
    //均值滤波操作
    blur(g_srcImage, g_dstImage2, Size(g_nMeanBlurValue + 1, g_nMeanBlurValue + 1), Point(-1, -1));
    //显示窗口
    imshow("【<2>均值滤波】", g_dstImage2);
}


//-----------------------------【ContrastAndBright( )函数】------------------------------------
//  描述：高斯滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_GaussianBlur(int, void *)
{
    //高斯滤波操作
    GaussianBlur(g_srcImage, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
    //显示窗口
    imshow("【<3>高斯滤波】", g_dstImage3);
}


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//       描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第34个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


#endif


#if SNIPPET022

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>


using namespace std;
using namespace cv;


int main()
{
    Mat img = imread("022.jpg");
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat dilate_img;
    dilate(img, dilate_img, element);
    Mat ercode_img;
    erode(img, ercode_img, element);
    Mat dilate_img2;
    morphologyEx(img, dilate_img2, MORPH_DILATE, element);
    Mat ercode_img2;
    morphologyEx(img, ercode_img2, MORPH_ERODE, element);
    Mat open_img;
    morphologyEx(img, open_img, MORPH_OPEN, element);
    Mat close_img;
    morphologyEx(img, close_img, MORPH_CLOSE, element);
    Mat gradient_img;
    morphologyEx(img, gradient_img, MORPH_GRADIENT, element);
    Mat tophat_img;
    morphologyEx(img, tophat_img, MORPH_TOPHAT, element);
    Mat blackhat_img;
    morphologyEx(img, blackhat_img, MORPH_BLACKHAT, element);
    Mat tmpImage = img;
    Mat resize_img1, resize_img2;
    //进行尺寸调整操作
    resize(tmpImage, resize_img1, Size(tmpImage.cols / 2, tmpImage.rows / 2), (0, 0), (0, 0), 3);
    resize(tmpImage, resize_img2, Size(tmpImage.cols * 2, tmpImage.rows * 2), (0, 0), (0, 0), 3);
    Mat pyrup_img;
    pyrUp(tmpImage, pyrup_img, Size(tmpImage.cols * 2, tmpImage.rows * 2));
    Mat pyrdown_img;
    pyrDown(tmpImage, pyrdown_img, Size(tmpImage.cols / 2, tmpImage.rows / 2));
    imshow("img", img);
    imshow("dilate_img", dilate_img);
    imshow("ercode_img", ercode_img);
    imshow("dilate_img2", dilate_img2);
    imshow("ercode_img2", ercode_img2);
    imshow("open_img", open_img);
    imshow("close_img", close_img);
    imshow("gradient_img", gradient_img);
    imshow("tophat_img", tophat_img);
    imshow("blackhat_img", blackhat_img);
    imshow("resize_img1", resize_img1);
    imshow("resize_img2", resize_img2);
    imshow("pyrup_img", pyrup_img);
    imshow("pyrdown_img", pyrdown_img);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET023

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    // 【1】载入原始图，且必须以二值图模式载入
    Mat srcImage = imread("023.jpg", 0);
    imshow("原始图", srcImage);
    //【2】初始化结果图
    Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
    //【3】srcImage取大于阈值119的那部分
    srcImage = srcImage > 119;
    imshow("取阈值后的原始图", srcImage);
    //【4】定义轮廓和层次结构
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //【5】查找轮廓
    //此句代码的OpenCV2版为：
    //findContours( srcImage, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    //此句代码的OpenCV3版为：
    findContours(srcImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    // 【6】遍历所有顶层的轮廓， 以随机颜色绘制出每个连接组件颜色
    int index = 0;

    for (; index >= 0; index = hierarchy[index][0]) {
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        //此句代码的OpenCV2版为：
        //drawContours( dstImage, contours, index, color, CV_FILLED, 8, hierarchy );
        //此句代码的OpenCV3版为：
        drawContours(dstImage, contours, index, color, FILLED, 8, hierarchy);
    }

    //【7】显示最后的轮廓图
    imshow("轮廓图", dstImage);
    waitKey(0);
}

#endif


#if SNIPPET024

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】--------------------------------------------
//      描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "【原始图窗口】"           //为窗口标题定义的宏 
#define WINDOW_NAME2 "【轮廓图】"                   //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//      描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage;
Mat g_grayImage;
int g_nThresh = 80;
int g_nThresh_max = 255;
RNG g_rng(12345);
Mat g_cannyMat_output;
vector<vector<Point>> g_vContours;
vector<Vec4i> g_vHierarchy;


//-----------------------------------【全局函数声明部分】--------------------------------------
//      描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText();
void on_ThreshChange(int, void*);


//-----------------------------------【main( )函数】--------------------------------------------
//      描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //【0】改变console字体颜色
    system("color 1F");
    //【0】显示欢迎和帮助文字
    ShowHelpText();
    // 加载源图像
    g_srcImage = imread("024.jpg", 1);

    if (!g_srcImage.data) {
        printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n");
        return false;
    }

    // 转成灰度并模糊化降噪
    cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
    blur(g_grayImage, g_grayImage, Size(3, 3));
    // 创建窗口
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    imshow(WINDOW_NAME1, g_srcImage);
    //创建滚动条并初始化
    createTrackbar("canny阈值", WINDOW_NAME1, &g_nThresh, g_nThresh_max, on_ThreshChange);
    on_ThreshChange(0, 0);
    waitKey(0);
    return (0);
}

//-----------------------------------【on_ThreshChange( )函数】------------------------------
//      描述：回调函数
//----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*)
{
    // 用Canny算子检测边缘
    Canny(g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh * 2, 3);
    // 寻找轮廓
    findContours(g_cannyMat_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // 绘出轮廓
    Mat drawing = Mat::zeros(g_cannyMat_output.size(), CV_8UC3);

    for (int i = 0; i < g_vContours.size(); i++) {
        Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//任意值
        drawContours(drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point());
    }

    // 显示效果图
    imshow(WINDOW_NAME2, drawing);
}


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//      描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第70个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\n\t欢迎来到【在图形中寻找轮廓】示例程序~\n\n");
    printf("\n\n\t按键操作说明: \n\n"
           "\t\t键盘按键任意键- 退出程序\n\n"
           "\t\t滑动滚动条-改变阈值\n");
}

#endif


#if SNIPPET025

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>


#define BLACK_RANGE1 cv::Scalar(0, 0, 0)
#define BLACK_RANGE2 cv::Scalar(180, 255, 46)


#define GRAY_RANGE1 cv::Scalar(0, 0, 46)
#define GRAY_RANGE2 cv::Scalar(180, 43, 220)


#define WHITE_RANGE1 cv::Scalar(0, 0, 221)
#define WHITE_RANGE2 cv::Scalar(180, 30, 255)

#define RED_RANGE1 cv::Scalar(0, 43, 46)
#define RED_RANGE2 cv::Scalar(10, 255, 255)
#define RED_RANGE3 cv::Scalar(156, 43, 46)
#define RED_RANGE4 cv::Scalar(180, 255, 255)


#define ORANGE_RANGE1 cv::Scalar(11, 43, 46)
#define ORANGE_RANGE2 cv::Scalar(25, 255, 255)

#define YELLOW_RANGE1 cv::Scalar(26, 43, 46)
#define YELLOW_RANGE2 cv::Scalar(34, 255, 255)


#define GREEN_RANGE1 cv::Scalar(35, 43, 46)
#define GREEN_RANGE2 cv::Scalar(77, 255, 255)

#define CYAN_RANGE1 cv::Scalar(78, 43, 46)
#define CYAN_RANGE2 cv::Scalar(99, 255, 255)


#define BLUE_RANGE1 cv::Scalar(100, 43, 46)
#define BLUE_RANGE2 cv::Scalar(124, 255, 255)

#define VIOLET_RANGE1 cv::Scalar(125, 43, 46)
#define VIOLET_RANGE2 cv::Scalar(155, 255, 255)



using namespace std;
using namespace cv;

void check_red_range(const Mat& in, Mat& out)
{
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);
    cv::Mat lower_red_hue_range;
    cv::Mat upper_red_hue_range;
    // in hsv red is splitted in two areals. This function checks both of them...
    cv::inRange(hsv_image, RED_RANGE1, RED_RANGE2, lower_red_hue_range);
    cv::inRange(hsv_image, RED_RANGE3, RED_RANGE4, upper_red_hue_range);
    //...and merges them.
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, out);
}


void check_orange_range(const Mat& in, Mat& out)
{
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, ORANGE_RANGE1, ORANGE_RANGE2, out);
}



void check_yellow_range(const Mat& in, Mat& out)
{
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, YELLOW_RANGE1, YELLOW_RANGE2, out);
}


void check_green_range(const Mat& in, Mat& out)
{
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, GREEN_RANGE1, GREEN_RANGE2, out);
}

void check_cyan_range(const Mat& in, Mat& out)
{
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, CYAN_RANGE1, CYAN_RANGE2, out);
}

void check_blue_range(const Mat& in, Mat& out)
{
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, BLUE_RANGE1, BLUE_RANGE2, out);
}

void check_violet_range(const Mat& in, Mat& out)
{
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image, VIOLET_RANGE1, VIOLET_RANGE2, out);
}



int main()
{
    Mat img = imread("025.jpg", IMREAD_COLOR);
    Mat bilateral_filter_img;
    bilateralFilter(img, bilateral_filter_img, 13, 100, 50);
    Mat red_out_img;
    check_red_range(bilateral_filter_img, red_out_img);
    Mat orange_out_img;
    check_orange_range(bilateral_filter_img, orange_out_img);
    Mat yellow_out_img;
    check_yellow_range(bilateral_filter_img, yellow_out_img);
    Mat green_out_img;
    check_green_range(bilateral_filter_img, green_out_img);
    Mat cyan_out_img;
    check_cyan_range(bilateral_filter_img, cyan_out_img);
    Mat blue_out_img;
    check_blue_range(bilateral_filter_img, blue_out_img);
    Mat violet_out_img;
    check_violet_range(bilateral_filter_img, violet_out_img);
    imshow("img", img);
    imshow("bilateral_filter_img", bilateral_filter_img);
    imshow("red_out_img", red_out_img);
    imshow("orange_out_img", orange_out_img);
    imshow("yellow_out_img", yellow_out_img);
    imshow("green_out_img", green_out_img);
    imshow("cyan_out_img", cyan_out_img);
    imshow("blue_out_img", blue_out_img);
    imshow("violet_out_img", violet_out_img);
    waitKey(0);
    return 0;
}


#endif


#if SNIPPET026

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第71个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\t欢迎来到【凸包检测】示例程序~\n\n");
    printf("\n\t按键操作说明: \n\n"
           "\t\t键盘按键【ESC】、【Q】、【q】- 退出程序\n\n"
           "\t\t键盘按键任意键 - 重新生成随机点，并进行凸包检测\n");
}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    //改变console字体颜色
    system("color 1F");
    //显示帮助文字
    ShowHelpText();
    //初始化变量和随机值
    Mat image(600, 600, CV_8UC3);
    RNG& rng = theRNG();

    //循环，按下ESC,Q,q键程序退出，否则有键按下便一直更新
    while (1) {
        //参数初始化
        char key;//键值
        int count = (unsigned)rng % 100 + 3;//随机生成点的数量
        vector<Point> points; //点值

        //随机生成点坐标
        for (int i = 0; i < count; i++) {
            Point point;
            point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
            point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
            points.push_back(point);
        }

        //检测凸包
        vector<int> hull;
        convexHull(Mat(points), hull, true);
        //绘制出随机颜色的点
        image = Scalar::all(0);

        for (int i = 0; i < count; i++)
            circle(image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA);

        //准备参数
        int hullcount = (int)hull.size();//凸包的边数
        Point point0 = points[hull[hullcount - 1]];//连接凸包边的坐标点

        //绘制凸包的边
        for (int i = 0; i < hullcount; i++) {
            Point point = points[hull[i]];
            line(image, point0, point, Scalar(255, 255, 255), 2, LINE_AA);
            point0 = point;
        }

        //显示效果图
        imshow("凸包检测示例", image);
        //按下ESC,Q,或者q，程序退出
        key = (char)waitKey();

        if (key == 27 || key == 'q' || key == 'Q')
            break;
    }

    return 0;
}



#endif


#if SNIPPET027

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】--------------------------------------------
//  描述：定义一些辅助宏
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "【原始图窗口】"                   //为窗口标题定义的宏 
#define WINDOW_NAME2 "【效果图窗口】"                   //为窗口标题定义的宏 



//-----------------------------------【全局变量声明部分】--------------------------------------
//  描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage;
Mat g_grayImage;
int g_nThresh = 50;
int g_maxThresh = 255;
RNG g_rng(12345);
Mat srcImage_copy = g_srcImage.clone();
Mat g_thresholdImage_output;
vector<vector<Point> > g_vContours;
vector<Vec4i> g_vHierarchy;


//-----------------------------------【全局函数声明部分】--------------------------------------
//   描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText();
void on_ThreshChange(int, void*);
void ShowHelpText();

//-----------------------------------【main( )函数】------------------------------------------
//   描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    system("color 3F");
    ShowHelpText();
    // 加载源图像
    g_srcImage = imread("027.jpg", 1);
    // 将原图转换成灰度图并进行模糊降
    cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
    blur(g_grayImage, g_grayImage, Size(3, 3));
    // 创建原图窗口并显示
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    imshow(WINDOW_NAME1, g_srcImage);
    //创建滚动条
    createTrackbar(" 阈值:", WINDOW_NAME1, &g_nThresh, g_maxThresh, on_ThreshChange);
    on_ThreshChange(0, 0);//调用一次进行初始化
    waitKey(0);
    return (0);
}

//-----------------------------------【thresh_callback( )函数】----------------------------------
//      描述：回调函数
//----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*)
{
    // 对图像进行二值化，控制阈值
    threshold(g_grayImage, g_thresholdImage_output, g_nThresh, 255, THRESH_BINARY);
    // 寻找轮廓
    findContours(g_thresholdImage_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // 遍历每个轮廓，寻找其凸包
    vector<vector<Point> >hull(g_vContours.size());

    for (unsigned int i = 0; i < g_vContours.size(); i++) {
        convexHull(Mat(g_vContours[i]), hull[i], false);
    }

    // 绘出轮廓及其凸包
    Mat drawing = Mat::zeros(g_thresholdImage_output.size(), CV_8UC3);

    for (unsigned int i = 0; i < g_vContours.size(); i++) {
        Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
        drawContours(drawing, g_vContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
    }

    // 显示效果图
    imshow(WINDOW_NAME2, drawing);
}


//-----------------------------------【ShowHelpText( )函数】-----------------------------
//       描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第72个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}

#endif


#if SNIPPET028

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;



//-----------------------------------【ShowHelpText( )函数】-----------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第73个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\n\n\t\t\t欢迎来到【矩形包围示例】示例程序~\n\n");
    printf("\n\n\t按键操作说明: \n\n"
           "\t\t键盘按键【ESC】、【Q】、【q】- 退出程序\n\n"
           "\t\t键盘按键任意键 - 重新生成随机点，并寻找最小面积的包围矩形\n");
}

int main()
{
    //改变console字体颜色
    system("color 1F");
    //显示帮助文字
    ShowHelpText();
    //初始化变量和随机值
    Mat image(600, 600, CV_8UC3);
    RNG& rng = theRNG();

    //循环，按下ESC,Q,q键程序退出，否则有键按下便一直更新
    while (1) {
        //参数初始化
        int count = rng.uniform(3, 103);//随机生成点的数量
        vector<Point> points;//点值

        //随机生成点坐标
        for (int i = 0; i < count; i++) {
            Point point;
            point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
            point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
            points.push_back(point);
        }

        //对给定的 2D 点集，寻找最小面积的包围矩形
        RotatedRect box = minAreaRect(Mat(points));
        Point2f vertex[4];
        box.points(vertex);
        //绘制出随机颜色的点
        image = Scalar::all(0);

        for (int i = 0; i < count; i++)
            circle(image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA);

        //绘制出最小面积的包围矩形
        for (int i = 0; i < 4; i++)
            line(image, vertex[i], vertex[(i + 1) % 4], Scalar(100, 200, 211), 2, LINE_AA);

        //显示窗口
        imshow("矩形包围示例", image);
        //按下ESC,Q,或者q，程序退出
        char key = (char)waitKey();

        if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

    return 0;
}


#endif


#if SNIPPET029

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第13个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\n\t\t\t欢迎来到【寻找最小面积的包围圆】示例程序~\n");
    printf("\n\n\t按键操作说明: \n\n"
           "\t\t键盘按键【ESC】、【Q】、【q】- 退出程序\n\n"
           "\t\t键盘按键任意键 - 重新生成随机点，并寻找最小面积的包围圆\n");
}

int main()
{
    //改变console字体颜色
    system("color 1F");
    //显示帮助文字
    ShowHelpText();
    //初始化变量和随机值
    Mat image(600, 600, CV_8UC3);
    RNG& rng = theRNG();

    //循环，按下ESC,Q,q键程序退出，否则有键按下便一直更新
    while (1) {
        //参数初始化
        int count = rng.uniform(3, 103);//随机生成点的数量
        vector<Point> points;//点值

        //随机生成点坐标
        for (int i = 0; i < count; i++) {
            Point point;
            point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
            point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
            points.push_back(point);
        }

        //对给定的 2D 点集，寻找最小面积的包围圆
        Point2f center;
        float radius = 0;
        minEnclosingCircle(Mat(points), center, radius);
        //绘制出随机颜色的点
        image = Scalar::all(0);

        for (int i = 0; i < count; i++)
            circle(image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA);

        //绘制出最小面积的包围圆
        circle(image, center, cvRound(radius), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2, LINE_AA);
        //显示窗口
        imshow("圆形包围示例", image);
        //按下ESC,Q,或者q，程序退出
        char key = (char)waitKey();

        if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

    return 0;
}


#endif


#if SNIPPET030

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/* @brief 得到H分量的直方图图像
@param src 输入图像
@param histimg 输出颜色直方图
@return void 返回值为空
*/
void getHistImg(const Mat src, Mat &histimg)
{
    Mat hue, hist;
    int hsize = 16;//直方图bin的个数
    float hranges[] = { 0, 180 };
    const float* phranges = hranges;
    int ch[] = { 0, 0 };
    hue.create(src.size(), src.depth());
    mixChannels(&src, 1, &hue, 1, ch, 1);//得到H分量
    calcHist(&hue, 1, 0, Mat(), hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, NORM_MINMAX);
    histimg = Scalar::all(0);
    int binW = histimg.cols / hsize;
    Mat buf(1, hsize, CV_8UC3);

    for (int i = 0; i < hsize; i++)
        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);

    cvtColor(buf, buf, COLOR_HSV2BGR);

    for (int i = 0; i < hsize; i++) {
        int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
        rectangle(histimg, Point(i * binW, histimg.rows),
                  Point((i + 1)*binW, histimg.rows - val),
                  Scalar(buf.at<Vec3b>(i)), -1, 8);
    }
}

int main(void)
{
    Mat src, histimg = Mat::zeros(540, 540, CV_8UC3);
    // 载入图片
    src = imread("001.jpg");

    if (!src.data) {
        cout << "load image failed" << endl;
        return -1;
    }

    // 调用
    getHistImg(src, histimg);
    imshow("histImage", histimg);
    imshow("srcImage", src);
    waitKey(0);
    return 0;
}


#endif


#if SNIPPET031

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;


int main()
{
    Mat Image = imread("001.jpg");
    cvtColor(Image, Image, COLOR_BGR2GRAY);
    const int channels[1] = { 0 };
    const int histSize[1] = { 256 };
    float hranges[2] = { 0, 255 };
    const float* ranges[1] = { hranges };
    MatND hist;
    calcHist(&Image, 1, channels, Mat(), hist, 1, histSize, ranges);
    imshow("img", Image);
    imshow("hist", hist);
    waitKey(0);
    return 0;
}



#endif


#if SNIPPET032

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Windows.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

Mat hwnd2mat(HWND hwnd)
{
    HDC hwindowDC, hwindowCompatibleDC;
    int height, width, srcheight, srcwidth;
    HBITMAP hbwindow;
    Mat src;
    BITMAPINFOHEADER  bi;
    hwindowDC = GetDC(hwnd);
    hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);
    RECT windowsize;    // get the height and width of the screen
    GetClientRect(hwnd, &windowsize);
    srcheight = windowsize.bottom;
    srcwidth = windowsize.right;
    height = windowsize.bottom / 2;  //change this to whatever size you want to resize to
    width = windowsize.right / 2;
    src.create(height, width, CV_8UC4);
    // create a bitmap
    hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
    bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
    bi.biWidth = width;
    bi.biHeight = -height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;
    // use the previously created device context with the bitmap
    SelectObject(hwindowCompatibleDC, hbwindow);
    // copy from the window device context to the bitmap device context
    StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow
    // avoid memory leak
    DeleteObject(hbwindow);
    DeleteDC(hwindowCompatibleDC);
    ReleaseDC(hwnd, hwindowDC);
    return src;
}

int main()
{
    Mat img = hwnd2mat(::GetDesktopWindow());
    imshow("img", img);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET033

/*! 
 * \brief 位图转 Mat
 * 
 * 位图转 Mat
 * 
 */


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Windows.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main()

{
    int x_size = 800, y_size = 600; // <-- Your res for the image
    HBITMAP hBitmap; // <-- The image represented by hBitmap
    Mat matBitmap; // <-- The image represented by mat
    // Initialize DCs
    HDC hdcSys = GetDC(NULL); // Get DC of the target capture..
    HDC hdcMem = CreateCompatibleDC(hdcSys); // Create compatible DC
    void *ptrBitmapPixels; // <-- Pointer variable that will contain the potinter for the pixels
    // Create hBitmap with Pointer to the pixels of the Bitmap
    BITMAPINFO bi;
    HDC hdc;
    ZeroMemory(&bi, sizeof(BITMAPINFO));
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = x_size;
    bi.bmiHeader.biHeight = -y_size;  //negative so (0,0) is at top left
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    hdc = GetDC(NULL);
    hBitmap = CreateDIBSection(hdc, &bi, DIB_RGB_COLORS, &ptrBitmapPixels, NULL, 0);
    // ^^ The output: hBitmap & ptrBitmapPixels
    // Set hBitmap in the hdcMem
    SelectObject(hdcMem, hBitmap);
    // Set matBitmap to point to the pixels of the hBitmap
    matBitmap = Mat(y_size, x_size, CV_8UC4, ptrBitmapPixels, 0);
    //                ^^ note: first it is y, then it is x. very confusing
    // * SETUP DONE *
    // Now update the pixels using BitBlt
    BitBlt(hdcMem, 0, 0, x_size, y_size, hdcSys, 0, 0, SRCCOPY);
    // Just to do some image processing on the pixels.. (Dont have to to this)
    Mat matRef = matBitmap(Range(100, 200), Range(100, 200));
    //                              y1    y2            x1     x2
    bitwise_not(matRef, matRef); // Invert the colors in this x1,x2,y1,y2
    // Display the results through Mat
    imshow("Title", matBitmap);
    // Wait until some key is pressed
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET034

/*! 
 * \brief H-S二维直方图的绘制 
 * 
 * H-S二维直方图的绘制
 * 
 */


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;



//-----------------------------------【ShowHelpText( )函数】-----------------------------
//		 描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第79个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{

    //【1】载入源图，转化为HSV颜色模型
    Mat srcImage, hsvImage;
    srcImage = imread("034.jpg");
    cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);

    system("color 2F");
    ShowHelpText();

    //【2】参数准备
    //将色调量化为30个等级，将饱和度量化为32个等级
    int hueBinNum = 30;//色调的直方图直条数量
    int saturationBinNum = 32;//饱和度的直方图直条数量
    int histSize[] = { hueBinNum, saturationBinNum };
    // 定义色调的变化范围为0到179
    float hueRanges[] = { 0, 180 };
    //定义饱和度的变化范围为0（黑、白、灰）到255（纯光谱颜色）
    float saturationRanges[] = { 0, 256 };
    const float* ranges[] = { hueRanges, saturationRanges };
    MatND dstHist;
    //参数准备，calcHist函数中将计算第0通道和第1通道的直方图
    int channels[] = { 0, 1 };

    //【3】正式调用calcHist，进行直方图计算
    calcHist(&hsvImage,//输入的数组
        1, //数组个数为1
        channels,//通道索引
        Mat(), //不使用掩膜
        dstHist, //输出的目标直方图
        2, //需要计算的直方图的维度为2
        histSize, //存放每个维度的直方图尺寸的数组
        ranges,//每一维数值的取值范围数组
        true, // 指示直方图是否均匀的标识符，true表示均匀的直方图
        false);//累计标识符，false表示直方图在配置阶段会被清零

    //【4】为绘制直方图准备参数
    double maxValue = 0;//最大值
    minMaxLoc(dstHist, 0, &maxValue, 0, 0);//查找数组和子数组的全局最小值和最大值存入maxValue中
    int scale = 10;
    Mat histImg = Mat::zeros(saturationBinNum*scale, hueBinNum * 10, CV_8UC3);

    //【5】双层循环，进行直方图绘制
    for (int hue = 0; hue < hueBinNum; hue++)
        for (int saturation = 0; saturation < saturationBinNum; saturation++)
        {
            float binValue = dstHist.at<float>(hue, saturation);//直方图组距的值
            int intensity = cvRound(binValue * 255 / maxValue);//强度

            //正式进行绘制
            rectangle(histImg, Point(hue*scale, saturation*scale),
                Point((hue + 1)*scale - 1, (saturation + 1)*scale - 1),
                Scalar::all(intensity), FILLED);
        }

    //【6】显示效果图
    imshow("素材图", srcImage);
    imshow("H-S 直方图", histImg);

    waitKey();
}

#endif


#if SNIPPET035

/*! 
 * \brief 一维直方图的绘制
 * 
 * 一维直方图的绘制
 * 
 */


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;



//-----------------------------------【ShowHelpText( )函数】-----------------------------
//		 描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第80个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-------------------------------------------------------------------------------------------------
int main()
{
    //【1】载入原图并显示
    Mat srcImage = imread("035.jpg", 0);
    imshow("原图", srcImage);
    if (!srcImage.data) { cout << "fail to load image" << endl; 	return 0; }

    system("color 1F");
    ShowHelpText();

    //【2】定义变量
    MatND dstHist;       // 在cv中用CvHistogram *hist = cvCreateHist
    int dims = 1;
    float hranges[] = { 0, 255 };
    const float *ranges[] = { hranges };   // 这里需要为const类型
    int size = 256;
    int channels = 0;

    //【3】计算图像的直方图
    calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);    // cv 中是cvCalcHist
    int scale = 1;

    Mat dstImage(size * scale, size, CV_8U, Scalar(0));
    //【4】获取最大值和最小值
    double minValue = 0;
    double maxValue = 0;
    minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //  在cv中用的是cvGetMinMaxHistValue

    //【5】绘制出直方图
    int hpt = saturate_cast<int>(0.9 * size);
    for (int i = 0; i < 256; i++)
    {
        float binValue = dstHist.at<float>(i);           //   注意hist中是float类型    而在OpenCV1.0版中用cvQueryHistValue_1D
        int realValue = saturate_cast<int>(binValue * hpt / maxValue);
        rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
    }
    imshow("一维直方图", dstImage);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET036

/*! 
 * \brief RGB三色直方图的绘制 
 * 
 * RGB三色直方图的绘制 
 * 
 */


#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
using namespace cv;



//-----------------------------------【ShowHelpText( )函数】-----------------------------
//		 描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第81个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{

    //【1】载入素材图并显示
    Mat srcImage;
    srcImage = imread("036.jpg");
    imshow("素材图", srcImage);

    system("color 3F");
    ShowHelpText();

    //【2】参数准备
    int bins = 256;
    int hist_size[] = { bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    MatND redHist, grayHist, blueHist;
    int channels_r[] = { 0 };

    //【3】进行直方图的计算（红色分量部分）
    calcHist(&srcImage, 1, channels_r, Mat(), //不使用掩膜
        redHist, 1, hist_size, ranges,
        true, false);

    //【4】进行直方图的计算（绿色分量部分）
    int channels_g[] = { 1 };
    calcHist(&srcImage, 1, channels_g, Mat(), // do not use mask
        grayHist, 1, hist_size, ranges,
        true, // the histogram is uniform
        false);

    //【5】进行直方图的计算（蓝色分量部分）
    int channels_b[] = { 2 };
    calcHist(&srcImage, 1, channels_b, Mat(), // do not use mask
        blueHist, 1, hist_size, ranges,
        true, // the histogram is uniform
        false);

    //-----------------------绘制出三色直方图------------------------
    //参数准备
    double maxValue_red, maxValue_green, maxValue_blue;
    minMaxLoc(redHist, 0, &maxValue_red, 0, 0);
    minMaxLoc(grayHist, 0, &maxValue_green, 0, 0);
    minMaxLoc(blueHist, 0, &maxValue_blue, 0, 0);
    int scale = 1;
    int histHeight = 256;
    Mat histImage = Mat::zeros(histHeight, bins * 3, CV_8UC3);

    //正式开始绘制
    for (int i = 0; i < bins; i++)
    {
        //参数准备
        float binValue_red = redHist.at<float>(i);
        float binValue_green = grayHist.at<float>(i);
        float binValue_blue = blueHist.at<float>(i);
        int intensity_red = cvRound(binValue_red*histHeight / maxValue_red);  //要绘制的高度
        int intensity_green = cvRound(binValue_green*histHeight / maxValue_green);  //要绘制的高度
        int intensity_blue = cvRound(binValue_blue*histHeight / maxValue_blue);  //要绘制的高度

        //绘制红色分量的直方图
        rectangle(histImage, Point(i*scale, histHeight - 1),
            Point((i + 1)*scale - 1, histHeight - intensity_red),
            Scalar(255, 0, 0));

        //绘制绿色分量的直方图
        rectangle(histImage, Point((i + bins)*scale, histHeight - 1),
            Point((i + bins + 1)*scale - 1, histHeight - intensity_green),
            Scalar(0, 255, 0));

        //绘制蓝色分量的直方图
        rectangle(histImage, Point((i + bins * 2)*scale, histHeight - 1),
            Point((i + bins * 2 + 1)*scale - 1, histHeight - intensity_blue),
            Scalar(0, 0, 255));

    }

    //在窗口中显示出绘制好的直方图
    imshow("图像的RGB直方图", histImage);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET037

/*! 
 * \brief 直方图对比 
 * 
 * 直方图对比
 * 
 */


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


//-----------------------------------【ShowHelpText( )函数】-----------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第82个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\n欢迎来到【直方图对比】示例程序~\n\n");

}


//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    //【0】改变console字体颜色
    system("color 2F");

    //【1】显示帮助文字
    ShowHelpText();

    //【1】声明储存基准图像和另外两张对比图像的矩阵( RGB 和 HSV )
    Mat srcImage_base, hsvImage_base;
    Mat srcImage_test1, hsvImage_test1;
    Mat srcImage_test2, hsvImage_test2;
    Mat hsvImage_halfDown;

    //【2】载入基准图像(srcImage_base) 和两张测试图像srcImage_test1、srcImage_test2，并显示
    srcImage_base = imread("037_1.jpg", 1);
    srcImage_test1 = imread("037_2.jpg", 1);
    srcImage_test2 = imread("037_3.jpg", 1);
    //显示载入的3张图像
    imshow("基准图像", srcImage_base);
    imshow("测试图像1", srcImage_test1);
    imshow("测试图像2", srcImage_test2);

    // 【3】将图像由BGR色彩空间转换到 HSV色彩空间
    cvtColor(srcImage_base, hsvImage_base, COLOR_BGR2HSV);
    cvtColor(srcImage_test1, hsvImage_test1, COLOR_BGR2HSV);
    cvtColor(srcImage_test2, hsvImage_test2, COLOR_BGR2HSV);

    //【4】创建包含基准图像下半部的半身图像(HSV格式)
    hsvImage_halfDown = hsvImage_base(Range(hsvImage_base.rows / 2, hsvImage_base.rows - 1), Range(0, hsvImage_base.cols - 1));

    //【5】初始化计算直方图需要的实参
    // 对hue通道使用30个bin,对saturatoin通道使用32个bin
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue的取值范围从0到256, saturation取值范围从0到180
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 180 };
    const float* ranges[] = { h_ranges, s_ranges };
    // 使用第0和第1通道
    int channels[] = { 0, 1 };

    // 【6】创建储存直方图的 MatND 类的实例:
    MatND baseHist;
    MatND halfDownHist;
    MatND testHist1;
    MatND testHist2;

    // 【7】计算基准图像，两张测试图像，半身基准图像的HSV直方图:
    calcHist(&hsvImage_base, 1, channels, Mat(), baseHist, 2, histSize, ranges, true, false);
    normalize(baseHist, baseHist, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&hsvImage_halfDown, 1, channels, Mat(), halfDownHist, 2, histSize, ranges, true, false);
    normalize(halfDownHist, halfDownHist, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&hsvImage_test1, 1, channels, Mat(), testHist1, 2, histSize, ranges, true, false);
    normalize(testHist1, testHist1, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&hsvImage_test2, 1, channels, Mat(), testHist2, 2, histSize, ranges, true, false);
    normalize(testHist2, testHist2, 0, 1, NORM_MINMAX, -1, Mat());


    //【8】按顺序使用4种对比标准将基准图像的直方图与其余各直方图进行对比:
    for (int i = 0; i < 4; i++)
    {
        //进行图像直方图的对比
        int compare_method = i;
        double base_base = compareHist(baseHist, baseHist, compare_method);
        double base_half = compareHist(baseHist, halfDownHist, compare_method);
        double base_test1 = compareHist(baseHist, testHist1, compare_method);
        double base_test2 = compareHist(baseHist, testHist2, compare_method);
        //输出结果
        printf(" 方法 [%d] 的匹配结果如下：\n\n 【基准图 - 基准图】：%f, 【基准图 - 半身图】：%f,【基准图 - 测试图1】： %f, 【基准图 - 测试图2】：%f \n-----------------------------------------------------------------\n", i, base_base, base_half, base_test1, base_test2);
    }

    printf("检测结束。");
    waitKey(0);
    return 0;
}


#endif


#if SNIPPET038

/*! 
 * \brief 反向投影 
 * 
 * 反向投影
 * 
 */


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;


//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【原始图】"        //为窗口标题定义的宏 


//-----------------------------------【全局变量声明部分】--------------------------------------
//          描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_hsvImage; Mat g_hueImage;
int g_bins = 30;//直方图组距

//-----------------------------------【全局函数声明部分】--------------------------------------
//          描述：全局函数声明
//-----------------------------------------------------------------------------------------------
static void ShowHelpText();
void on_BinChange(int, void*);

//--------------------------------------【main( )函数】-----------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    //【0】改变console字体颜色
    system("color 6F");

    //【0】显示帮助文字
    ShowHelpText();

    //【1】读取源图像，并转换到 HSV 空间
    g_srcImage = imread("038.jpg", 1);
    if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~！ \n"); return false; }
    cvtColor(g_srcImage, g_hsvImage, COLOR_BGR2HSV);

    //【2】分离 Hue 色调通道
    g_hueImage.create(g_hsvImage.size(), g_hsvImage.depth());
    int ch[] = { 0, 0 };
    mixChannels(&g_hsvImage, 1, &g_hueImage, 1, ch, 1);

    //【3】创建 Trackbar 来输入bin的数目
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    createTrackbar("色调组距 ", WINDOW_NAME1, &g_bins, 180, on_BinChange);
    on_BinChange(0, 0);//进行一次初始化

    //【4】显示效果图
    imshow(WINDOW_NAME1, g_srcImage);

    // 等待用户按键
    waitKey(0);
    return 0;
}


//-----------------------------------【on_HoughLines( )函数】--------------------------------
//          描述：响应滑动条移动消息的回调函数
//---------------------------------------------------------------------------------------------
void on_BinChange(int, void*)
{
    //【1】参数准备
    MatND hist;
    int histSize = MAX(g_bins, 2);
    float hue_range[] = { 0, 180 };
    const float* ranges = { hue_range };

    //【2】计算直方图并归一化
    calcHist(&g_hueImage, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
    normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

    //【3】计算反向投影
    MatND backproj;
    calcBackProject(&g_hueImage, 1, 0, hist, backproj, &ranges, 1, true);

    //【4】显示反向投影
    imshow("反向投影图", backproj);

    //【5】绘制直方图的参数准备
    int w = 400; int h = 400;
    int bin_w = cvRound((double)w / histSize);
    Mat histImg = Mat::zeros(w, h, CV_8UC3);

    //【6】绘制直方图
    for (int i = 0; i < g_bins; i++)
    {
        rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)), Scalar(100, 123, 255), -1);
    }

    //【7】显示直方图窗口
    imshow("直方图", histImg);
}


//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第83个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");

    //输出一些帮助信息
    printf("\n\n\t欢迎来到【反向投影】示例程序\n\n");
    printf("\n\t请调整滑动条观察图像效果\n\n");

}

#endif


#if SNIPPET039

/*! 
 * \brief 模板匹配 
 * 
 * 模板匹配
 * 
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


//-----------------------------------【宏定义部分】-------------------------------------------- 
//  描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【原始图片】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【匹配窗口】"        //为窗口标题定义的宏 

//-----------------------------------【全局变量声明部分】------------------------------------
//          描述：全局变量的声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_templateImage; Mat g_resultImage;
int g_nMatchMethod;
int g_nMaxTrackbarNum = 5;

//-----------------------------------【全局函数声明部分】--------------------------------------
//          描述：全局函数的声明
//-----------------------------------------------------------------------------------------------
void on_Matching(int, void*);
static void ShowHelpText();


//-----------------------------------【main( )函数】--------------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
    //【0】改变console字体颜色
    system("color 1F");

    //【0】显示帮助文字
    ShowHelpText();

    //【1】载入原图像和模板块
    g_srcImage = imread("039_1.jpg", 1);
    g_templateImage = imread("039_2.jpg", 1);

    //【2】创建窗口
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    namedWindow(WINDOW_NAME2, WINDOW_AUTOSIZE);

    //【3】创建滑动条并进行一次初始化
    createTrackbar("方法", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum, on_Matching);
    on_Matching(0, 0);

    waitKey(0);
    return 0;

}

//-----------------------------------【on_Matching( )函数】--------------------------------
//          描述：回调函数
//-------------------------------------------------------------------------------------------
void on_Matching(int, void*)
{
    //【1】给局部变量初始化
    Mat srcImage;
    g_srcImage.copyTo(srcImage);

    //【2】初始化用于结果输出的矩阵
    int resultImage_rows = g_srcImage.rows - g_templateImage.rows + 1;
    int resultImage_cols = g_srcImage.cols - g_templateImage.cols + 1;
    g_resultImage.create(resultImage_rows, resultImage_cols, CV_32FC1);

    //【3】进行匹配和标准化
    matchTemplate(g_srcImage, g_templateImage, g_resultImage, g_nMatchMethod);
    normalize(g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat());

    //【4】通过函数 minMaxLoc 定位最匹配的位置
    double minValue; double maxValue; Point minLocation; Point maxLocation;
    Point matchLocation;
    minMaxLoc(g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat());

    //【5】对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值有着更高的匹配结果. 而其余的方法, 数值越大匹配效果越好
    //此句代码的OpenCV2版为：
    //if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
    //此句代码的OpenCV3版为：
    if (g_nMatchMethod == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED)
    {
        matchLocation = minLocation;
    }
    else
    {
        matchLocation = maxLocation;
    }

    //【6】绘制出矩形，并显示最终结果
    rectangle(srcImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);
    rectangle(g_resultImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);

    imshow(WINDOW_NAME1, srcImage);
    imshow(WINDOW_NAME2, g_resultImage);

}



//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第84个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\t欢迎来到【模板匹配】示例程序~\n");
    printf("\n\n\t请调整滑动条观察图像效果\n\n");
    printf("\n\t滑动条对应的方法数值说明: \n\n"
        "\t\t方法【0】- 平方差匹配法(SQDIFF)\n"
        "\t\t方法【1】- 归一化平方差匹配法(SQDIFF NORMED)\n"
        "\t\t方法【2】- 相关匹配法(TM CCORR)\n"
        "\t\t方法【3】- 归一化相关匹配法(TM CCORR NORMED)\n"
        "\t\t方法【4】- 相关系数匹配法(TM COEFF)\n"
        "\t\t方法【5】- 归一化相关系数匹配法(TM COEFF NORMED)\n");
}


#endif

#if SNIPPET040

/*! 
 * \brief cornerHarris 函数用法示例 
 * 
 * cornerHarris 函数用法示例 
 * 
 */


#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
using namespace cv;

int main()
{
    //以灰度模式载入图像并显示
    Mat srcImage = imread("040.jpg", 0);
    imshow("原始图", srcImage);

    //进行Harris角点检测找出角点
    Mat cornerStrength;
    cornerHarris(srcImage, cornerStrength, 2, 3, 0.01);

    //对灰度图进行阈值操作，得到二值图并显示  
    Mat harrisCorner;
    threshold(cornerStrength, harrisCorner, 0.00001, 255, THRESH_BINARY);
    imshow("角点检测后的二值效果图", harrisCorner);

    waitKey(0);
    return 0;
}

#endif


#if SNIPPET041

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;


//-----------------------------------【宏定义部分】--------------------------------------------  
//  描述：定义一些辅助宏  
//------------------------------------------------------------------------------------------------  
#define WINDOW_NAME1 "【程序窗口1】"        //为窗口标题定义的宏  
#define WINDOW_NAME2 "【程序窗口2】"        //为窗口标题定义的宏  

//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_srcImage1, g_grayImage;
int thresh = 30; //当前阈值
int max_thresh = 175; //最大阈值


//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void on_CornerHarris(int, void*);//回调函数
static void ShowHelpText();

//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //【0】改变console字体颜色
    system("color 3F");

    //【0】显示帮助文字
    ShowHelpText();

    //【1】载入原始图并进行克隆保存
    g_srcImage = imread("041.jpg", 1);
    if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
    imshow("原始图", g_srcImage);
    g_srcImage1 = g_srcImage.clone();

    //【2】存留一张灰度图
    cvtColor(g_srcImage1, g_grayImage, COLOR_BGR2GRAY);

    //【3】创建窗口和滚动条
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    createTrackbar("阈值: ", WINDOW_NAME1, &thresh, max_thresh, on_CornerHarris);

    //【4】调用一次回调函数，进行初始化
    on_CornerHarris(0, 0);

    waitKey(0);
    return(0);
}

//-----------------------------------【on_HoughLines( )函数】--------------------------------
//		描述：回调函数
//----------------------------------------------------------------------------------------------

void on_CornerHarris(int, void*)
{
    //---------------------------【1】定义一些局部变量-----------------------------
    Mat dstImage;//目标图
    Mat normImage;//归一化后的图
    Mat scaledImage;//线性变换后的八位无符号整型的图

    //---------------------------【2】初始化---------------------------------------
    //置零当前需要显示的两幅图，即清除上一次调用此函数时他们的值
    dstImage = Mat::zeros(g_srcImage.size(), CV_32FC1);
    g_srcImage1 = g_srcImage.clone();

    //---------------------------【3】正式检测-------------------------------------
    //进行角点检测
    cornerHarris(g_grayImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);

    // 归一化与转换
    normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(normImage, scaledImage);//将归一化后的图线性变换成8位无符号整型 

    //---------------------------【4】进行绘制-------------------------------------
    // 将检测到的，且符合阈值条件的角点绘制出来
    for (int j = 0; j < normImage.rows; j++)
    {
        for (int i = 0; i < normImage.cols; i++)
        {
            if ((int)normImage.at<float>(j, i) > thresh + 80)
            {
                circle(g_srcImage1, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
                circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
            }
        }
    }
    //---------------------------【4】显示最终效果---------------------------------
    imshow(WINDOW_NAME1, g_srcImage1);
    imshow(WINDOW_NAME2, scaledImage);

}

//-----------------------------------【ShowHelpText( )函数】----------------------------------
//		描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //输出欢迎信息和OpenCV版本
    printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
    printf("\n\n\t\t\t此为本书OpenCV3版的第86个配套示例程序\n");
    printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //输出一些帮助信息
    printf("\n\n\n\t【欢迎来到Harris角点检测示例程序~】\n\n");
    printf("\n\t请调整滚动条观察图像效果~\n\n");
}

#endif


#if SNIPPET042

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat image1;
    cv::Mat image2;

    // open images
    image1 = cv::imread("boldt.jpg");
    image2 = cv::imread("rain.jpg");
    if (!image1.data)
        return 0;
    if (!image2.data)
        return 0;

    cv::namedWindow("Image 1");
    cv::imshow("Image 1", image1);
    cv::namedWindow("Image 2");
    cv::imshow("Image 2", image2);

    cv::Mat result;
    // add two images
    cv::addWeighted(image1, 0.7, image2, 0.9, 0., result);

    cv::namedWindow("result");
    cv::imshow("result", result);

    // using overloaded operator
    result = 0.7*image1 + 0.9*image2;

    cv::namedWindow("result with operators");
    cv::imshow("result with operators", result);

    image2 = cv::imread("rain.jpg", 0);

    // create vector of 3 images
    std::vector<cv::Mat> planes;
    // split 1 3-channel image into 3 1-channel images
    cv::split(image1, planes);
    // add to blue channel
    planes[0] += image2;
    // merge the 3 1-channel images into 1 3-channel image
    cv::merge(planes, result);

    cv::namedWindow("Result on blue channel");
    cv::imshow("Result on blue channel", result);

    cv::waitKey();

    return 0;
}



#endif


#if SNIPPET043

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// 1st version
// see recipe Scanning an image with pointers
void colorReduce(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols * image.channels(); // total number of elements per line

    for (int j = 0; j < nl; j++) {

        // get the address of row j
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {

            // process each pixel ---------------------

            data[i] = data[i] / div * div + div / 2;

            // end of pixel processing ----------------

        } // end of line
    }
}

// version with input/ouput images
// see recipe Scanning an image with pointers
void colorReduceIO(const cv::Mat &image, // input image
    cv::Mat &result,      // output image
    int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols; // number of columns
    int nchannels = image.channels(); // number of channels

    // allocate output image if necessary
    result.create(image.rows, image.cols, image.type());

    for (int j = 0; j < nl; j++) {

        // get the addresses of input and output row j
        const uchar* data_in = image.ptr<uchar>(j);
        uchar* data_out = result.ptr<uchar>(j);

        for (int i = 0; i < nc*nchannels; i++) {

            // process each pixel ---------------------

            data_out[i] = data_in[i] / div * div + div / 2;

            // end of pixel processing ----------------

        } // end of line
    }
}

// Test 1
// this version uses the dereference operator *
void colorReduce1(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols * image.channels(); // total number of elements per line
    uchar div2 = div >> 1; // div2 = div/2

    for (int j = 0; j < nl; j++) {

        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {


            // process each pixel ---------------------

            *data++ = *data / div * div + div2;

            // end of pixel processing ----------------

        } // end of line
    }
}

// Test 2
// this version uses the modulo operator
void colorReduce2(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols * image.channels(); // total number of elements per line
    uchar div2 = div >> 1; // div2 = div/2

    for (int j = 0; j < nl; j++) {

        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {

            // process each pixel ---------------------

            int v = *data;
            *data++ = v - v % div + div2;

            // end of pixel processing ----------------

        } // end of line
    }
}

// Test 3
// this version uses a binary mask
void colorReduce3(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols * image.channels(); // total number of elements per line
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
    uchar div2 = 1 << (n - 1); // div2 = div/2

    for (int j = 0; j < nl; j++) {

        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {

            // process each pixel ---------------------

            *data &= mask;     // masking
            *data++ |= div2;   // add div/2

          // end of pixel processing ----------------

        } // end of line
    }
}


// Test 4
// this version uses direct pointer arithmetic with a binary mask
void colorReduce4(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols * image.channels(); // total number of elements per line
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    int step = image.step; // effective width
    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
    uchar div2 = div >> 1; // div2 = div/2

    // get the pointer to the image buffer
    uchar *data = image.data;

    for (int j = 0; j < nl; j++) {

        for (int i = 0; i < nc; i++) {

            // process each pixel ---------------------

            *(data + i) &= mask;
            *(data + i) += div2;

            // end of pixel processing ----------------

        } // end of line

        data += step;  // next line
    }
}

// Test 5
// this version recomputes row size each time
void colorReduce5(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {

        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < image.cols * image.channels(); i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div / 2;

            // end of pixel processing ----------------

        } // end of line
    }
}

// Test 6
// this version optimizes the case of continuous image
void colorReduce6(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols * image.channels(); // total number of elements per line

    if (image.isContinuous()) {
        // then no padded pixels
        nc = nc * nl;
        nl = 1;  // it is now a 1D array
    }

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
    uchar div2 = div >> 1; // div2 = div/2

   // this loop is executed only once
   // in case of continuous images
    for (int j = 0; j < nl; j++) {

        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div2;

            // end of pixel processing ----------------

        } // end of line
    }
}

// Test 7
// this versions applies reshape on continuous image
void colorReduce7(cv::Mat image, int div = 64) {

    if (image.isContinuous()) {
        // no padded pixels
        image.reshape(1,   // new number of channels
            1); // new number of rows
    }
    // number of columns set accordingly

    int nl = image.rows; // number of lines
    int nc = image.cols*image.channels(); // number of columns

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
    uchar div2 = div >> 1; // div2 = div/2

    for (int j = 0; j < nl; j++) {

        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {

            // process each pixel ---------------------

            *data &= mask;
            *data++ += div2;

            // end of pixel processing ----------------

        } // end of line
    }
}

// Test 8
// this version processes the 3 channels inside the loop with Mat_ iterators
void colorReduce8(cv::Mat image, int div = 64) {

    // get iterators
    cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();
    uchar div2 = div >> 1; // div2 = div/2

    for (; it != itend; ++it) {

        // process each pixel ---------------------

        (*it)[0] = (*it)[0] / div * div + div2;
        (*it)[1] = (*it)[1] / div * div + div2;
        (*it)[2] = (*it)[2] / div * div + div2;

        // end of pixel processing ----------------
    }
}

// Test 9
// this version uses iterators on Vec3b
void colorReduce9(cv::Mat image, int div = 64) {

    // get iterators
    cv::MatIterator_<cv::Vec3b> it = image.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> itend = image.end<cv::Vec3b>();

    const cv::Vec3b offset(div / 2, div / 2, div / 2);

    for (; it != itend; ++it) {

        // process each pixel ---------------------

        *it = *it / div * div + offset;
        // end of pixel processing ----------------
    }
}

// Test 10
// this version uses iterators with a binary mask
void colorReduce10(cv::Mat image, int div = 64) {

    // div must be a power of 2
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
    uchar div2 = div >> 1; // div2 = div/2

    // get iterators
    cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();

    // scan all pixels
    for (; it != itend; ++it) {

        // process each pixel ---------------------

        (*it)[0] &= mask;
        (*it)[0] += div2;
        (*it)[1] &= mask;
        (*it)[1] += div2;
        (*it)[2] &= mask;
        (*it)[2] += div2;

        // end of pixel processing ----------------
    }
}

// Test 11
// this versions uses ierators from Mat_ 
void colorReduce11(cv::Mat image, int div = 64) {

    // get iterators
    cv::Mat_<cv::Vec3b> cimage = image;
    cv::Mat_<cv::Vec3b>::iterator it = cimage.begin();
    cv::Mat_<cv::Vec3b>::iterator itend = cimage.end();
    uchar div2 = div >> 1; // div2 = div/2

    for (; it != itend; it++) {

        // process each pixel ---------------------

        (*it)[0] = (*it)[0] / div * div + div2;
        (*it)[1] = (*it)[1] / div * div + div2;
        (*it)[2] = (*it)[2] / div * div + div2;

        // end of pixel processing ----------------
    }
}


// Test 12
// this version uses the at method
void colorReduce12(cv::Mat image, int div = 64) {

    int nl = image.rows; // number of lines
    int nc = image.cols; // number of columns
    uchar div2 = div >> 1; // div2 = div/2

    for (int j = 0; j < nl; j++) {
        for (int i = 0; i < nc; i++) {

            // process each pixel ---------------------

            image.at<cv::Vec3b>(j, i)[0] = image.at<cv::Vec3b>(j, i)[0] / div * div + div2;
            image.at<cv::Vec3b>(j, i)[1] = image.at<cv::Vec3b>(j, i)[1] / div * div + div2;
            image.at<cv::Vec3b>(j, i)[2] = image.at<cv::Vec3b>(j, i)[2] / div * div + div2;

            // end of pixel processing ----------------

        } // end of line
    }
}


// Test 13
// this version uses Mat overloaded operators
void colorReduce13(cv::Mat image, int div = 64) {

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    // mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

    // perform color reduction
    image = (image&cv::Scalar(mask, mask, mask)) + cv::Scalar(div / 2, div / 2, div / 2);
}

// Test 14
// this version uses a look up table
void colorReduce14(cv::Mat image, int div = 64) {

    cv::Mat lookup(1, 256, CV_8U);

    for (int i = 0; i < 256; i++) {

        lookup.at<uchar>(i) = i / div * div + div / 2;
    }

    cv::LUT(image, lookup, image);
}

#define NTESTS 15
#define NITERATIONS 10

int main()
{
    // read the image
    cv::Mat image = cv::imread("boldt.jpg");

    // time and process the image
    const int64 start = cv::getTickCount();
    colorReduce(image, 64);
    //Elapsed time in seconds
    double duration = (cv::getTickCount() - start) / cv::getTickFrequency();

    // display the image
    std::cout << "Duration= " << duration << "secs" << std::endl;
    cv::namedWindow("Image");
    cv::imshow("Image", image);

    cv::waitKey();

    // test different versions of the function

    int64 t[NTESTS], tinit;
    // timer values set to 0
    for (int i = 0; i < NTESTS; i++)
        t[i] = 0;

    cv::Mat images[NTESTS];
    cv::Mat result;

    // the versions to be tested
    typedef void(*FunctionPointer)(cv::Mat, int);
    FunctionPointer functions[NTESTS] = { colorReduce, colorReduce1, colorReduce2, colorReduce3, colorReduce4,
                                          colorReduce5, colorReduce6, colorReduce7, colorReduce8, colorReduce9,
                                          colorReduce10, colorReduce11, colorReduce12, colorReduce13, colorReduce14 };
    // repeat the tests several times
    int n = NITERATIONS;
    for (int k = 0; k < n; k++) {

        std::cout << k << " of " << n << std::endl;

        // test each version
        for (int c = 0; c < NTESTS; c++) {

            images[c] = cv::imread("boldt.jpg");

            // set timer and call function
            tinit = cv::getTickCount();
            functions[c](images[c], 64);
            t[c] += cv::getTickCount() - tinit;

            std::cout << ".";
        }

        std::cout << std::endl;
    }

    // short description of each function
    std::string descriptions[NTESTS] = {
        "original version:",
        "with dereference operator:",
        "using modulo operator:",
        "using a binary mask:",
        "direct ptr arithmetic:",
        "row size recomputation:",
        "continuous image:",
        "reshape continuous image:",
        "with iterators:",
        "Vec3b iterators:",
        "iterators and mask:",
        "iterators from Mat_:",
        "at method:",
        "overloaded operators:",
        "look-up table:",
    };

    for (int i = 0; i < NTESTS; i++) {

        cv::namedWindow(descriptions[i]);
        cv::imshow(descriptions[i], images[i]);
    }

    // print average execution time
    std::cout << std::endl << "-------------------------------------------" << std::endl << std::endl;
    for (int i = 0; i < NTESTS; i++) {

        std::cout << i << ". " << descriptions[i] << 1000.*t[i] / cv::getTickFrequency() / n << "ms" << std::endl;
    }

    cv::waitKey();
    return 0;
}

#endif

#if SNIPPET044

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


int main()
{
    // Read input image
    cv::Mat image = cv::imread("boldt.jpg");
    if (!image.data)
        return 0;

    // Display the image
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    // define bounding rectangle 
    cv::Rect rectangle(50, 25, 210, 180);
    // the models (internally used)
    cv::Mat bgModel, fgModel;
    // segmentation result
    cv::Mat result; // segmentation (4 possible values)

    // GrabCut segmentation
    cv::grabCut(image,    // input image
        result,   // segmentation result
        rectangle,// rectangle containing foreground 
        bgModel, fgModel, // models
        5,        // number of iterations
        cv::GC_INIT_WITH_RECT); // use rectangle

    // Get the pixels marked as likely foreground
    cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
    // or:
    //	result= result&1;

    // create a white image
    cv::Mat foreground(image.size(), CV_8UC3,
        cv::Scalar(255, 255, 255));

    image.copyTo(foreground, result); // bg pixels not copied

    // draw rectangle on original image
    cv::rectangle(image, rectangle, cv::Scalar(255, 255, 255), 1);
    cv::namedWindow("Image with rectangle");
    cv::imshow("Image with rectangle", image);

    // display result
    cv::namedWindow("Foreground object");
    cv::imshow("Foreground object", foreground);

    cv::waitKey();
    return 0;
}


#endif


#if SNIPPET045

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>

void detectHScolor(const cv::Mat& image,		// input image 
    double minHue, double maxHue,	// Hue interval 
    double minSat, double maxSat,	// saturation interval
    cv::Mat& mask) {				// output mask

    // convert into HSV space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // split the 3 channels into 3 images
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    // channels[0] is the Hue
    // channels[1] is the Saturation
    // channels[2] is the Value

    // Hue masking
    cv::Mat mask1; // below maxHue
    cv::threshold(channels[0], mask1, maxHue, 255, cv::THRESH_BINARY_INV);
    cv::Mat mask2; // over minHue
    cv::threshold(channels[0], mask2, minHue, 255, cv::THRESH_BINARY);

    cv::Mat hueMask; // hue mask
    if (minHue < maxHue)
        hueMask = mask1 & mask2;
    else // if interval crosses the zero-degree axis
        hueMask = mask1 | mask2;

    // Saturation masking
    // below maxSat
    cv::threshold(channels[1], mask1, maxSat, 255, cv::THRESH_BINARY_INV);
    // over minSat
    cv::threshold(channels[1], mask2, minSat, 255, cv::THRESH_BINARY);

    cv::Mat satMask; // saturation mask
    satMask = mask1 & mask2;

    // combined mask
    mask = hueMask & satMask;
}

int main()
{
    // read the image
    cv::Mat image = cv::imread("boldt.jpg");
    if (!image.data)
        return 0;

    // show original image
    cv::namedWindow("Original image");
    cv::imshow("Original image", image);

    // convert into HSV space
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // split the 3 channels into 3 images
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    // channels[0] is the Hue
    // channels[1] is the Saturation
    // channels[2] is the Value

    // display value
    cv::namedWindow("Value");
    cv::imshow("Value", channels[2]);

    // display saturation
    cv::namedWindow("Saturation");
    cv::imshow("Saturation", channels[1]);

    // display hue
    cv::namedWindow("Hue");
    cv::imshow("Hue", channels[0]);

    // image with fixed value
    cv::Mat newImage;
    cv::Mat tmp(channels[2].clone());
    // Value channel will be 255 for all pixels
    channels[2] = 255;
    // merge back the channels
    cv::merge(channels, hsv);
    // re-convert to BGR
    cv::cvtColor(hsv, newImage, cv::COLOR_HSV2BGR);

    cv::namedWindow("Fixed Value Image");
    cv::imshow("Fixed Value Image", newImage);

    // image with fixed saturation
    channels[1] = 255;
    channels[2] = tmp;
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, newImage, cv::COLOR_HSV2BGR);

    cv::namedWindow("Fixed saturation");
    cv::imshow("Fixed saturation", newImage);

    // image with fixed value and fixed saturation
    channels[1] = 255;
    channels[2] = 255;
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, newImage, cv::COLOR_HSV2BGR);

    cv::namedWindow("Fixed saturation/value");
    cv::imshow("Fixed saturation/value", newImage);

    // artificial image shown the all possible HS colors
    cv::Mat hs(128, 360, CV_8UC3);
    for (int h = 0; h < 360; h++) {
        for (int s = 0; s < 128; s++) {
            hs.at<cv::Vec3b>(s, h)[0] = h / 2;     // all hue angles
            hs.at<cv::Vec3b>(s, h)[1] = 255 - s * 2; // from high saturation to low
            hs.at<cv::Vec3b>(s, h)[2] = 255;     // constant value
        }
    }

    cv::cvtColor(hs, newImage, cv::COLOR_HSV2BGR);

    cv::namedWindow("Hue/Saturation");
    cv::imshow("Hue/Saturation", newImage);

    // Testing skin detection

    // read the image
    image = cv::imread("girl.jpg");
    if (!image.data)
        return 0;

    // show original image
    cv::namedWindow("Original image");
    cv::imshow("Original image", image);

    // detect skin tone
    cv::Mat mask;
    detectHScolor(image,
        160, 10, // hue from 320 degrees to 20 degrees 
        25, 166, // saturation from ~0.1 to 0.65
        mask);

    // show masked image
    cv::Mat detected(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    image.copyTo(detected, mask);
    cv::imshow("Detection result", detected);

    // A test comparing luminance and brightness

    // create linear intensity image
    cv::Mat linear(100, 256, CV_8U);
    for (int i = 0; i < 256; i++) {

        linear.col(i) = i;
    }

    // create a Lab image
    linear.copyTo(channels[0]);
    cv::Mat constante(100, 256, CV_8U, cv::Scalar(128));
    constante.copyTo(channels[1]);
    constante.copyTo(channels[2]);
    cv::merge(channels, image);

    // convert back to BGR
    cv::Mat brightness;
    cv::cvtColor(image, brightness, cv::COLOR_Lab2BGR);
    cv::split(brightness, channels);

    // create combined image
    cv::Mat combined(200, 256, CV_8U);
    cv::Mat half1(combined, cv::Rect(0, 0, 256, 100));
    linear.copyTo(half1);
    cv::Mat half2(combined, cv::Rect(0, 100, 256, 100));
    channels[0].copyTo(half2);

    cv::namedWindow("Luminance vs Brightness");
    cv::imshow("Luminance vs Brightness", combined);

    cv::waitKey();
}

#endif

#if SNIPPET046

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    // Read image 
    Mat src = imread("threshold.png", IMREAD_GRAYSCALE);
    Mat dst;

    {
        // Basic threhold example 
        threshold(src, dst, 0, 255, THRESH_BINARY);
        imshow("opencv-threshold-example.jpg", dst);
    }

    {
        // Thresholding with maxval set to 128
        threshold(src, dst, 0, 128, THRESH_BINARY);
        imshow("opencv-thresh-binary-maxval.jpg", dst);
    }

    {
        // Thresholding with threshold value set 127 
        threshold(src, dst, 127, 255, THRESH_BINARY);
        imshow("opencv-thresh-binary.jpg", dst);
    }

    {
        // Thresholding using THRESH_BINARY_INV 
        threshold(src, dst, 127, 255, THRESH_BINARY_INV);
        imshow("opencv-thresh-binary-inv.jpg", dst);
    }

    {
        // Thresholding using THRESH_TRUNC 
        threshold(src, dst, 127, 255, THRESH_TRUNC);
        imshow("opencv-thresh-trunc.jpg", dst);
    }

    {
        // Thresholding using THRESH_TOZERO 
        threshold(src, dst, 127, 255, THRESH_TOZERO);
        imshow("opencv-thresh-tozero.jpg", dst);
    }

    {
        // Thresholding using THRESH_TOZERO_INV 
        threshold(src, dst, 127, 255, THRESH_TOZERO_INV);
        imshow("opencv-thresh-to-zero-inv.jpg", dst);
    }

    cv::waitKey(0);
}


#endif


#if SNIPPET047


#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void*);

int main(int argc, char** argv)
{
   
    Mat src = imread("HappyFish.jpg");
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    blur(src_gray, src_gray, Size(3, 3));
    const char* source_window = "Source";
    namedWindow(source_window);
    imshow(source_window, src);
    const int max_thresh = 255;
    createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
    thresh_callback(0, 0);
    waitKey();
    return 0;
}
void thresh_callback(int, void*)
{
    Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 2);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
    imshow("Contours", drawing);
}


#endif


#if SNIPPET048

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void*);
int main(int argc, char** argv)
{
    
    Mat src = imread("stuff.jpg");
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    blur(src_gray, src_gray, Size(3, 3));
    const char* source_window = "Source";
    namedWindow(source_window);
    imshow(source_window, src);
    const int max_thresh = 255;
    createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
    thresh_callback(0, 0);
    waitKey();
    return 0;
}
void thresh_callback(int, void*)
{
    Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 2);
    vector<vector<Point> > contours;
    findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> >hull(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        convexHull(contours[i], hull[i]);
    }
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color);
        drawContours(drawing, hull, (int)i, color);
    }
    imshow("Hull demo", drawing);
}

#endif


#if SNIPPET049


#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <iomanip>
using namespace cv;
using namespace std;
Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void*);
int main(int argc, char** argv)
{
    
    Mat src = imread("stuff.jpg");
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    blur(src_gray, src_gray, Size(3, 3));
    const char* source_window = "Source";
    namedWindow(source_window);
    imshow(source_window, src);
    const int max_thresh = 255;
    createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
    thresh_callback(0, 0);
    waitKey();
    return 0;
}
void thresh_callback(int, void*)
{
    Mat canny_output;
    Canny(src_gray, canny_output, thresh, thresh * 2, 3);
    vector<vector<Point> > contours;
    findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i]);
    }
    vector<Point2f> mc(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        //add 1e-5 to avoid division by zero
        mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
            static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
        cout << "mc[" << i << "]=" << mc[i] << endl;
    }
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color, 2);
        circle(drawing, mc[i], 4, color, -1);
    }
    imshow("Contours", drawing);
    cout << "\t Info: Area and Contour Length \n";
    for (size_t i = 0; i < contours.size(); i++)
    {
        cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
            << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength(contours[i], true) << endl;
    }
}


#endif


#if SNIPPET050

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout
        << "\nThis program illustrates the use of findContours and drawContours\n"
        << "The original image is put up along with the image of drawn contours\n"
        << "Usage:\n"
        << "./contours2\n"
        << "\nA trackbar is put up which controls the contour level from -3 to 3\n"
        << endl;
}

const int w = 500;
int levels = 3;

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

static void on_trackbar(int, void*)
{
    Mat cnt_img = Mat::zeros(w, w, CV_8UC3);
    int _levels = levels - 3;
    drawContours(cnt_img, contours, _levels <= 0 ? 3 : -1, Scalar(128, 255, 255),
        3, LINE_AA, hierarchy, std::abs(_levels));

    imshow("contours", cnt_img);
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{help h||}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    Mat img = Mat::zeros(w, w, CV_8UC1);
    //Draw 6 faces
    for (int i = 0; i < 6; i++)
    {
        int dx = (i % 2) * 250 - 30;
        int dy = (i / 2) * 150;
        const Scalar white = Scalar(255);
        const Scalar black = Scalar(0);

        if (i == 0)
        {
            for (int j = 0; j <= 10; j++)
            {
                double angle = (j + 5)*CV_PI / 21;
                line(img, Point(cvRound(dx + 100 + j * 10 - 80 * cos(angle)),
                    cvRound(dy + 100 - 90 * sin(angle))),
                    Point(cvRound(dx + 100 + j * 10 - 30 * cos(angle)),
                        cvRound(dy + 100 - 30 * sin(angle))), white, 1, 8, 0);
            }
        }

        ellipse(img, Point(dx + 150, dy + 100), Size(100, 70), 0, 0, 360, white, -1, 8, 0);
        ellipse(img, Point(dx + 115, dy + 70), Size(30, 20), 0, 0, 360, black, -1, 8, 0);
        ellipse(img, Point(dx + 185, dy + 70), Size(30, 20), 0, 0, 360, black, -1, 8, 0);
        ellipse(img, Point(dx + 115, dy + 70), Size(15, 15), 0, 0, 360, white, -1, 8, 0);
        ellipse(img, Point(dx + 185, dy + 70), Size(15, 15), 0, 0, 360, white, -1, 8, 0);
        ellipse(img, Point(dx + 115, dy + 70), Size(5, 5), 0, 0, 360, black, -1, 8, 0);
        ellipse(img, Point(dx + 185, dy + 70), Size(5, 5), 0, 0, 360, black, -1, 8, 0);
        ellipse(img, Point(dx + 150, dy + 100), Size(10, 5), 0, 0, 360, black, -1, 8, 0);
        ellipse(img, Point(dx + 150, dy + 150), Size(40, 10), 0, 0, 360, black, -1, 8, 0);
        ellipse(img, Point(dx + 27, dy + 100), Size(20, 35), 0, 0, 360, white, -1, 8, 0);
        ellipse(img, Point(dx + 273, dy + 100), Size(20, 35), 0, 0, 360, white, -1, 8, 0);
    }
    //show the faces
    namedWindow("image", 1);
    imshow("image", img);
    //Extract the contours so that
    vector<vector<Point> > contours0;
    findContours(img, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    contours.resize(contours0.size());
    for (size_t k = 0; k < contours0.size(); k++)
        approxPolyDP(Mat(contours0[k]), contours[k], 3, true);

    namedWindow("contours", 1);
    createTrackbar("levels+3", "contours", &levels, 7, on_trackbar);

    on_trackbar(0, 0);
    waitKey();

    return 0;
}


#endif

#if SNIPPET051

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#define HUEMAX 179
#define SATMAX 255
#define VALMAX 255

using namespace std;
using namespace cv;

Mat HSV;
int H = 170;
int S = 200;
int V = 200;
int R = 0;
int G = 0;
int B = 0;

int MAX_H = 179;
int MAX_S = 255;
int MAX_V = 255;
int mouse_x = 0;
int mouse_y = 0;
char window_name[20] = "HSV Color Plot";

//Global variable for hsv color wheel plot
int max_hue_range = 179;
int max_step = 3; //nuber of pixel for each hue color
int wheel_width = max_hue_range * max_step;
int wheel_hight = 50;
int wheel_x = 50; //x-position of wheel
int wheel_y = 5;//y-position of wheel

//Global variable plot for satuarion-value plot
int S_V_Width = MAX_S;
int S_V_Height = MAX_S;
int S_V_x = 10;
int S_V_y = wheel_y + wheel_hight + 20;

//Global variable for HSV plot
int HSV_Width = 150;
int HSV_Height = 150;
int HSV_x = S_V_x + S_V_Width + 30;
int HSV_y = S_V_y + 50;


void onTrackbar_changed(int, void*);
static void onMouse(int event, int x, int y, int, void*);
void drawPointers(void);

int main()
{
    HSV.create(390, 640, CV_8UC3);
    HSV.setTo(Scalar(200, 0, 200));

    namedWindow(window_name);
    createTrackbar("Hue", window_name, &H, HUEMAX, onTrackbar_changed);
    createTrackbar("Saturation", window_name, &S, SATMAX, onTrackbar_changed);
    createTrackbar("Value", window_name, &V, VALMAX, onTrackbar_changed);
    onTrackbar_changed(0, 0); //initialize window

    setMouseCallback(window_name, onMouse, 0);
    while (true)
    {
        int c;
        c = waitKey(20);
        if ((char)c == 27)
        {
            break;
        }
    }

    return 0;
}

void onTrackbar_changed(int, void*) {

    //Plot color wheel.
    int hue_range = 0;
    int step = 1;
    for (int i = wheel_y; i < wheel_hight + wheel_y; i++) {
        hue_range = 0;
        for (int j = wheel_x; j < wheel_width + wheel_x; j++) {
            if (hue_range >= max_hue_range) hue_range = 0;
            if (step++ == max_step) {
                hue_range++;
                step = 1;
            }
            Vec3b pix;
            pix.val[0] = hue_range;
            pix.val[1] = 255;
            pix.val[2] = 255;


            HSV.at<Vec3b>(i, j) = pix;

        }
    }


    //Plot for saturation and value
    int sat_range = 0;
    int value_range = 255;
    for (int i = S_V_y; i < S_V_Height + S_V_y; i++) {
        value_range--;
        sat_range = 0;
        for (int j = S_V_x; j < S_V_Width + S_V_x; j++) {
            Vec3b pix;
            pix.val[0] = H;
            pix.val[1] = sat_range++;
            pix.val[2] = value_range;
            HSV.at<Vec3b>(i, j) = pix;

        }

    }

    //Plot for HSV
    Mat roi1(HSV, Rect(HSV_x, HSV_y, HSV_Width, HSV_Height));
    roi1 = Scalar(H, S, V);
    drawPointers();

    Mat RGB;
    cvtColor(HSV, RGB, COLOR_HSV2BGR);

    imshow(window_name, RGB);
    imwrite("hsv.jpg", RGB);

}

static void onMouse(int event, int x, int y, int f, void*) {
    if (f&EVENT_FLAG_LBUTTON) {
        mouse_x = x;
        mouse_y = y;
        if (((wheel_x <= x) && (x <= wheel_x + wheel_width)) && ((wheel_y <= y) && (y <= wheel_y + wheel_hight))) {
            H = (x - wheel_x) / max_step;
            setTrackbarPos("Hue", window_name, H);
        }
        else if (((S_V_x <= x) && (x <= S_V_x + S_V_Width)) && ((S_V_y <= y) && (y <= S_V_y + S_V_Height))) {

            S = x - S_V_x;
            y = y - S_V_y;
            V = 255 - y;

            setTrackbarPos("Saturation", window_name, S);
            setTrackbarPos("Value", window_name, V);
        }

    }

}

void drawPointers() {
    // Point p(S_V_x+S,S_V_y+(255-V));
    Point p(S, 255 - V);

    int index = 10;
    Point p1, p2;
    p1.x = p.x - index;
    p1.y = p.y;
    p2.x = p.x + index;
    p2.y = p.y;

    Mat roi1(HSV, Rect(S_V_x, S_V_y, S_V_Width, S_V_Height));
    line(roi1, p1, p2, Scalar(255, 255, 255), 1, LINE_AA, 0);
    p1.x = p.x;
    p1.y = p.y - index;
    p2.x = p.x;
    p2.y = p.y + index;
    line(roi1, p1, p2, Scalar(255, 255, 255), 1, LINE_AA, 0);

    int x_index = wheel_x + H * max_step;
    if (x_index >= wheel_x + wheel_width) x_index = wheel_x + wheel_width - 2;
    if (x_index <= wheel_x) x_index = wheel_x + 2;

    p1.x = x_index;
    p1.y = wheel_y + 1;
    p2.x = x_index;
    p2.y = wheel_y + 20;
    line(HSV, p1, p2, Scalar(255, 255, 255), 2, LINE_AA, 0);

    Mat RGB(1, 1, CV_8UC3);
    Mat temp;
    RGB = Scalar(H, S, V);
    cvtColor(RGB, temp, COLOR_HSV2BGR);
    Vec3b rgb = temp.at<Vec3b>(0, 0);
    B = rgb.val[0];
    G = rgb.val[1];
    R = rgb.val[2];

    Mat roi2(HSV, Rect(450, 130, 175, 175));
    roi2 = Scalar(200, 0, 200);

    char name[30];
    sprintf(name, "R=%d", R);
    putText(HSV, name, Point(460, 155), FONT_HERSHEY_SIMPLEX, .7, Scalar(5, 255, 255), 2, 8, false);

    sprintf(name, "G=%d", G);
    putText(HSV, name, Point(460, 180), FONT_HERSHEY_SIMPLEX, .7, Scalar(5, 255, 255), 2, 8, false);

    sprintf(name, "B=%d", B);
    putText(HSV, name, Point(460, 205), FONT_HERSHEY_SIMPLEX, .7, Scalar(5, 255, 255), 2, 8, false);


    sprintf(name, "H=%d", H);
    putText(HSV, name, Point(545, 155), FONT_HERSHEY_SIMPLEX, .7, Scalar(5, 255, 255), 2, 8, false);

    sprintf(name, "S=%d", S);
    putText(HSV, name, Point(545, 180), FONT_HERSHEY_SIMPLEX, .7, Scalar(5, 255, 255), 2, 8, false);

    sprintf(name, "V=%d", V);
    putText(HSV, name, Point(545, 205), FONT_HERSHEY_SIMPLEX, .7, Scalar(5, 255, 255), 2, 8, false);
}

#endif


#if SNIPPET052

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

#define 	CV_FILLED   -1
#define     CV_LOAD_IMAGE_GRAYSCALE  0

vector<vector<Point> > contours; 
vector<Vec4i> hierarchy; 
vector<char> fillCtr; 
Mat c;

void DrawTree(int idx, int level)
{
    int i = idx;
    if (level % 2 == 0)
        drawContours(c, contours, i, Scalar(255, 0, 0), CV_FILLED);
    else
        drawContours(c, contours, i, Scalar(0, 255, 0), CV_FILLED);
    fillCtr[i] = 1;
    while (hierarchy[i][0] != -1)
    {
        int j = hierarchy[i][0];
        if (fillCtr[j] == 0)
        {
            if (level % 2 == 0)
                drawContours(c, contours, j, Scalar(255, 0, 0), CV_FILLED);
            else
                drawContours(c, contours, j, Scalar(0, 255, 0), CV_FILLED);
            fillCtr[j] = 1;
            DrawTree(j, level);
        }
        i = hierarchy[i][0];

    }
    if (hierarchy[idx][2] != -1)
        DrawTree(hierarchy[idx][2], level + 1);
}

int main(int argc, char **argv)

{
    Mat x = imread("14415468805620458.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    imshow("original", x);
    Mat y;
    threshold(x, y, 50, 255, THRESH_BINARY);
    imshow("threshold", y);
    Mat yc;
    findContours(y, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    c = Mat::zeros(x.size(), CV_8UC3);
    vector<Mat> plan = { x,x,x };
    merge(plan, c);
    cout << c.channels() << "\n";
    fillCtr.resize(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        if (hierarchy[i][3] == -1 && fillCtr[i] == 0)
        {
            DrawTree(i, 0);
        }
    }

    imshow("contour", c);
    waitKey(0);
}

#endif