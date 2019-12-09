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