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


//--------------------------------��onMouse( )�ص�������------------------------------------
//      �������������ص�
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
    //�˾�����OpenCV2��Ϊ��
    //case CV_EVENT_LBUTTONDOWN:
    //�˾�����OpenCV3��Ϊ��
    case EVENT_LBUTTONDOWN:
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        break;

    //�˾�����OpenCV2��Ϊ��
    //case CV_EVENT_LBUTTONUP:
    //�˾�����OpenCV3��Ϊ��
    case EVENT_LBUTTONUP:
        selectObject = false;

        if (selection.width > 0 && selection.height > 0)
            trackObject = -1;

        break;
    }
}

//--------------------------------��help( )������----------------------------------------------
//      ���������������Ϣ
//-------------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    cout << "\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n"
         << "\n\n\t\t\t��Ϊ����OpenCV3��ĵ�8������ʾ������\n"
         << "\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" << CV_VERSION
         << "\n\n  ----------------------------------------------------------------------------";
    cout << "\n\n\t��Demo��ʾ�˻��ھ�ֵƯ�Ƶ�׷�٣�tracking������\n"
         "\t��������ѡһ������ɫ�����壬��������׷�ٲ���\n";
    cout << "\n\n\t����˵���� \n"
         "\t\t������ѡ��������ʼ������\n"
         "\t\tESC - �˳�����\n"
         "\t\tc - ֹͣ׷��\n"
         "\t\tb - ��/��-ͶӰ��ͼ\n"
         "\t\th - ��ʾ/����-����ֱ��ͼ\n"
         "\t\tp - ��ͣ��Ƶ\n";
}

const char* keys = {
    "{1|  | 0 | camera number}"
};



//-----------------------------------��main( )������--------------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
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
            cout << "���ܳ�ʼ������ͷ\n";
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
                        //�˾�����OpenCV3��Ϊ��
                        normalize(hist, hist, 0, 255, NORM_MINMAX);
                        //�˾�����OpenCV2��Ϊ��
                        //normalize(hist, hist, 0, 255, CV_MINMAX);
                        trackWindow = selection;
                        trackObject = 1;
                        histimg = Scalar::all(0);
                        int binW = histimg.cols / hsize;
                        Mat buf(1, hsize, CV_8UC3);

                        for (int i = 0; i < hsize; i++)
                            buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);

                        //�˾�����OpenCV3��Ϊ��
                        cvtColor(buf, buf, COLOR_HSV2BGR);
                        //�˾�����OpenCV2��Ϊ��
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
                                                    //�˾�����OpenCV3��Ϊ��
                                                    TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
                    //�˾�����OpenCV2��Ϊ��
                    //TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                    if (trackWindow.area() <= 1) {
                        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                           trackWindow.x + r, trackWindow.y + r) &
                                      Rect(0, 0, cols, rows);
                    }

                    if (backprojMode)
                        cvtColor(backproj, image, COLOR_GRAY2BGR);

                    //�˾�����OpenCV3��Ϊ��
                    ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
                    //�˾�����OpenCV2��Ϊ��
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
        // ����ԭͼ
        Mat image = imread("1.jpg");
        //��������
        namedWindow("�����˲���ԭͼ��");
        namedWindow("�����˲���Ч��ͼ��");
        //��ʾԭͼ
        imshow("�����˲���ԭͼ��", image);
        //���з����˲�����
        Mat out;
        boxFilter(image, out, -1, Size(5, 5));
        //��ʾЧ��ͼ
        imshow("�����˲���Ч��ͼ��", out);
        waitKey(0);
    }
    {
        //��1������ԭʼͼ
        Mat srcImage = imread("1.jpg");
        //��2����ʾԭʼͼ
        imshow("��ֵ�˲���ԭͼ��", srcImage);
        //��3�����о�ֵ�˲�����
        Mat dstImage;
        blur(srcImage, dstImage, Size(7, 7));
        //��4����ʾЧ��ͼ
        imshow("��ֵ�˲���Ч��ͼ��", dstImage);
        waitKey(0);
    }
    {
        // ����ԭͼ
        Mat image = imread("1.jpg");
        //��������
        namedWindow("��˹�˲���ԭͼ��");
        namedWindow("��˹�˲���Ч��ͼ��");
        //��ʾԭͼ
        imshow("��˹�˲���ԭͼ��", image);
        //���и�˹�˲�����
        Mat out;
        GaussianBlur(image, out, Size(5, 5), 0, 0);
        //��ʾЧ��ͼ
        imshow("��˹�˲���Ч��ͼ��", out);
        waitKey(0);
    }
    {
        // ����ԭͼ
        Mat image = imread("1.jpg");
        //��������
        namedWindow("��ֵ�˲���ԭͼ��");
        namedWindow("��ֵ�˲���Ч��ͼ��");
        //��ʾԭͼ
        imshow("��ֵ�˲���ԭͼ��", image);
        //������ֵ�˲�����
        Mat out;
        medianBlur(image, out, 7);
        //��ʾЧ��ͼ
        imshow("��ֵ�˲���Ч��ͼ��", out);
        waitKey(0);
    }
    {
        // ����ԭͼ
        Mat image = imread("1.jpg");
        //��������
        namedWindow("˫���˲���ԭͼ��");
        namedWindow("˫���˲���Ч��ͼ��");
        //��ʾԭͼ
        imshow("˫���˲���ԭͼ��", image);
        //����˫���˲�����
        Mat out;
        bilateralFilter(image, out, 25, 25 * 2, 25 / 2);
        //��ʾЧ��ͼ
        imshow("˫���˲���Ч��ͼ��", out);
        waitKey(0);
    }
    {
        // ��1������Դͼ��
        Mat srcImage, dstImage;
        srcImage = imread("1.jpg", 1);

        if (!srcImage.data) {
            printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����~�� \n");
            return false;
        }

        // ��2��תΪ�Ҷ�ͼ����ʾ����
        cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
        imshow("ԭʼͼ", srcImage);
        // ��3������ֱ��ͼ���⻯
        equalizeHist(srcImage, dstImage);
        // ��4����ʾ���
        imshow("����ֱ��ͼ���⻯���ͼ", dstImage);
        // �ȴ��û������˳�����
        waitKey(0);
    }
    {
        //����ԭʼͼ
        Mat image = imread("1.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
        //��������
        namedWindow("��ԭʼͼ������");
        namedWindow("��Ч��ͼ������");
        //��ʾԭʼͼ
        imshow("��ԭʼͼ������", image);
        //�����
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //������̬ѧ����
        morphologyEx(image, image, MORPH_DILATE, element);
        //��ʾЧ��ͼ
        imshow("��Ч��ͼ������", image);
        waitKey(0);
    }
    {
        //����ԭʼͼ
        Mat image = imread("1.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
        //��������
        namedWindow("��ԭʼͼ����ʴ");
        namedWindow("��Ч��ͼ����ʴ");
        //��ʾԭʼͼ
        imshow("��ԭʼͼ����ʴ", image);
        //�����
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //������̬ѧ����
        morphologyEx(image, image, MORPH_ERODE, element);
        //��ʾЧ��ͼ
        imshow("��Ч��ͼ����ʴ", image);
        waitKey(0);
    }
    {
        //����ԭʼͼ
        Mat image = imread("1.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
        //��������
        namedWindow("��ԭʼͼ��������");
        namedWindow("��Ч��ͼ��������");
        //��ʾԭʼͼ
        imshow("��ԭʼͼ��������", image);
        //�����
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //������̬ѧ����
        morphologyEx(image, image, MORPH_OPEN, element);
        //��ʾЧ��ͼ
        imshow("��Ч��ͼ��������", image);
        waitKey(0);
    }
    {
        //����ԭʼͼ
        Mat image = imread("1.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
        //��������
        namedWindow("��ԭʼͼ��������");
        namedWindow("��Ч��ͼ��������");
        //��ʾԭʼͼ
        imshow("��ԭʼͼ��������", image);
        //�����
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
        //������̬ѧ����
        morphologyEx(image, image, MORPH_CLOSE, element);
        //��ʾЧ��ͼ
        imshow("��Ч��ͼ��������", image);
        waitKey(0);
    }
    {
        //����ԭʼͼ
        Mat srcImage = imread("1.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
        Mat srcImage1 = srcImage.clone();
        //��ʾԭʼͼ
        imshow("��ԭʼͼ��Canny��Ե���", srcImage);
        //----------------------------------------------------------------------------------
        //  һ����򵥵�canny�÷����õ�ԭͼ��ֱ���á�
        //  ע�⣺�˷�����OpenCV2�п��ã���OpenCV3����ʧЧ
        //----------------------------------------------------------------------------------
        //Canny( srcImage, srcImage, 150, 100,3 );
        //imshow("��Ч��ͼ��Canny��Ե���", srcImage);
        //----------------------------------------------------------------------------------
        //  �����߽׵�canny�÷���ת�ɻҶ�ͼ�����룬��canny����󽫵õ��ı�Ե��Ϊ���룬����ԭͼ��Ч��ͼ�ϣ��õ���ɫ�ı�Եͼ
        //----------------------------------------------------------------------------------
        Mat dstImage, edge, grayImage;
        // ��1��������srcͬ���ͺʹ�С�ľ���(dst)
        dstImage.create(srcImage1.size(), srcImage1.type());
        // ��2����ԭͼ��ת��Ϊ�Ҷ�ͼ��
        cvtColor(srcImage1, grayImage, COLOR_BGR2GRAY);
        // ��3������ʹ�� 3x3�ں�������
        blur(grayImage, edge, Size(3, 3));
        // ��4������Canny����
        Canny(edge, edge, 3, 9, 3);
        //��5����g_dstImage�ڵ�����Ԫ������Ϊ0
        dstImage = Scalar::all(0);
        //��6��ʹ��Canny��������ı�Եͼg_cannyDetectedEdges��Ϊ���룬����ԭͼg_srcImage����Ŀ��ͼg_dstImage��
        srcImage1.copyTo(dstImage, edge);
        //��7����ʾЧ��ͼ
        imshow("��Ч��ͼ��Canny��Ե���2", dstImage);
        waitKey(0);
    }
    {
        // ��1������ԭʼͼ���ұ����Զ�ֵͼģʽ����
        Mat srcImage = imread("1.jpg", 0);
        imshow("ԭʼͼ", srcImage);
        //��2����ʼ�����ͼ
        Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
        //��3��srcImageȡ������ֵ119���ǲ���
        srcImage = srcImage > 119;
        imshow("ȡ��ֵ���ԭʼͼ", srcImage);
        //��4�����������Ͳ�νṹ
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        //��5����������
        //�˾�����OpenCV2��Ϊ��
        //findContours( srcImage, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
        //�˾�����OpenCV3��Ϊ��
        findContours(srcImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        // ��6���������ж���������� �������ɫ���Ƴ�ÿ�����������ɫ
        int index = 0;

        for (; index >= 0; index = hierarchy[index][0]) {
            Scalar color(rand() & 255, rand() & 255, rand() & 255);
            //�˾�����OpenCV2��Ϊ��
            //drawContours( dstImage, contours, index, color, CV_FILLED, 8, hierarchy );
            //�˾�����OpenCV3��Ϊ��
            drawContours(dstImage, contours, index, color, FILLED, 8, hierarchy);
        }

        //��7����ʾ��������ͼ
        imshow("����ͼ", dstImage);
        waitKey(0);
    }
    {
        //��0������ԭʼͼ
        Mat srcImage = imread("1.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
        imshow("��ԭʼͼ��Canny��Ե���", srcImage);    //��ʾԭʼͼ
        Mat dstImage, edge, grayImage;  //��������
        //��1��������srcͬ���ͺʹ�С�ľ���(dst)
        dstImage.create(srcImage.size(), srcImage.type());
        //��2����ԭͼ��ת��Ϊ�Ҷ�ͼ��
        //�˾�����OpenCV2��Ϊ��
        //cvtColor( srcImage, grayImage, CV_BGR2GRAY );
        //�˾�����OpenCV3��Ϊ��
        cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
        //��3������ʹ�� 3x3�ں�������
        blur(grayImage, edge, Size(3, 3));
        //��4������Canny����
        Canny(edge, edge, 3, 9, 3);
        //��5����ʾЧ��ͼ
        imshow("��Ч��ͼ��Canny��Ե���", edge);
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
    // ��1������һ��ͼƬ
    Mat img = imread("001.jpg");
    // ��2���ڴ�������ʾ�����ͼƬ
    imshow("�������ͼƬ��", img);
    // ��3���ȴ�6000 ms�󴰿��Զ��ر�
    waitKey(6000);
}

#endif


#if SNIPPET003

#include <opencv2/opencv.hpp>
using namespace cv;

void main()
{
    // ��1������һ��ͼƬ������ͼ��
    Mat srcImage = imread("002.jpg");
    // ��2����ʾ�����ͼƬ
    imshow("��ԭʼͼ��", srcImage);
    // ��3���ȴ����ⰴ������
    waitKey(0);
}

#endif


#if SNIPPET004

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;


int main()
{
    //����ԭͼ
    Mat srcImage = imread("003.jpg");
    //��ʾԭͼ
    imshow("��ԭͼ����ʴ����", srcImage);
    //���и�ʴ����
    Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat dstImage;
    erode(srcImage, dstImage, element);
    //��ʾЧ��ͼ
    imshow("��Ч��ͼ����ʴ����", dstImage);
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
    //��1������ԭʼͼ
    Mat srcImage = imread("005.jpg");
    //��2����ʾԭʼͼ
    imshow("��ֵ�˲���ԭͼ��", srcImage);
    //��3�����о�ֵ�˲�����
    Mat dstImage;
    blur(srcImage, dstImage, Size(7, 7));
    //��4����ʾЧ��ͼ
    imshow("��ֵ�˲���Ч��ͼ��", dstImage);
    waitKey(0);
}

#endif


#if SNIPPET006

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


int main()
{
    //��0������ԭʼͼ
    Mat srcImage = imread("006.jpg");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ
    imshow("��ԭʼͼ��Canny��Ե���", srcImage);    //��ʾԭʼͼ
    Mat dstImage, edge, grayImage;  //��������
    //��1��������srcͬ���ͺʹ�С�ľ���(dst)
    dstImage.create(srcImage.size(), srcImage.type());
    //��2����ԭͼ��ת��Ϊ�Ҷ�ͼ��
    //�˾�����OpenCV2��Ϊ��
    //cvtColor( srcImage, grayImage, CV_BGR2GRAY );
    //�˾�����OpenCV3��Ϊ��
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
    //��3������ʹ�� 3x3�ں�������
    blur(grayImage, edge, Size(3, 3));
    //��4������Canny����
    Canny(edge, edge, 3, 9, 3);
    //��5����ʾЧ��ͼ
    imshow("��Ч��ͼ��Canny��Ե���", edge);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET007

#include <opencv2\opencv.hpp>

using namespace cv;

int main()
{
    //��1��������Ƶ
    VideoCapture capture("007.avi");

    //��2��ѭ����ʾÿһ֡
    while (1) {
        Mat frame;//����һ��Mat���������ڴ洢ÿһ֡��ͼ��
        capture >> frame;  //��ȡ��ǰ֡

        //����Ƶ������ɣ��˳�ѭ��
        if (frame.empty()) {
            break;
        }

        imshow("��ȡ��Ƶ", frame);  //��ʾ��ǰ֡
        waitKey(30);  //��ʱ30ms
    }

    return 0;
}

#endif


#if SNIPPET008

#include <opencv2\opencv.hpp>
using namespace cv;


int main()
{
    //��1��������ͷ������Ƶ
    VideoCapture capture(0);

    //��2��ѭ����ʾÿһ֡
    while (1) {
        Mat frame;  //����һ��Mat���������ڴ洢ÿһ֡��ͼ��
        capture >> frame;  //��ȡ��ǰ֡
        imshow("��ȡ��Ƶ", frame);  //��ʾ��ǰ֡
        waitKey(30);  //��ʱ30ms
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



//-----------------------------------��ȫ�ֱ���������-----------------------------------------
//      ����������ȫ�ֱ���
//-------------------------------------------------------------------------------------------------
Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;


//--------------------------------��onMouse( )�ص�������------------------------------------
//      �������������ص�
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
    //�˾�����OpenCV2��Ϊ��
    //case CV_EVENT_LBUTTONDOWN:
    //�˾�����OpenCV3��Ϊ��
    case EVENT_LBUTTONDOWN:
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        break;

    //�˾�����OpenCV2��Ϊ��
    //case CV_EVENT_LBUTTONUP:
    //�˾�����OpenCV3��Ϊ��
    case EVENT_LBUTTONUP:
        selectObject = false;

        if (selection.width > 0 && selection.height > 0)
            trackObject = -1;

        break;
    }
}

//--------------------------------��help( )������----------------------------------------------
//      ���������������Ϣ
//-------------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    cout << "\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n"
         << "\n\n\t\t\t��Ϊ����OpenCV3��ĵ�8������ʾ������\n"
         << "\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" << CV_VERSION
         << "\n\n  ----------------------------------------------------------------------------";
    cout << "\n\n\t��Demo��ʾ�˻��ھ�ֵƯ�Ƶ�׷�٣�tracking������\n"
         "\t��������ѡһ������ɫ�����壬��������׷�ٲ���\n";
    cout << "\n\n\t����˵���� \n"
         "\t\t������ѡ��������ʼ������\n"
         "\t\tESC - �˳�����\n"
         "\t\tc - ֹͣ׷��\n"
         "\t\tb - ��/��-ͶӰ��ͼ\n"
         "\t\th - ��ʾ/����-����ֱ��ͼ\n"
         "\t\tp - ��ͣ��Ƶ\n";
}

const char* keys = {
    "{1|  | 0 | camera number}"
};


//-----------------------------------��main( )������--------------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
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
        cout << "���ܳ�ʼ������ͷ\n";
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
                    //�˾�����OpenCV3��Ϊ��
                    normalize(hist, hist, 0, 255, NORM_MINMAX);
                    //�˾�����OpenCV2��Ϊ��
                    //normalize(hist, hist, 0, 255, CV_MINMAX);
                    trackWindow = selection;
                    trackObject = 1;
                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);

                    for (int i = 0; i < hsize; i++)
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);

                    //�˾�����OpenCV3��Ϊ��
                    cvtColor(buf, buf, COLOR_HSV2BGR);
                    //�˾�����OpenCV2��Ϊ��
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
                                                //�˾�����OpenCV3��Ϊ��
                                                TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
                //�˾�����OpenCV2��Ϊ��
                //TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                if (trackWindow.area() <= 1) {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                if (backprojMode)
                    cvtColor(backproj, image, COLOR_GRAY2BGR);

                //�˾�����OpenCV3��Ϊ��
                ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
                //�˾�����OpenCV2��Ϊ��
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





//-----------------------------------��ȫ�ֺ���������-----------------------------------------
//      ����������ȫ�ֺ���
//-------------------------------------------------------------------------------------------------
void tracking(Mat &frame, Mat &output);
bool addNewPoints();
bool acceptTrackedPoint(int i);

//-----------------------------------��ȫ�ֱ���������-----------------------------------------
//      ����������ȫ�ֱ���
//-------------------------------------------------------------------------------------------------
string window_name = "optical flow tracking";
Mat gray;   // ��ǰͼƬ
Mat gray_prev;  // Ԥ��ͼƬ
vector<Point2f> points[2];  // point0Ϊ�������ԭ��λ�ã�point1Ϊ���������λ��
vector<Point2f> initial;    // ��ʼ�����ٵ��λ��
vector<Point2f> features;   // ��������
int maxCount = 500; // �������������
double qLevel = 0.01;   // �������ĵȼ�
double minDist = 10.0;  // ��������֮�����С����
vector<uchar> status;   // ����������״̬��������������Ϊ1������Ϊ0
vector<float> err;


//--------------------------------��help( )������----------------------------------------------
//      ���������������Ϣ
//-------------------------------------------------------------------------------------------------
static void help()
{
    //�����ӭ��Ϣ��OpenCV�汾
    cout << "\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n"
         << "\n\n\t\t\t��Ϊ����OpenCV3��ĵ�9������ʾ������\n"
         << "\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" << CV_VERSION
         << "\n\n  ----------------------------------------------------------------------------";
}


//-----------------------------------��main( )������--------------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
//-------------------------------------------------------------------------------------------------
int main()
{
    Mat frame;
    Mat result;
    VideoCapture capture("007.avi");
    help();

    if (capture.isOpened()) { // ����ͷ��ȡ�ļ�����
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
// brief: ����
// parameter: frame �������Ƶ֡
//            output �и��ٽ������Ƶ֡
// return: void
//-------------------------------------------------------------------------------------------------
void tracking(Mat &frame, Mat &output)
{
    //�˾�����OpenCV3��Ϊ��
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    //�˾�����OpenCV2��Ϊ��
    //cvtColor(frame, gray, CV_BGR2GRAY);
    frame.copyTo(output);

    // ���������
    if (addNewPoints()) {
        goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);
        points[0].insert(points[0].end(), features.begin(), features.end());
        initial.insert(initial.end(), features.begin(), features.end());
    }

    if (gray_prev.empty()) {
        gray.copyTo(gray_prev);
    }

    // l-k�������˶�����
    calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);
    // ȥ��һЩ���õ�������
    int k = 0;

    for (size_t i = 0; i < points[1].size(); i++) {
        if (acceptTrackedPoint(i)) {
            initial[k] = initial[i];
            points[1][k++] = points[1][i];
        }
    }

    points[1].resize(k);
    initial.resize(k);

    // ��ʾ��������˶��켣
    for (size_t i = 0; i < points[1].size(); i++) {
        line(output, initial[i], points[1][i], Scalar(0, 0, 255));
        circle(output, points[1][i], 3, Scalar(0, 255, 0), -1);
    }

    // �ѵ�ǰ���ٽ����Ϊ��һ�˲ο�
    swap(points[1], points[0]);
    swap(gray_prev, gray);
    imshow(window_name, output);
}

//-------------------------------------------------------------------------------------------------
// function: addNewPoints
// brief: ����µ��Ƿ�Ӧ�ñ����
// parameter:
// return: �Ƿ���ӱ�־
//-------------------------------------------------------------------------------------------------
bool addNewPoints()
{
    return points[0].size() <= 10;
}

//-------------------------------------------------------------------------------------------------
// function: acceptTrackedPoint
// brief: ������Щ���ٵ㱻����
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

//-----------------------------------���궨�岿�֡�--------------------------------------------
//  ����������һЩ������
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "���������Ĵ���&���Ի��ʾ����"        //Ϊ���ڱ��ⶨ��ĺ� 


//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//      ������ȫ�ֱ�������
//-----------------------------------------------------------------------------------------------
const int g_nMaxAlphaValue = 100;//Alphaֵ�����ֵ
int g_nAlphaValueSlider;//��������Ӧ�ı���
double g_dAlphaValue;
double g_dBetaValue;

//�����洢ͼ��ı���
Mat g_srcImage1;
Mat g_srcImage2;
Mat g_dstImage;


//-----------------------------------��on_Trackbar( )������--------------------------------
//      ��������Ӧ�������Ļص�����
//------------------------------------------------------------------------------------------
void on_Trackbar(int, void*)
{
    //�����ǰalphaֵ��������ֵ�ı���
    g_dAlphaValue = (double)g_nAlphaValueSlider / g_nMaxAlphaValue;
    //��betaֵΪ1��ȥalphaֵ
    g_dBetaValue = (1.0 - g_dAlphaValue);
    //����alpha��betaֵ�������Ի��
    addWeighted(g_srcImage1, g_dAlphaValue, g_srcImage2, g_dBetaValue, 0.0, g_dstImage);
    //��ʾЧ��ͼ
    imshow(WINDOW_NAME, g_dstImage);
}


//-----------------------------��ShowHelpText( )������--------------------------------------
//      ���������������Ϣ
//-------------------------------------------------------------------------------------------------
//-----------------------------------��ShowHelpText( )������----------------------------------
//      ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�17������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------��main( )������-----------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //��ʾ������Ϣ
    ShowHelpText();
    //����ͼ�� (��ͼ��ĳߴ�����ͬ)
    g_srcImage1 = imread("011.jpg");
    g_srcImage2 = imread("012.jpg");

    if (!g_srcImage1.data) {
        printf("��ȡ��һ��ͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����~�� \n");
        return -1;
    }

    if (!g_srcImage2.data) {
        printf("��ȡ�ڶ���ͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����~��\n");
        return -1;
    }

    //���û�������ֵΪ70
    g_nAlphaValueSlider = 70;
    //��������
    namedWindow(WINDOW_NAME, 1);
    //�ڴ����Ĵ����д���һ���������ؼ�
    char TrackbarName[50];
    sprintf(TrackbarName, "͸��ֵ %d", g_nMaxAlphaValue);
    createTrackbar(TrackbarName, WINDOW_NAME, &g_nAlphaValueSlider, g_nMaxAlphaValue, on_Trackbar);
    //����ڻص���������ʾ
    on_Trackbar(g_nAlphaValueSlider, 0);
    //��������˳�
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET012

#include <opencv2/opencv.hpp>
using namespace cv;

//-----------------------------------���궨�岿�֡�--------------------------------------------
//  ����������һЩ������
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME "�����򴰿ڡ�"        //Ϊ���ڱ��ⶨ��ĺ� 


//-----------------------------------��ȫ�ֺ����������֡�------------------------------------
//      ������ȫ�ֺ���������
//------------------------------------------------------------------------------------------------
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawRectangle(cv::Mat& img, cv::Rect box);
void ShowHelpText();

//-----------------------------------��ȫ�ֱ����������֡�-----------------------------------
//      ������ȫ�ֱ���������
//-----------------------------------------------------------------------------------------------
Rect g_rectangle;
bool g_bDrawingBox = false;//�Ƿ���л���
RNG g_rng(12345);



//-----------------------------------��main( )������--------------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //��0���ı�console������ɫ
    system("color 9F");
    //��0����ʾ��ӭ�Ͱ�������
    ShowHelpText();
    //��1��׼������
    g_rectangle = Rect(-1, -1, 0, 0);
    Mat srcImage(600, 800, CV_8UC3), tempImage;
    srcImage.copyTo(tempImage);
    g_rectangle = Rect(-1, -1, 0, 0);
    srcImage = Scalar::all(0);
    //��2�������������ص�����
    namedWindow(WINDOW_NAME);
    setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);

    //��3��������ѭ���������л��Ƶı�ʶ��Ϊ��ʱ�����л���
    while (1) {
        srcImage.copyTo(tempImage);//����Դͼ����ʱ����

        if (g_bDrawingBox) DrawRectangle(tempImage, g_rectangle);//�����л��Ƶı�ʶ��Ϊ�棬����л���

        imshow(WINDOW_NAME, tempImage);

        if (waitKey(10) == 27) break;//����ESC���������˳�
    }

    return 0;
}



//--------------------------------��on_MouseHandle( )������-----------------------------
//      ���������ص����������ݲ�ͬ������¼����в�ͬ�Ĳ���
//-----------------------------------------------------------------------------------------------
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
    Mat& image = *(cv::Mat*) param;

    switch (event) {
    //����ƶ���Ϣ
    case EVENT_MOUSEMOVE: {
        if (g_bDrawingBox) { //����Ƿ���л��Ƶı�ʶ��Ϊ�棬���¼�³��Ϳ�RECT�ͱ�����
            g_rectangle.width = x - g_rectangle.x;
            g_rectangle.height = y - g_rectangle.y;
        }
    }
    break;

    //���������Ϣ
    case EVENT_LBUTTONDOWN: {
        g_bDrawingBox = true;
        g_rectangle = Rect(x, y, 0, 0);//��¼��ʼ��
    }
    break;

    //���̧����Ϣ
    case EVENT_LBUTTONUP: {
        g_bDrawingBox = false;//�ñ�ʶ��Ϊfalse

        //�Կ�͸�С��0�Ĵ���
        if (g_rectangle.width < 0) {
            g_rectangle.x += g_rectangle.width;
            g_rectangle.width *= -1;
        }

        if (g_rectangle.height < 0) {
            g_rectangle.y += g_rectangle.height;
            g_rectangle.height *= -1;
        }

        //���ú������л���
        DrawRectangle(image, g_rectangle);
    }
    break;
    }
}



//-----------------------------------��DrawRectangle( )������------------------------------
//      �������Զ���ľ��λ��ƺ���
//-----------------------------------------------------------------------------------------------
void DrawRectangle(cv::Mat& img, cv::Rect box)
{
    cv::rectangle(img, box.tl(), box.br(), cv::Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));//�����ɫ
}


//-----------------------------------��ShowHelpText( )������-----------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�18������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\n\n\t��ӭ��������꽻����ʾ��ʾ������\n");
    printf("\n\n\t���ڴ����е�����������϶��Ի��ƾ���\n");
}

#endif


#if SNIPPET013

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;



//-----------------------------��ShowHelpText( )������--------------------------------------
//      ���������������Ϣ
//-------------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�19������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\n\n\t��ӭ����������ͼ������-Mat�ࡿʾ������~\n\n");
    printf("\n\n\t����˵����\n\n\t��ʾ������������ʾMat��ĸ�ʽ��������ܣ��������Ϊ��");
    printf("\n\n\n\t��1��OpenCVĬ�Ϸ��");
    printf("\n\n\t��2��Python���");
    printf("\n\n\t��3�����ŷָ����");
    printf("\n\n\t��4��Numpy���");
    printf("\n\n\t��5��C���Է��\n\n");
    printf("\n  --------------------------------------------------------------------------\n");
}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main(int, char**)
{
    //�ı����̨��ǰ��ɫ�ͱ���ɫ
    system("color 8F");
    //��ʾ��������
    ShowHelpText();
    Mat I = Mat::eye(4, 4, CV_64F);
    I.at<double>(1, 1) = CV_PI;
    cout << "\nI = " << I << ";\n" << endl;
    Mat r = Mat(10, 3, CV_8UC3);
    randu(r, Scalar::all(0), Scalar::all(255));
    //�˶δ����OpenCV2��Ϊ��
    //cout << "r (OpenCVĬ�Ϸ��) = " << r << ";" << endl << endl;
    //cout << "r (Python���) = " << format(r,"python") << ";" << endl << endl;
    //cout << "r (Numpy���) = " << format(r,"numpy") << ";" << endl << endl;
    //cout << "r (���ŷָ����) = " << format(r,"csv") << ";" << endl<< endl;
    //cout << "r (C���Է��) = " << format(r,"C") << ";" << endl << endl;
    //�˶δ����OpenCV3��Ϊ��
    cout << "r (OpenCVĬ�Ϸ��) = " << r << ";" << endl << endl;
    cout << "r (Python���) = " << format(r, Formatter::FMT_PYTHON) << ";" << endl << endl;
    cout << "r (Numpy���) = " << format(r, Formatter::FMT_NUMPY) << ";" << endl << endl;
    cout << "r (���ŷָ����) = " << format(r, Formatter::FMT_CSV) << ";" << endl << endl;
    cout << "r (C���Է��) = " << format(r, Formatter::FMT_C) << ";" << endl << endl;
    Point2f p(6, 2);
    cout << "��2ά�㡿p = " << p << ";\n" << endl;
    Point3f p3f(8, 2, 0);
    cout << "��3ά�㡿p3f = " << p3f << ";\n" << endl;
    vector<float> v;
    v.push_back(3);
    v.push_back(5);
    v.push_back(7);
    cout << "������Mat��vector��shortvec = " << Mat(v) << ";\n" << endl;
    vector<Point2f> points(20);

    for (size_t i = 0; i < points.size(); ++i)
        points[i] = Point2f((float)(i * 5), (float)(i % 7));

    cout << "����ά��������points = " << points << ";";
    getchar();//��������˳�
    return 0;
}


#endif


#if SNIPPET014

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

//�˳������OpenCV3����Ҫ�������ͷ�ļ���
#include <opencv2/imgproc/imgproc.hpp>



//-----------------------------------���궨�岿�֡�--------------------------------------------
//      ����������һЩ������
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "������ͼ1��"        //Ϊ���ڱ��ⶨ��ĺ� 
#define WINDOW_NAME2 "������ͼ2��"        //Ϊ���ڱ��ⶨ��ĺ� 
#define WINDOW_WIDTH 600//���崰�ڴ�С�ĺ�



//--------------------------------��ȫ�ֺ����������֡�-------------------------------------
//      ������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
void DrawEllipse(Mat img, double angle);//������Բ
void DrawFilledCircle(Mat img, Point center);//����Բ
void DrawPolygon(Mat img);//���ƶ����
void DrawLine(Mat img, Point start, Point end);//�����߶�



//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�20������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}




//---------------------------------------��main( )������--------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main(void)
{
    // �����հ׵�Matͼ��
    Mat atomImage = Mat::zeros(WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3);
    Mat rookImage = Mat::zeros(WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3);
    ShowHelpText();
    // ---------------------<1>���ƻ�ѧ�е�ԭ��ʾ��ͼ------------------------
    //��1.1���Ȼ��Ƴ���Բ
    DrawEllipse(atomImage, 90);
    DrawEllipse(atomImage, 0);
    DrawEllipse(atomImage, 45);
    DrawEllipse(atomImage, -45);
    //��1.2���ٻ���Բ��
    DrawFilledCircle(atomImage, Point(WINDOW_WIDTH / 2, WINDOW_WIDTH / 2));
    // ----------------------------<2>�������ͼ-----------------------------
    //��2.1���Ȼ��Ƴ���Բ
    DrawPolygon(rookImage);
    // ��2.2�����ƾ���
    rectangle(rookImage,
              Point(0, 7 * WINDOW_WIDTH / 8),
              Point(WINDOW_WIDTH, WINDOW_WIDTH),
              Scalar(0, 255, 255),
              -1,
              8);
    // ��2.3������һЩ�߶�
    DrawLine(rookImage, Point(0, 15 * WINDOW_WIDTH / 16), Point(WINDOW_WIDTH, 15 * WINDOW_WIDTH / 16));
    DrawLine(rookImage, Point(WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8), Point(WINDOW_WIDTH / 4, WINDOW_WIDTH));
    DrawLine(rookImage, Point(WINDOW_WIDTH / 2, 7 * WINDOW_WIDTH / 8), Point(WINDOW_WIDTH / 2, WINDOW_WIDTH));
    DrawLine(rookImage, Point(3 * WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8), Point(3 * WINDOW_WIDTH / 4, WINDOW_WIDTH));
    // ---------------------------<3>��ʾ���Ƴ���ͼ��------------------------
    imshow(WINDOW_NAME1, atomImage);
    moveWindow(WINDOW_NAME1, 0, 200);
    imshow(WINDOW_NAME2, rookImage);
    moveWindow(WINDOW_NAME2, WINDOW_WIDTH, 200);
    waitKey(0);
    return (0);
}



//-------------------------------��DrawEllipse( )������--------------------------------
//      �������Զ���Ļ��ƺ�����ʵ���˻��Ʋ�ͬ�Ƕȡ���ͬ�ߴ����Բ
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


//-----------------------------------��DrawFilledCircle( )������---------------------------
//      �������Զ���Ļ��ƺ�����ʵ����ʵ��Բ�Ļ���
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


//-----------------------------------��DrawPolygon( )������--------------------------
//      �������Զ���Ļ��ƺ�����ʵ���˰�����εĻ���
//--------------------------------------------------------------------------------------
void DrawPolygon(Mat img)
{
    int lineType = 8;
    //����һЩ��
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


//-----------------------------------��DrawLine( )������--------------------------
//      �������Զ���Ļ��ƺ�����ʵ�����ߵĻ���
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

//-----------------------------------��ȫ�ֺ����������֡�-----------------------------------
//          ������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div);
void ShowHelpText();



//--------------------------------------��main( )������---------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    //��1������ԭʼͼ����ʾ
    Mat srcImage = imread("001.jpg");
    imshow("ԭʼͼ��", srcImage);
    //��2����ԭʼͼ�Ĳ����������������Ч��ͼ
    Mat dstImage;
    dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//Ч��ͼ�Ĵ�С��������ԭͼƬ��ͬ
    ShowHelpText();
    //��3����¼��ʼʱ��
    double time0 = static_cast<double>(getTickCount());
    //��4��������ɫ�ռ���������
    colorReduce(srcImage, dstImage, 32);
    //��5����������ʱ�䲢���
    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    cout << "\t�˷�������ʱ��Ϊ�� " << time0 << "��" << endl;  //�������ʱ��
    //��6����ʾЧ��ͼ
    imshow("Ч��ͼ", dstImage);
    waitKey(0);
}


//---------------------------------��colorReduce( )������---------------------------------
//          ������ʹ�á�ָ����ʣ�C������[ ]�����������ɫ�ռ���������
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
    //����׼��
    outputImage = inputImage.clone();  //����ʵ�ε���ʱ����
    int rowNumber = outputImage.rows;  //����
    int colNumber = outputImage.cols * outputImage.channels(); //���� x ͨ����=ÿһ��Ԫ�صĸ���

    //˫��ѭ�����������е�����ֵ
    for (int i = 0; i < rowNumber; i++) { //��ѭ��
        uchar* data = outputImage.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ

        for (int j = 0; j < colNumber; j++) { //��ѭ��
            // ---------����ʼ����ÿ�����ء�-------------
            data[j] = data[j] / div * div + div / 2;
            // ----------�����������---------------------
        }  //�д������
    }
}


//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�21������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}

#endif



#if SNIPPET016

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;



//-----------------------------------��ȫ�ֺ����������֡�-----------------------------------
//      ������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div);
void ShowHelpText();



//--------------------------------------��main( )������--------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    //��1������ԭʼͼ����ʾ
    Mat srcImage = imread("1.jpg");
    imshow("ԭʼͼ��", srcImage);
    //��2����ԭʼͼ�Ĳ����������������Ч��ͼ
    Mat dstImage;
    dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//Ч��ͼ�Ĵ�С��������ԭͼƬ��ͬ
    ShowHelpText();
    //��3����¼��ʼʱ��
    double time0 = static_cast<double>(getTickCount());
    //��4��������ɫ�ռ���������
    colorReduce(srcImage, dstImage, 32);
    //��5����������ʱ�䲢���
    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    cout << "�˷�������ʱ��Ϊ�� " << time0 << "��" << endl;  //�������ʱ��
    //��6����ʾЧ��ͼ
    imshow("Ч��ͼ", dstImage);
    waitKey(0);
}




//-------------------------------------��colorReduce( )������-----------------------------
//      ������ʹ�á������������������ɫ�ռ���������
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
    //����׼��
    outputImage = inputImage.clone();  //����ʵ�ε���ʱ����
    //��ȡ������
    Mat_<Vec3b>::iterator it = outputImage.begin<Vec3b>();  //��ʼλ�õĵ�����
    Mat_<Vec3b>::iterator itend = outputImage.end<Vec3b>();  //��ֹλ�õĵ�����

    //��ȡ��ɫͼ������
    for (; it != itend; ++it) {
        // ------------------------����ʼ����ÿ�����ء�--------------------
        (*it)[0] = (*it)[0] / div * div + div / 2;
        (*it)[1] = (*it)[1] / div * div + div / 2;
        (*it)[2] = (*it)[2] / div * div + div / 2;
        // ------------------------�����������----------------------------
    }
}



//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�22������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}



#endif


#if SNIPPET017

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

//-----------------------------------��ȫ�ֺ����������֡�-----------------------------------
//          ������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div);
void ShowHelpText();


//--------------------------------------��main( )������---------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    system("color 9F");
    //��1������ԭʼͼ����ʾ
    Mat srcImage = imread("1.jpg");
    imshow("ԭʼͼ��", srcImage);
    //��2����ԭʼͼ�Ĳ����������������Ч��ͼ
    Mat dstImage;
    dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//Ч��ͼ�Ĵ�С��������ԭͼƬ��ͬ
    ShowHelpText();
    //��3����¼��ʼʱ��
    double time0 = static_cast<double>(getTickCount());
    //��4��������ɫ�ռ���������
    colorReduce(srcImage, dstImage, 32);
    //��5����������ʱ�䲢���
    time0 = ((double)getTickCount() - time0) / getTickFrequency();
    cout << "�˷�������ʱ��Ϊ�� " << time0 << "��" << endl;  //�������ʱ��
    //��6����ʾЧ��ͼ
    imshow("Ч��ͼ", dstImage);
    waitKey(0);
}


//----------------------------------��colorReduce( )������-------------------------------
//          ������ʹ�á���̬��ַ�������at�������汾����ɫ�ռ���������
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
    //����׼��
    outputImage = inputImage.clone();  //����ʵ�ε���ʱ����
    int rowNumber = outputImage.rows;  //����
    int colNumber = outputImage.cols;  //����

    //��ȡ��ɫͼ������
    for (int i = 0; i < rowNumber; i++) {
        for (int j = 0; j < colNumber; j++) {
            // ------------------------����ʼ����ÿ�����ء�--------------------
            outputImage.at<Vec3b>(i, j)[0] = outputImage.at<Vec3b>(i, j)[0] / div * div + div / 2;  //��ɫͨ��
            outputImage.at<Vec3b>(i, j)[1] = outputImage.at<Vec3b>(i, j)[1] / div * div + div / 2;  //��ɫͨ��
            outputImage.at<Vec3b>(i, j)[2] = outputImage.at<Vec3b>(i, j)[2] / div * div + div / 2;  //����ͨ��
            // -------------------------�����������----------------------------
        }  // �д������
    }
}


//-------------------------------��ShowHelpText( )������--------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�23������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


#endif


#if SNIPPET018

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;



//---------------------------------���궨�岿�֡�---------------------------------------------
//      ����������������ʹ�ú궨��
//-------------------------------------------------------------------------------------------------
#define NTESTS 14
#define NITERATIONS 20



//----------------------------------------- ������һ��-------------------------------------------
//      ˵��������.ptr �� []
//-------------------------------------------------------------------------------------------------
void colorReduce0(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols * image.channels(); //ÿ��Ԫ�ص���Ԫ������

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            data[i] = data[i] / div * div + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}

//-----------------------------------����������-------------------------------------------------
//      ˵�������� .ptr �� * ++
//-------------------------------------------------------------------------------------------------
void colorReduce1(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols * image.channels(); //ÿ��Ԫ�ص���Ԫ������

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            *data++ = *data / div * div + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}

//-----------------------------------------����������-------------------------------------------
//      ˵��������.ptr �� * ++ �Լ�ģ����
//-------------------------------------------------------------------------------------------------
void colorReduce2(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols * image.channels(); //ÿ��Ԫ�ص���Ԫ������

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            int v = *data;
            *data++ = v - v % div + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}

//----------------------------------------�������ġ�---------------------------------------------
//      ˵��������.ptr �� * ++ �Լ�λ����
//----------------------------------------------------------------------------------------------------
void colorReduce3(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols * image.channels(); //ÿ��Ԫ�ص���Ԫ������
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //����ֵ
    uchar mask = 0xFF << n; // e.g. ���� div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //------------��ʼ����ÿ������-------------------
            *data++ = *data & mask + div / 2;
            //-------------�������ش���------------------------
        }  //���д������
    }
}


//----------------------------------------�������塿----------------------------------------------
//      ˵��������ָ����������
//---------------------------------------------------------------------------------------------------
void colorReduce4(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols * image.channels(); //ÿ��Ԫ�ص���Ԫ������
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    int step = image.step; //��Ч���
    //����ֵ
    uchar mask = 0xFF << n; // e.g. ���� div=16, mask= 0xF0
    //��ȡָ��ͼ�񻺳�����ָ��
    uchar *data = image.data;

    for (int j = 0; j < nl; j++) {
        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            *(data + i) = *data & mask + div / 2;
            //-------------�������ش���------------------------
        } //���д������

        data += step;  // next line
    }
}

//---------------------------------------����������----------------------------------------------
//      ˵�������� .ptr �� * ++�Լ�λ���㡢image.cols * image.channels()
//-------------------------------------------------------------------------------------------------
void colorReduce5(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //����ֵ
    uchar mask = 0xFF << n; // e.g. ����div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < image.cols * image.channels(); i++) {
            //-------------��ʼ����ÿ������-------------------
            *data++ = *data & mask + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}

// -------------------------------------�������ߡ�----------------------------------------------
//      ˵��������.ptr �� * ++ �Լ�λ����(continuous)
//-------------------------------------------------------------------------------------------------
void colorReduce6(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols * image.channels(); //ÿ��Ԫ�ص���Ԫ������

    if (image.isContinuous()) {
        //���������
        nc = nc * nl;
        nl = 1;  // Ϊһά����
    }

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //����ֵ
    uchar mask = 0xFF << n; // e.g. ����div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            *data++ = *data & mask + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}

//------------------------------------�������ˡ�------------------------------------------------
//      ˵�������� .ptr �� * ++ �Լ�λ���� (continuous+channels)
//-------------------------------------------------------------------------------------------------
void colorReduce7(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols; //����

    if (image.isContinuous()) {
        //���������
        nc = nc * nl;
        nl = 1;  // Ϊһά����
    }

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //����ֵ
    uchar mask = 0xFF << n; // e.g. ����div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            *data++ = *data & mask + div / 2;
            *data++ = *data & mask + div / 2;
            *data++ = *data & mask + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}


// -----------------------------------�������š� ------------------------------------------------
//      ˵��������Mat_ iterator
//-------------------------------------------------------------------------------------------------
void colorReduce8(Mat &image, int div = 64)
{
    //��ȡ������
    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    Mat_<Vec3b>::iterator itend = image.end<Vec3b>();

    for (; it != itend; ++it) {
        //-------------��ʼ����ÿ������-------------------
        (*it)[0] = (*it)[0] / div * div + div / 2;
        (*it)[1] = (*it)[1] / div * div + div / 2;
        (*it)[2] = (*it)[2] / div * div + div / 2;
        //-------------�������ش���------------------------
    }//���д������
}

//-------------------------------------������ʮ��-----------------------------------------------
//      ˵��������Mat_ iterator�Լ�λ����
//-------------------------------------------------------------------------------------------------
void colorReduce9(Mat &image, int div = 64)
{
    // div������2����
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //����ֵ
    uchar mask = 0xFF << n; // e.g. ���� div=16, mask= 0xF0
    // ��ȡ������
    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    Mat_<Vec3b>::iterator itend = image.end<Vec3b>();

    //ɨ������Ԫ��
    for (; it != itend; ++it) {
        //-------------��ʼ����ÿ������-------------------
        (*it)[0] = (*it)[0] & mask + div / 2;
        (*it)[1] = (*it)[1] & mask + div / 2;
        (*it)[2] = (*it)[2] & mask + div / 2;
        //-------------�������ش���------------------------
    }//���д������
}

//------------------------------------������ʮһ��---------------------------------------------
//      ˵��������Mat Iterator_
//-------------------------------------------------------------------------------------------------
void colorReduce10(Mat &image, int div = 64)
{
    //��ȡ������
    Mat_<Vec3b> cimage = image;
    Mat_<Vec3b>::iterator it = cimage.begin();
    Mat_<Vec3b>::iterator itend = cimage.end();

    for (; it != itend; it++) {
        //-------------��ʼ����ÿ������-------------------
        (*it)[0] = (*it)[0] / div * div + div / 2;
        (*it)[1] = (*it)[1] / div * div + div / 2;
        (*it)[2] = (*it)[2] / div * div + div / 2;
        //-------------�������ش���------------------------
    }
}

//--------------------------------------������ʮ����--------------------------------------------
//      ˵�������ö�̬��ַ�������at
//-------------------------------------------------------------------------------------------------
void colorReduce11(Mat &image, int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols; //����

    for (int j = 0; j < nl; j++) {
        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            image.at<Vec3b>(j, i)[0] = image.at<Vec3b>(j, i)[0] / div * div + div / 2;
            image.at<Vec3b>(j, i)[1] = image.at<Vec3b>(j, i)[1] / div * div + div / 2;
            image.at<Vec3b>(j, i)[2] = image.at<Vec3b>(j, i)[2] / div * div + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}

//----------------------------------������ʮ����-----------------------------------------------
//      ˵��������ͼ������������
//-------------------------------------------------------------------------------------------------
void colorReduce12(const Mat &image, //����ͼ��
                   Mat &result,      // ���ͼ��
                   int div = 64)
{
    int nl = image.rows; //����
    int nc = image.cols; //����
    //׼���ó�ʼ�����Mat�����ͼ��
    result.create(image.rows, image.cols, image.type());
    //��������������ͼ��
    nc = nc * nl;
    nl = 1;  //��ά����
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //����ֵ
    uchar mask = 0xFF << n; // e.g.����div=16, mask= 0xF0

    for (int j = 0; j < nl; j++) {
        uchar* data = result.ptr<uchar>(j);
        const uchar* idata = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++) {
            //-------------��ʼ����ÿ������-------------------
            *data++ = (*idata++)&mask + div / 2;
            *data++ = (*idata++)&mask + div / 2;
            *data++ = (*idata++)&mask + div / 2;
            //-------------�������ش���------------------------
        } //���д������
    }
}

//--------------------------------------������ʮ�ġ�-------------------------------------------
//      ˵�������ò���������
//-------------------------------------------------------------------------------------------------
void colorReduce13(Mat &image, int div = 64)
{
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
    //����ֵ
    uchar mask = 0xFF << n; // e.g. ����div=16, mask= 0xF0
    //����ɫ�ʻ�ԭ
    image = (image & Scalar(mask, mask, mask)) + Scalar(div / 2, div / 2, div / 2);
}




//-----------------------------------��ShowHelpText( )������-----------------------------
//      ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�24������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    printf("\n\n���ڽ��д�ȡ���������Եȡ���\n\n");
}




//-----------------------------------��main( )������--------------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
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

    //ʱ��ֵ��Ϊ0
    for (int i = 0; i < NTESTS; i++)
        t[i] = 0;

    // ����ظ�����
    int n = NITERATIONS;

    for (int k = 0; k < n; k++) {
        cout << k << " of " << n << endl;
        image1 = imread("1.png");
        //������һ������.ptr �� []
        tinit = getTickCount();
        colorReduce0(image1);
        t[0] += getTickCount() - tinit;
        //�������������� .ptr �� * ++
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce1(image1);
        t[1] += getTickCount() - tinit;
        //��������������.ptr �� * ++ �Լ�ģ����
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce2(image1);
        t[2] += getTickCount() - tinit;
        //�������ġ� ����.ptr �� * ++ �Լ�λ����
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce3(image1);
        t[3] += getTickCount() - tinit;
        //�������塿 ����ָ�����������
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce4(image1);
        t[4] += getTickCount() - tinit;
        //�������������� .ptr �� * ++�Լ�λ���㡢image.cols * image.channels()
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce5(image1);
        t[5] += getTickCount() - tinit;
        //�������ߡ�����.ptr �� * ++ �Լ�λ����(continuous)
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce6(image1);
        t[6] += getTickCount() - tinit;
        //�������ˡ����� .ptr �� * ++ �Լ�λ���� (continuous+channels)
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce7(image1);
        t[7] += getTickCount() - tinit;
        //�������š� ����Mat_ iterator
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce8(image1);
        t[8] += getTickCount() - tinit;
        //������ʮ�� ����Mat_ iterator�Լ�λ����
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce9(image1);
        t[9] += getTickCount() - tinit;
        //������ʮһ������Mat Iterator_
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce10(image1);
        t[10] += getTickCount() - tinit;
        //������ʮ���� ���ö�̬��ַ�������at
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce11(image1);
        t[11] += getTickCount() - tinit;
        //������ʮ���� ����ͼ������������
        image1 = imread("1.png");
        tinit = getTickCount();
        Mat result;
        colorReduce12(image1, result);
        t[12] += getTickCount() - tinit;
        image2 = result;
        //������ʮ�ġ� ���ò���������
        image1 = imread("1.png");
        tinit = getTickCount();
        colorReduce13(image1);
        t[13] += getTickCount() - tinit;
        //------------------------------
    }

    //���ͼ��
    imshow("ԭʼͼ��", image0);
    imshow("���", image2);
    imshow("ͼ����", image1);
    // ���ƽ��ִ��ʱ��
    cout << endl << "-------------------------------------------" << endl << endl;
    cout << "\n������һ������.ptr �� []�ķ�������ʱ��Ϊ " << 1000.*t[0] / getTickFrequency() / n << "ms" << endl;
    cout << "\n�������������� .ptr �� * ++ �ķ�������ʱ��Ϊ" << 1000.*t[1] / getTickFrequency() / n << "ms" << endl;
    cout << "\n��������������.ptr �� * ++ �Լ�ģ�����ķ�������ʱ��Ϊ" << 1000.*t[2] / getTickFrequency() / n << "ms" << endl;
    cout << "\n�������ġ�����.ptr �� * ++ �Լ�λ�����ķ�������ʱ��Ϊ" << 1000.*t[3] / getTickFrequency() / n << "ms" << endl;
    cout << "\n�������塿����ָ����������ķ�������ʱ��Ϊ" << 1000.*t[4] / getTickFrequency() / n << "ms" << endl;
    cout << "\n�������������� .ptr �� * ++�Լ�λ���㡢channels()�ķ�������ʱ��Ϊ" << 1000.*t[5] / getTickFrequency() / n << "ms" << endl;
    cout << "\n�������ߡ�����.ptr �� * ++ �Լ�λ����(continuous)�ķ�������ʱ��Ϊ" << 1000.*t[6] / getTickFrequency() / n << "ms" << endl;
    cout << "\n�������ˡ����� .ptr �� * ++ �Լ�λ���� (continuous+channels)�ķ�������ʱ��Ϊ" << 1000.*t[7] / getTickFrequency() / n << "ms" << endl;
    cout << "\n�������š�����Mat_ iterator �ķ�������ʱ��Ϊ" << 1000.*t[8] / getTickFrequency() / n << "ms" << endl;
    cout << "\n������ʮ������Mat_ iterator�Լ�λ����ķ�������ʱ��Ϊ" << 1000.*t[9] / getTickFrequency() / n << "ms" << endl;
    cout << "\n������ʮһ������Mat Iterator_�ķ�������ʱ��Ϊ" << 1000.*t[10] / getTickFrequency() / n << "ms" << endl;
    cout << "\n������ʮ�������ö�̬��ַ�������at �ķ�������ʱ��Ϊ" << 1000.*t[11] / getTickFrequency() / n << "ms" << endl;
    cout << "\n������ʮ��������ͼ�������������ķ�������ʱ��Ϊ" << 1000.*t[12] / getTickFrequency() / n << "ms" << endl;
    cout << "\n������ʮ�ġ����ò��������صķ�������ʱ��Ϊ" << 1000.*t[13] / getTickFrequency() / n << "ms" << endl;
    waitKey();
    return 0;
}



#endif


#if SNIPPET019

#include "opencv2/opencv.hpp"
#include <time.h>
using namespace cv;


//-----------------------------------��ShowHelpText( )������----------------------------------
//       ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�29������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//-----------------------------------��main( )������--------------------------------------------
//  ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
//-----------------------------------------------------------------------------------------------
int main()
{
    //�ı�console������ɫ
    system("color 5F");
    ShowHelpText();
    //��ʼ��
    FileStorage fs("test.yaml", FileStorage::WRITE);
    //��ʼ�ļ�д��
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
    printf("\n�ļ���д��ϣ����ڹ���Ŀ¼�²鿴���ɵ��ļ�~");
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
    // ԭͼ
    Mat img = imread("020.jpg");
    // �����˲�
    Mat box_filter_img;
    boxFilter(img, box_filter_img, -1, Size(5, 5));
    // ��ֵ�˲�
    Mat mean_blur_img;
    blur(img, mean_blur_img, Size(7, 7));
    // ��˹�˲�
    Mat gaussian_blur_img;
    GaussianBlur(img, gaussian_blur_img, Size(5, 5), 0, 0);
    // ��ֵ�˲�
    Mat media_blur_img;
    medianBlur(img, media_blur_img, 7);
    // ˫���˲�
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


//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//  ������ȫ�ֱ�������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage1, g_dstImage2, g_dstImage3;//�洢ͼƬ��Mat����
int g_nBoxFilterValue = 3;  //�����˲�����ֵ
int g_nMeanBlurValue = 3;  //��ֵ�˲�����ֵ
int g_nGaussianBlurValue = 3;  //��˹�˲�����ֵ


//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//  ������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
//�ĸ��켣���Ļص�����
static void on_BoxFilter(int, void *);      //��ֵ�˲�
static void on_MeanBlur(int, void *);       //��ֵ�˲�
static void on_GaussianBlur(int, void *);           //��˹�˲�
void ShowHelpText();


//-----------------------------------��main( )������--------------------------------------------
//  ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
//-----------------------------------------------------------------------------------------------
int main()
{
    //�ı�console������ɫ
    system("color 5F");
    //�����������
    ShowHelpText();
    // ����ԭͼ
    g_srcImage = imread("020.jpg", 1);

    if (!g_srcImage.data) {
        printf("Oh��no����ȡsrcImage����~�� \n");
        return false;
    }

    //��¡ԭͼ������Mat������
    g_dstImage1 = g_srcImage.clone();
    g_dstImage2 = g_srcImage.clone();
    g_dstImage3 = g_srcImage.clone();
    //��ʾԭͼ
    namedWindow("��<0>ԭͼ���ڡ�", 1);
    imshow("��<0>ԭͼ���ڡ�", g_srcImage);
    //=================��<1>�����˲���==================
    //��������
    namedWindow("��<1>�����˲���", 1);
    //�����켣��
    createTrackbar("�ں�ֵ��", "��<1>�����˲���", &g_nBoxFilterValue, 40, on_BoxFilter);
    on_BoxFilter(g_nBoxFilterValue, 0);
    //================================================
    //=================��<2>��ֵ�˲���==================
    //��������
    namedWindow("��<2>��ֵ�˲���", 1);
    //�����켣��
    createTrackbar("�ں�ֵ��", "��<2>��ֵ�˲���", &g_nMeanBlurValue, 40, on_MeanBlur);
    on_MeanBlur(g_nMeanBlurValue, 0);
    //================================================
    //=================��<3>��˹�˲���=====================
    //��������
    namedWindow("��<3>��˹�˲���", 1);
    //�����켣��
    createTrackbar("�ں�ֵ��", "��<3>��˹�˲���", &g_nGaussianBlurValue, 40, on_GaussianBlur);
    on_GaussianBlur(g_nGaussianBlurValue, 0);
    //================================================
    //���һЩ������Ϣ
    cout << endl << "\t���гɹ���������������۲�ͼ��Ч��~\n\n"
         << "\t���¡�q����ʱ�������˳���\n";

    //���¡�q����ʱ�������˳�
    while (char(waitKey(1)) != 'q') {}

    return 0;
}


//-----------------------------��on_BoxFilter( )������------------------------------------
//  �����������˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_BoxFilter(int, void *)
{
    //�����˲�����
    boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
    //��ʾ����
    imshow("��<1>�����˲���", g_dstImage1);
}


//-----------------------------��on_MeanBlur( )������------------------------------------
//  ��������ֵ�˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_MeanBlur(int, void *)
{
    //��ֵ�˲�����
    blur(g_srcImage, g_dstImage2, Size(g_nMeanBlurValue + 1, g_nMeanBlurValue + 1), Point(-1, -1));
    //��ʾ����
    imshow("��<2>��ֵ�˲���", g_dstImage2);
}


//-----------------------------��ContrastAndBright( )������------------------------------------
//  ��������˹�˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_GaussianBlur(int, void *)
{
    //��˹�˲�����
    GaussianBlur(g_srcImage, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
    //��ʾ����
    imshow("��<3>��˹�˲���", g_dstImage3);
}


//-----------------------------------��ShowHelpText( )������----------------------------------
//       ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�34������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
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
    //���гߴ��������
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
    // ��1������ԭʼͼ���ұ����Զ�ֵͼģʽ����
    Mat srcImage = imread("023.jpg", 0);
    imshow("ԭʼͼ", srcImage);
    //��2����ʼ�����ͼ
    Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
    //��3��srcImageȡ������ֵ119���ǲ���
    srcImage = srcImage > 119;
    imshow("ȡ��ֵ���ԭʼͼ", srcImage);
    //��4�����������Ͳ�νṹ
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //��5����������
    //�˾�����OpenCV2��Ϊ��
    //findContours( srcImage, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    //�˾�����OpenCV3��Ϊ��
    findContours(srcImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    // ��6���������ж���������� �������ɫ���Ƴ�ÿ�����������ɫ
    int index = 0;

    for (; index >= 0; index = hierarchy[index][0]) {
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        //�˾�����OpenCV2��Ϊ��
        //drawContours( dstImage, contours, index, color, CV_FILLED, 8, hierarchy );
        //�˾�����OpenCV3��Ϊ��
        drawContours(dstImage, contours, index, color, FILLED, 8, hierarchy);
    }

    //��7����ʾ��������ͼ
    imshow("����ͼ", dstImage);
    waitKey(0);
}

#endif


#if SNIPPET024

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;


//-----------------------------------���궨�岿�֡�--------------------------------------------
//      ����������һЩ������
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "��ԭʼͼ���ڡ�"           //Ϊ���ڱ��ⶨ��ĺ� 
#define WINDOW_NAME2 "������ͼ��"                   //Ϊ���ڱ��ⶨ��ĺ� 


//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//      ������ȫ�ֱ���������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage;
Mat g_grayImage;
int g_nThresh = 80;
int g_nThresh_max = 255;
RNG g_rng(12345);
Mat g_cannyMat_output;
vector<vector<Point>> g_vContours;
vector<Vec4i> g_vHierarchy;


//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//      ������ȫ�ֺ���������
//-----------------------------------------------------------------------------------------------
static void ShowHelpText();
void on_ThreshChange(int, void*);


//-----------------------------------��main( )������--------------------------------------------
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //��0���ı�console������ɫ
    system("color 1F");
    //��0����ʾ��ӭ�Ͱ�������
    ShowHelpText();
    // ����Դͼ��
    g_srcImage = imread("024.jpg", 1);

    if (!g_srcImage.data) {
        printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n");
        return false;
    }

    // ת�ɻҶȲ�ģ��������
    cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
    blur(g_grayImage, g_grayImage, Size(3, 3));
    // ��������
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    imshow(WINDOW_NAME1, g_srcImage);
    //��������������ʼ��
    createTrackbar("canny��ֵ", WINDOW_NAME1, &g_nThresh, g_nThresh_max, on_ThreshChange);
    on_ThreshChange(0, 0);
    waitKey(0);
    return (0);
}

//-----------------------------------��on_ThreshChange( )������------------------------------
//      �������ص�����
//----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*)
{
    // ��Canny���Ӽ���Ե
    Canny(g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh * 2, 3);
    // Ѱ������
    findContours(g_cannyMat_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // �������
    Mat drawing = Mat::zeros(g_cannyMat_output.size(), CV_8UC3);

    for (int i = 0; i < g_vContours.size(); i++) {
        Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//����ֵ
        drawContours(drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point());
    }

    // ��ʾЧ��ͼ
    imshow(WINDOW_NAME2, drawing);
}


//-----------------------------------��ShowHelpText( )������----------------------------------
//      ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�70������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\n\t��ӭ��������ͼ����Ѱ��������ʾ������~\n\n");
    printf("\n\n\t��������˵��: \n\n"
           "\t\t���̰��������- �˳�����\n\n"
           "\t\t����������-�ı���ֵ\n");
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

//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�71������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\t��ӭ������͹����⡿ʾ������~\n\n");
    printf("\n\t��������˵��: \n\n"
           "\t\t���̰�����ESC������Q������q��- �˳�����\n\n"
           "\t\t���̰�������� - ������������㣬������͹�����\n");
}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    //�ı�console������ɫ
    system("color 1F");
    //��ʾ��������
    ShowHelpText();
    //��ʼ�����������ֵ
    Mat image(600, 600, CV_8UC3);
    RNG& rng = theRNG();

    //ѭ��������ESC,Q,q�������˳��������м����±�һֱ����
    while (1) {
        //������ʼ��
        char key;//��ֵ
        int count = (unsigned)rng % 100 + 3;//������ɵ������
        vector<Point> points; //��ֵ

        //������ɵ�����
        for (int i = 0; i < count; i++) {
            Point point;
            point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
            point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
            points.push_back(point);
        }

        //���͹��
        vector<int> hull;
        convexHull(Mat(points), hull, true);
        //���Ƴ������ɫ�ĵ�
        image = Scalar::all(0);

        for (int i = 0; i < count; i++)
            circle(image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA);

        //׼������
        int hullcount = (int)hull.size();//͹���ı���
        Point point0 = points[hull[hullcount - 1]];//����͹���ߵ������

        //����͹���ı�
        for (int i = 0; i < hullcount; i++) {
            Point point = points[hull[i]];
            line(image, point0, point, Scalar(255, 255, 255), 2, LINE_AA);
            point0 = point;
        }

        //��ʾЧ��ͼ
        imshow("͹�����ʾ��", image);
        //����ESC,Q,����q�������˳�
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


//-----------------------------------���궨�岿�֡�--------------------------------------------
//  ����������һЩ������
//------------------------------------------------------------------------------------------------
#define WINDOW_NAME1 "��ԭʼͼ���ڡ�"                   //Ϊ���ڱ��ⶨ��ĺ� 
#define WINDOW_NAME2 "��Ч��ͼ���ڡ�"                   //Ϊ���ڱ��ⶨ��ĺ� 



//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//  ������ȫ�ֱ���������
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


//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//   ������ȫ�ֺ���������
//-----------------------------------------------------------------------------------------------
static void ShowHelpText();
void on_ThreshChange(int, void*);
void ShowHelpText();

//-----------------------------------��main( )������------------------------------------------
//   ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    system("color 3F");
    ShowHelpText();
    // ����Դͼ��
    g_srcImage = imread("027.jpg", 1);
    // ��ԭͼת���ɻҶ�ͼ������ģ����
    cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
    blur(g_grayImage, g_grayImage, Size(3, 3));
    // ����ԭͼ���ڲ���ʾ
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    imshow(WINDOW_NAME1, g_srcImage);
    //����������
    createTrackbar(" ��ֵ:", WINDOW_NAME1, &g_nThresh, g_maxThresh, on_ThreshChange);
    on_ThreshChange(0, 0);//����һ�ν��г�ʼ��
    waitKey(0);
    return (0);
}

//-----------------------------------��thresh_callback( )������----------------------------------
//      �������ص�����
//----------------------------------------------------------------------------------------------
void on_ThreshChange(int, void*)
{
    // ��ͼ����ж�ֵ����������ֵ
    threshold(g_grayImage, g_thresholdImage_output, g_nThresh, 255, THRESH_BINARY);
    // Ѱ������
    findContours(g_thresholdImage_output, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // ����ÿ��������Ѱ����͹��
    vector<vector<Point> >hull(g_vContours.size());

    for (unsigned int i = 0; i < g_vContours.size(); i++) {
        convexHull(Mat(g_vContours[i]), hull[i], false);
    }

    // �����������͹��
    Mat drawing = Mat::zeros(g_thresholdImage_output.size(), CV_8UC3);

    for (unsigned int i = 0; i < g_vContours.size(); i++) {
        Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
        drawContours(drawing, g_vContours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
        drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
    }

    // ��ʾЧ��ͼ
    imshow(WINDOW_NAME2, drawing);
}


//-----------------------------------��ShowHelpText( )������-----------------------------
//       ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�72������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}

#endif


#if SNIPPET028

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;



//-----------------------------------��ShowHelpText( )������-----------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�73������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\n\n\t\t\t��ӭ���������ΰ�Χʾ����ʾ������~\n\n");
    printf("\n\n\t��������˵��: \n\n"
           "\t\t���̰�����ESC������Q������q��- �˳�����\n\n"
           "\t\t���̰�������� - ������������㣬��Ѱ����С����İ�Χ����\n");
}

int main()
{
    //�ı�console������ɫ
    system("color 1F");
    //��ʾ��������
    ShowHelpText();
    //��ʼ�����������ֵ
    Mat image(600, 600, CV_8UC3);
    RNG& rng = theRNG();

    //ѭ��������ESC,Q,q�������˳��������м����±�һֱ����
    while (1) {
        //������ʼ��
        int count = rng.uniform(3, 103);//������ɵ������
        vector<Point> points;//��ֵ

        //������ɵ�����
        for (int i = 0; i < count; i++) {
            Point point;
            point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
            point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
            points.push_back(point);
        }

        //�Ը����� 2D �㼯��Ѱ����С����İ�Χ����
        RotatedRect box = minAreaRect(Mat(points));
        Point2f vertex[4];
        box.points(vertex);
        //���Ƴ������ɫ�ĵ�
        image = Scalar::all(0);

        for (int i = 0; i < count; i++)
            circle(image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA);

        //���Ƴ���С����İ�Χ����
        for (int i = 0; i < 4; i++)
            line(image, vertex[i], vertex[(i + 1) % 4], Scalar(100, 200, 211), 2, LINE_AA);

        //��ʾ����
        imshow("���ΰ�Χʾ��", image);
        //����ESC,Q,����q�������˳�
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

//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�13������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\n\t\t\t��ӭ������Ѱ����С����İ�ΧԲ��ʾ������~\n");
    printf("\n\n\t��������˵��: \n\n"
           "\t\t���̰�����ESC������Q������q��- �˳�����\n\n"
           "\t\t���̰�������� - ������������㣬��Ѱ����С����İ�ΧԲ\n");
}

int main()
{
    //�ı�console������ɫ
    system("color 1F");
    //��ʾ��������
    ShowHelpText();
    //��ʼ�����������ֵ
    Mat image(600, 600, CV_8UC3);
    RNG& rng = theRNG();

    //ѭ��������ESC,Q,q�������˳��������м����±�һֱ����
    while (1) {
        //������ʼ��
        int count = rng.uniform(3, 103);//������ɵ������
        vector<Point> points;//��ֵ

        //������ɵ�����
        for (int i = 0; i < count; i++) {
            Point point;
            point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
            point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
            points.push_back(point);
        }

        //�Ը����� 2D �㼯��Ѱ����С����İ�ΧԲ
        Point2f center;
        float radius = 0;
        minEnclosingCircle(Mat(points), center, radius);
        //���Ƴ������ɫ�ĵ�
        image = Scalar::all(0);

        for (int i = 0; i < count; i++)
            circle(image, points[i], 3, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), FILLED, LINE_AA);

        //���Ƴ���С����İ�ΧԲ
        circle(image, center, cvRound(radius), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2, LINE_AA);
        //��ʾ����
        imshow("Բ�ΰ�Χʾ��", image);
        //����ESC,Q,����q�������˳�
        char key = (char)waitKey();

        if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

    return 0;
}


#endif