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


#if SNIPPET030

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/* @brief �õ�H������ֱ��ͼͼ��
@param src ����ͼ��
@param histimg �����ɫֱ��ͼ
@return void ����ֵΪ��
*/
void getHistImg(const Mat src, Mat &histimg)
{
    Mat hue, hist;
    int hsize = 16;//ֱ��ͼbin�ĸ���
    float hranges[] = { 0, 180 };
    const float* phranges = hranges;
    int ch[] = { 0, 0 };
    hue.create(src.size(), src.depth());
    mixChannels(&src, 1, &hue, 1, ch, 1);//�õ�H����
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
    // ����ͼƬ
    src = imread("001.jpg");

    if (!src.data) {
        cout << "load image failed" << endl;
        return -1;
    }

    // ����
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
 * \brief λͼת Mat
 * 
 * λͼת Mat
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
 * \brief H-S��άֱ��ͼ�Ļ��� 
 * 
 * H-S��άֱ��ͼ�Ļ���
 * 
 */


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;



//-----------------------------------��ShowHelpText( )������-----------------------------
//		 ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�79������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{

    //��1������Դͼ��ת��ΪHSV��ɫģ��
    Mat srcImage, hsvImage;
    srcImage = imread("034.jpg");
    cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);

    system("color 2F");
    ShowHelpText();

    //��2������׼��
    //��ɫ������Ϊ30���ȼ��������Ͷ�����Ϊ32���ȼ�
    int hueBinNum = 30;//ɫ����ֱ��ͼֱ������
    int saturationBinNum = 32;//���Ͷȵ�ֱ��ͼֱ������
    int histSize[] = { hueBinNum, saturationBinNum };
    // ����ɫ���ı仯��ΧΪ0��179
    float hueRanges[] = { 0, 180 };
    //���履�Ͷȵı仯��ΧΪ0���ڡ��ס��ң���255����������ɫ��
    float saturationRanges[] = { 0, 256 };
    const float* ranges[] = { hueRanges, saturationRanges };
    MatND dstHist;
    //����׼����calcHist�����н������0ͨ���͵�1ͨ����ֱ��ͼ
    int channels[] = { 0, 1 };

    //��3����ʽ����calcHist������ֱ��ͼ����
    calcHist(&hsvImage,//���������
        1, //�������Ϊ1
        channels,//ͨ������
        Mat(), //��ʹ����Ĥ
        dstHist, //�����Ŀ��ֱ��ͼ
        2, //��Ҫ�����ֱ��ͼ��ά��Ϊ2
        histSize, //���ÿ��ά�ȵ�ֱ��ͼ�ߴ������
        ranges,//ÿһά��ֵ��ȡֵ��Χ����
        true, // ָʾֱ��ͼ�Ƿ���ȵı�ʶ����true��ʾ���ȵ�ֱ��ͼ
        false);//�ۼƱ�ʶ����false��ʾֱ��ͼ�����ý׶λᱻ����

    //��4��Ϊ����ֱ��ͼ׼������
    double maxValue = 0;//���ֵ
    minMaxLoc(dstHist, 0, &maxValue, 0, 0);//����������������ȫ����Сֵ�����ֵ����maxValue��
    int scale = 10;
    Mat histImg = Mat::zeros(saturationBinNum*scale, hueBinNum * 10, CV_8UC3);

    //��5��˫��ѭ��������ֱ��ͼ����
    for (int hue = 0; hue < hueBinNum; hue++)
        for (int saturation = 0; saturation < saturationBinNum; saturation++)
        {
            float binValue = dstHist.at<float>(hue, saturation);//ֱ��ͼ����ֵ
            int intensity = cvRound(binValue * 255 / maxValue);//ǿ��

            //��ʽ���л���
            rectangle(histImg, Point(hue*scale, saturation*scale),
                Point((hue + 1)*scale - 1, (saturation + 1)*scale - 1),
                Scalar::all(intensity), FILLED);
        }

    //��6����ʾЧ��ͼ
    imshow("�ز�ͼ", srcImage);
    imshow("H-S ֱ��ͼ", histImg);

    waitKey();
}

#endif


#if SNIPPET035

/*! 
 * \brief һάֱ��ͼ�Ļ���
 * 
 * һάֱ��ͼ�Ļ���
 * 
 */


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;



//-----------------------------------��ShowHelpText( )������-----------------------------
//		 ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�80������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-------------------------------------------------------------------------------------------------
int main()
{
    //��1������ԭͼ����ʾ
    Mat srcImage = imread("035.jpg", 0);
    imshow("ԭͼ", srcImage);
    if (!srcImage.data) { cout << "fail to load image" << endl; 	return 0; }

    system("color 1F");
    ShowHelpText();

    //��2���������
    MatND dstHist;       // ��cv����CvHistogram *hist = cvCreateHist
    int dims = 1;
    float hranges[] = { 0, 255 };
    const float *ranges[] = { hranges };   // ������ҪΪconst����
    int size = 256;
    int channels = 0;

    //��3������ͼ���ֱ��ͼ
    calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);    // cv ����cvCalcHist
    int scale = 1;

    Mat dstImage(size * scale, size, CV_8U, Scalar(0));
    //��4����ȡ���ֵ����Сֵ
    double minValue = 0;
    double maxValue = 0;
    minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //  ��cv���õ���cvGetMinMaxHistValue

    //��5�����Ƴ�ֱ��ͼ
    int hpt = saturate_cast<int>(0.9 * size);
    for (int i = 0; i < 256; i++)
    {
        float binValue = dstHist.at<float>(i);           //   ע��hist����float����    ����OpenCV1.0������cvQueryHistValue_1D
        int realValue = saturate_cast<int>(binValue * hpt / maxValue);
        rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
    }
    imshow("һάֱ��ͼ", dstImage);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET036

/*! 
 * \brief RGB��ɫֱ��ͼ�Ļ��� 
 * 
 * RGB��ɫֱ��ͼ�Ļ��� 
 * 
 */


#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
using namespace cv;



//-----------------------------------��ShowHelpText( )������-----------------------------
//		 ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�81������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{

    //��1�������ز�ͼ����ʾ
    Mat srcImage;
    srcImage = imread("036.jpg");
    imshow("�ز�ͼ", srcImage);

    system("color 3F");
    ShowHelpText();

    //��2������׼��
    int bins = 256;
    int hist_size[] = { bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    MatND redHist, grayHist, blueHist;
    int channels_r[] = { 0 };

    //��3������ֱ��ͼ�ļ��㣨��ɫ�������֣�
    calcHist(&srcImage, 1, channels_r, Mat(), //��ʹ����Ĥ
        redHist, 1, hist_size, ranges,
        true, false);

    //��4������ֱ��ͼ�ļ��㣨��ɫ�������֣�
    int channels_g[] = { 1 };
    calcHist(&srcImage, 1, channels_g, Mat(), // do not use mask
        grayHist, 1, hist_size, ranges,
        true, // the histogram is uniform
        false);

    //��5������ֱ��ͼ�ļ��㣨��ɫ�������֣�
    int channels_b[] = { 2 };
    calcHist(&srcImage, 1, channels_b, Mat(), // do not use mask
        blueHist, 1, hist_size, ranges,
        true, // the histogram is uniform
        false);

    //-----------------------���Ƴ���ɫֱ��ͼ------------------------
    //����׼��
    double maxValue_red, maxValue_green, maxValue_blue;
    minMaxLoc(redHist, 0, &maxValue_red, 0, 0);
    minMaxLoc(grayHist, 0, &maxValue_green, 0, 0);
    minMaxLoc(blueHist, 0, &maxValue_blue, 0, 0);
    int scale = 1;
    int histHeight = 256;
    Mat histImage = Mat::zeros(histHeight, bins * 3, CV_8UC3);

    //��ʽ��ʼ����
    for (int i = 0; i < bins; i++)
    {
        //����׼��
        float binValue_red = redHist.at<float>(i);
        float binValue_green = grayHist.at<float>(i);
        float binValue_blue = blueHist.at<float>(i);
        int intensity_red = cvRound(binValue_red*histHeight / maxValue_red);  //Ҫ���Ƶĸ߶�
        int intensity_green = cvRound(binValue_green*histHeight / maxValue_green);  //Ҫ���Ƶĸ߶�
        int intensity_blue = cvRound(binValue_blue*histHeight / maxValue_blue);  //Ҫ���Ƶĸ߶�

        //���ƺ�ɫ������ֱ��ͼ
        rectangle(histImage, Point(i*scale, histHeight - 1),
            Point((i + 1)*scale - 1, histHeight - intensity_red),
            Scalar(255, 0, 0));

        //������ɫ������ֱ��ͼ
        rectangle(histImage, Point((i + bins)*scale, histHeight - 1),
            Point((i + bins + 1)*scale - 1, histHeight - intensity_green),
            Scalar(0, 255, 0));

        //������ɫ������ֱ��ͼ
        rectangle(histImage, Point((i + bins * 2)*scale, histHeight - 1),
            Point((i + bins * 2 + 1)*scale - 1, histHeight - intensity_blue),
            Scalar(0, 0, 255));

    }

    //�ڴ�������ʾ�����ƺõ�ֱ��ͼ
    imshow("ͼ���RGBֱ��ͼ", histImage);
    waitKey(0);
    return 0;
}

#endif


#if SNIPPET037

/*! 
 * \brief ֱ��ͼ�Ա� 
 * 
 * ֱ��ͼ�Ա�
 * 
 */


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


//-----------------------------------��ShowHelpText( )������-----------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�82������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\n��ӭ������ֱ��ͼ�Աȡ�ʾ������~\n\n");

}


//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    //��0���ı�console������ɫ
    system("color 2F");

    //��1����ʾ��������
    ShowHelpText();

    //��1�����������׼ͼ����������ŶԱ�ͼ��ľ���( RGB �� HSV )
    Mat srcImage_base, hsvImage_base;
    Mat srcImage_test1, hsvImage_test1;
    Mat srcImage_test2, hsvImage_test2;
    Mat hsvImage_halfDown;

    //��2�������׼ͼ��(srcImage_base) �����Ų���ͼ��srcImage_test1��srcImage_test2������ʾ
    srcImage_base = imread("037_1.jpg", 1);
    srcImage_test1 = imread("037_2.jpg", 1);
    srcImage_test2 = imread("037_3.jpg", 1);
    //��ʾ�����3��ͼ��
    imshow("��׼ͼ��", srcImage_base);
    imshow("����ͼ��1", srcImage_test1);
    imshow("����ͼ��2", srcImage_test2);

    // ��3����ͼ����BGRɫ�ʿռ�ת���� HSVɫ�ʿռ�
    cvtColor(srcImage_base, hsvImage_base, COLOR_BGR2HSV);
    cvtColor(srcImage_test1, hsvImage_test1, COLOR_BGR2HSV);
    cvtColor(srcImage_test2, hsvImage_test2, COLOR_BGR2HSV);

    //��4������������׼ͼ���°벿�İ���ͼ��(HSV��ʽ)
    hsvImage_halfDown = hsvImage_base(Range(hsvImage_base.rows / 2, hsvImage_base.rows - 1), Range(0, hsvImage_base.cols - 1));

    //��5����ʼ������ֱ��ͼ��Ҫ��ʵ��
    // ��hueͨ��ʹ��30��bin,��saturatoinͨ��ʹ��32��bin
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue��ȡֵ��Χ��0��256, saturationȡֵ��Χ��0��180
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 180 };
    const float* ranges[] = { h_ranges, s_ranges };
    // ʹ�õ�0�͵�1ͨ��
    int channels[] = { 0, 1 };

    // ��6����������ֱ��ͼ�� MatND ���ʵ��:
    MatND baseHist;
    MatND halfDownHist;
    MatND testHist1;
    MatND testHist2;

    // ��7�������׼ͼ�����Ų���ͼ�񣬰����׼ͼ���HSVֱ��ͼ:
    calcHist(&hsvImage_base, 1, channels, Mat(), baseHist, 2, histSize, ranges, true, false);
    normalize(baseHist, baseHist, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&hsvImage_halfDown, 1, channels, Mat(), halfDownHist, 2, histSize, ranges, true, false);
    normalize(halfDownHist, halfDownHist, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&hsvImage_test1, 1, channels, Mat(), testHist1, 2, histSize, ranges, true, false);
    normalize(testHist1, testHist1, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&hsvImage_test2, 1, channels, Mat(), testHist2, 2, histSize, ranges, true, false);
    normalize(testHist2, testHist2, 0, 1, NORM_MINMAX, -1, Mat());


    //��8����˳��ʹ��4�ֶԱȱ�׼����׼ͼ���ֱ��ͼ�������ֱ��ͼ���жԱ�:
    for (int i = 0; i < 4; i++)
    {
        //����ͼ��ֱ��ͼ�ĶԱ�
        int compare_method = i;
        double base_base = compareHist(baseHist, baseHist, compare_method);
        double base_half = compareHist(baseHist, halfDownHist, compare_method);
        double base_test1 = compareHist(baseHist, testHist1, compare_method);
        double base_test2 = compareHist(baseHist, testHist2, compare_method);
        //������
        printf(" ���� [%d] ��ƥ�������£�\n\n ����׼ͼ - ��׼ͼ����%f, ����׼ͼ - ����ͼ����%f,����׼ͼ - ����ͼ1���� %f, ����׼ͼ - ����ͼ2����%f \n-----------------------------------------------------------------\n", i, base_base, base_half, base_test1, base_test2);
    }

    printf("��������");
    waitKey(0);
    return 0;
}


#endif


#if SNIPPET038

/*! 
 * \brief ����ͶӰ 
 * 
 * ����ͶӰ
 * 
 */


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;


//-----------------------------------���궨�岿�֡�-------------------------------------------- 
//  ����������һЩ������ 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "��ԭʼͼ��"        //Ϊ���ڱ��ⶨ��ĺ� 


//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//          ������ȫ�ֱ�������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_hsvImage; Mat g_hueImage;
int g_bins = 30;//ֱ��ͼ���

//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//          ������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
static void ShowHelpText();
void on_BinChange(int, void*);

//--------------------------------------��main( )������-----------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    //��0���ı�console������ɫ
    system("color 6F");

    //��0����ʾ��������
    ShowHelpText();

    //��1����ȡԴͼ�񣬲�ת���� HSV �ռ�
    g_srcImage = imread("038.jpg", 1);
    if (!g_srcImage.data) { printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����~�� \n"); return false; }
    cvtColor(g_srcImage, g_hsvImage, COLOR_BGR2HSV);

    //��2������ Hue ɫ��ͨ��
    g_hueImage.create(g_hsvImage.size(), g_hsvImage.depth());
    int ch[] = { 0, 0 };
    mixChannels(&g_hsvImage, 1, &g_hueImage, 1, ch, 1);

    //��3������ Trackbar ������bin����Ŀ
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    createTrackbar("ɫ����� ", WINDOW_NAME1, &g_bins, 180, on_BinChange);
    on_BinChange(0, 0);//����һ�γ�ʼ��

    //��4����ʾЧ��ͼ
    imshow(WINDOW_NAME1, g_srcImage);

    // �ȴ��û�����
    waitKey(0);
    return 0;
}


//-----------------------------------��on_HoughLines( )������--------------------------------
//          ��������Ӧ�������ƶ���Ϣ�Ļص�����
//---------------------------------------------------------------------------------------------
void on_BinChange(int, void*)
{
    //��1������׼��
    MatND hist;
    int histSize = MAX(g_bins, 2);
    float hue_range[] = { 0, 180 };
    const float* ranges = { hue_range };

    //��2������ֱ��ͼ����һ��
    calcHist(&g_hueImage, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
    normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

    //��3�����㷴��ͶӰ
    MatND backproj;
    calcBackProject(&g_hueImage, 1, 0, hist, backproj, &ranges, 1, true);

    //��4����ʾ����ͶӰ
    imshow("����ͶӰͼ", backproj);

    //��5������ֱ��ͼ�Ĳ���׼��
    int w = 400; int h = 400;
    int bin_w = cvRound((double)w / histSize);
    Mat histImg = Mat::zeros(w, h, CV_8UC3);

    //��6������ֱ��ͼ
    for (int i = 0; i < g_bins; i++)
    {
        rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)), Scalar(100, 123, 255), -1);
    }

    //��7����ʾֱ��ͼ����
    imshow("ֱ��ͼ", histImg);
}


//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�83������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");

    //���һЩ������Ϣ
    printf("\n\n\t��ӭ����������ͶӰ��ʾ������\n\n");
    printf("\n\t������������۲�ͼ��Ч��\n\n");

}

#endif


#if SNIPPET039

/*! 
 * \brief ģ��ƥ�� 
 * 
 * ģ��ƥ��
 * 
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


//-----------------------------------���궨�岿�֡�-------------------------------------------- 
//  ����������һЩ������ 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "��ԭʼͼƬ��"        //Ϊ���ڱ��ⶨ��ĺ� 
#define WINDOW_NAME2 "��ƥ�䴰�ڡ�"        //Ϊ���ڱ��ⶨ��ĺ� 

//-----------------------------------��ȫ�ֱ����������֡�------------------------------------
//          ������ȫ�ֱ���������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage; Mat g_templateImage; Mat g_resultImage;
int g_nMatchMethod;
int g_nMaxTrackbarNum = 5;

//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//          ������ȫ�ֺ���������
//-----------------------------------------------------------------------------------------------
void on_Matching(int, void*);
static void ShowHelpText();


//-----------------------------------��main( )������--------------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main()
{
    //��0���ı�console������ɫ
    system("color 1F");

    //��0����ʾ��������
    ShowHelpText();

    //��1������ԭͼ���ģ���
    g_srcImage = imread("039_1.jpg", 1);
    g_templateImage = imread("039_2.jpg", 1);

    //��2����������
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    namedWindow(WINDOW_NAME2, WINDOW_AUTOSIZE);

    //��3������������������һ�γ�ʼ��
    createTrackbar("����", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum, on_Matching);
    on_Matching(0, 0);

    waitKey(0);
    return 0;

}

//-----------------------------------��on_Matching( )������--------------------------------
//          �������ص�����
//-------------------------------------------------------------------------------------------
void on_Matching(int, void*)
{
    //��1�����ֲ�������ʼ��
    Mat srcImage;
    g_srcImage.copyTo(srcImage);

    //��2����ʼ�����ڽ������ľ���
    int resultImage_rows = g_srcImage.rows - g_templateImage.rows + 1;
    int resultImage_cols = g_srcImage.cols - g_templateImage.cols + 1;
    g_resultImage.create(resultImage_rows, resultImage_cols, CV_32FC1);

    //��3������ƥ��ͱ�׼��
    matchTemplate(g_srcImage, g_templateImage, g_resultImage, g_nMatchMethod);
    normalize(g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat());

    //��4��ͨ������ minMaxLoc ��λ��ƥ���λ��
    double minValue; double maxValue; Point minLocation; Point maxLocation;
    Point matchLocation;
    minMaxLoc(g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat());

    //��5�����ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ���Ÿ��ߵ�ƥ����. ������ķ���, ��ֵԽ��ƥ��Ч��Խ��
    //�˾�����OpenCV2��Ϊ��
    //if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
    //�˾�����OpenCV3��Ϊ��
    if (g_nMatchMethod == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED)
    {
        matchLocation = minLocation;
    }
    else
    {
        matchLocation = maxLocation;
    }

    //��6�����Ƴ����Σ�����ʾ���ս��
    rectangle(srcImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);
    rectangle(g_resultImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);

    imshow(WINDOW_NAME1, srcImage);
    imshow(WINDOW_NAME2, g_resultImage);

}



//-----------------------------------��ShowHelpText( )������----------------------------------
//          ���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�84������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\t��ӭ������ģ��ƥ�䡿ʾ������~\n");
    printf("\n\n\t������������۲�ͼ��Ч��\n\n");
    printf("\n\t��������Ӧ�ķ�����ֵ˵��: \n\n"
        "\t\t������0��- ƽ����ƥ�䷨(SQDIFF)\n"
        "\t\t������1��- ��һ��ƽ����ƥ�䷨(SQDIFF NORMED)\n"
        "\t\t������2��- ���ƥ�䷨(TM CCORR)\n"
        "\t\t������3��- ��һ�����ƥ�䷨(TM CCORR NORMED)\n"
        "\t\t������4��- ���ϵ��ƥ�䷨(TM COEFF)\n"
        "\t\t������5��- ��һ�����ϵ��ƥ�䷨(TM COEFF NORMED)\n");
}


#endif

#if SNIPPET040

/*! 
 * \brief cornerHarris �����÷�ʾ�� 
 * 
 * cornerHarris �����÷�ʾ�� 
 * 
 */


#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
using namespace cv;

int main()
{
    //�ԻҶ�ģʽ����ͼ����ʾ
    Mat srcImage = imread("040.jpg", 0);
    imshow("ԭʼͼ", srcImage);

    //����Harris�ǵ����ҳ��ǵ�
    Mat cornerStrength;
    cornerHarris(srcImage, cornerStrength, 2, 3, 0.01);

    //�ԻҶ�ͼ������ֵ�������õ���ֵͼ����ʾ  
    Mat harrisCorner;
    threshold(cornerStrength, harrisCorner, 0.00001, 255, THRESH_BINARY);
    imshow("�ǵ����Ķ�ֵЧ��ͼ", harrisCorner);

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


//-----------------------------------���궨�岿�֡�--------------------------------------------  
//  ����������һЩ������  
//------------------------------------------------------------------------------------------------  
#define WINDOW_NAME1 "�����򴰿�1��"        //Ϊ���ڱ��ⶨ��ĺ�  
#define WINDOW_NAME2 "�����򴰿�2��"        //Ϊ���ڱ��ⶨ��ĺ�  

//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------
//		������ȫ�ֱ�������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_srcImage1, g_grayImage;
int thresh = 30; //��ǰ��ֵ
int max_thresh = 175; //�����ֵ


//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//		������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------
void on_CornerHarris(int, void*);//�ص�����
static void ShowHelpText();

//-----------------------------------��main( )������--------------------------------------------
//		����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //��0���ı�console������ɫ
    system("color 3F");

    //��0����ʾ��������
    ShowHelpText();

    //��1������ԭʼͼ�����п�¡����
    g_srcImage = imread("041.jpg", 1);
    if (!g_srcImage.data) { printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n"); return false; }
    imshow("ԭʼͼ", g_srcImage);
    g_srcImage1 = g_srcImage.clone();

    //��2������һ�ŻҶ�ͼ
    cvtColor(g_srcImage1, g_grayImage, COLOR_BGR2GRAY);

    //��3���������ں͹�����
    namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
    createTrackbar("��ֵ: ", WINDOW_NAME1, &thresh, max_thresh, on_CornerHarris);

    //��4������һ�λص����������г�ʼ��
    on_CornerHarris(0, 0);

    waitKey(0);
    return(0);
}

//-----------------------------------��on_HoughLines( )������--------------------------------
//		�������ص�����
//----------------------------------------------------------------------------------------------

void on_CornerHarris(int, void*)
{
    //---------------------------��1������һЩ�ֲ�����-----------------------------
    Mat dstImage;//Ŀ��ͼ
    Mat normImage;//��һ�����ͼ
    Mat scaledImage;//���Ա任��İ�λ�޷������͵�ͼ

    //---------------------------��2����ʼ��---------------------------------------
    //���㵱ǰ��Ҫ��ʾ������ͼ���������һ�ε��ô˺���ʱ���ǵ�ֵ
    dstImage = Mat::zeros(g_srcImage.size(), CV_32FC1);
    g_srcImage1 = g_srcImage.clone();

    //---------------------------��3����ʽ���-------------------------------------
    //���нǵ���
    cornerHarris(g_grayImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);

    // ��һ����ת��
    normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(normImage, scaledImage);//����һ�����ͼ���Ա任��8λ�޷������� 

    //---------------------------��4�����л���-------------------------------------
    // ����⵽�ģ��ҷ�����ֵ�����Ľǵ���Ƴ���
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
    //---------------------------��4����ʾ����Ч��---------------------------------
    imshow(WINDOW_NAME1, g_srcImage1);
    imshow(WINDOW_NAME2, scaledImage);

}

//-----------------------------------��ShowHelpText( )������----------------------------------
//		���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
    //�����ӭ��Ϣ��OpenCV�汾
    printf("\n\n\t\t\t�ǳ���л����OpenCV3������š�һ�飡\n");
    printf("\n\n\t\t\t��Ϊ����OpenCV3��ĵ�86������ʾ������\n");
    printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);
    printf("\n\n  ----------------------------------------------------------------------------\n");
    //���һЩ������Ϣ
    printf("\n\n\n\t����ӭ����Harris�ǵ���ʾ������~��\n\n");
    printf("\n\t������������۲�ͼ��Ч��~\n\n");
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