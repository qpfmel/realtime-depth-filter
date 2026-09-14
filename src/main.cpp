#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main(){
    std::cout << "OpenCV version: " << CV_VERSION << std::endl; // OpenCV 버전 출력

    std::vector<int> params = {
        cv::CAP_PROP_FOURCC,    cv::VideoWriter::fourcc('Y', 'U', 'Y', '2'),
        cv::CAP_PROP_FRAME_WIDTH, 640,
        cv::CAP_PROP_FRAME_HEIGHT, 480,
        cv::CAP_PROP_FPS, 30
    };

    cv::VideoCapture cap(0, cv::CAP_DSHOW, params); // 카메라 열기
    if(!cap.isOpened()){    //카메라가 안열릴경우 오류 메세지 출력
        std::cerr << "Failed to open camera" << std::endl;
        return 1;
    }

    std::cout << "Reported: " 
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
              << cap.get(cv::CAP_PROP_FPS) << " fps" << std::endl;

    cv::Mat frame; //이미지를 채워넣을 공간 생성
    cap.read(frame); //frame에 이미지 채워넣기
    if(frame.empty()){ //frame이 비어져있을 경우 메세지
        std::cerr << "First frame is empty" << std::endl;
        return 1;
    }
    std::cout << "Actual frame: " << frame.cols << "x" << frame.rows << std::endl;

    while(true){
        cap.read(frame);
        if(frame.empty()){
            std::cerr << "Empty frame" << std::endl;
            break;
        }
        cv::imshow("Webcam", frame);
        if(cv::waitKey(1) == 27){ // esc 누를 경우 반복탈출(카메라 꺼짐)
            break;
        }
    }
    return 0;

}