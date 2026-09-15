#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(){
    std::cout << "OpenCV version: " << CV_VERSION << std::endl; // OpenCV 버전 출력

    std::vector<int> params = {
        cv::CAP_PROP_FOURCC,    cv::VideoWriter::fourcc('Y', 'U', 'Y', '2'),
        cv::CAP_PROP_FRAME_WIDTH, 640,
        cv::CAP_PROP_FRAME_HEIGHT, 480,
        cv::CAP_PROP_FPS, 30
    };

    cv::VideoCapture cap(0, cv::CAP_DSHOW, params); // 카메라 열기
    if(!cap.isOpened()){    //카메라가 안열릴경우 카메라 종류
        std::cerr << "Failed to open camera" << std::endl;
        return 1;
    }

    std::cout << "Reported: " 
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
              << cap.get(cv::CAP_PROP_FPS) << " fps" << std::endl;

    cv::Mat frame; //이미지를 채워넣을 공간 생성
    cap.read(frame); //frame에 이미지 채워넣기
    if(frame.empty()){ //첫 frame이 비어져있을 경우 카메라 종료
        std::cerr << "First frame is empty" << std::endl;
        return 1;
    }
    std::cout << "Actual frame: " << frame.cols << "x" << frame.rows << std::endl;

    int frameCount = 0; //fps 측정용 변수
    auto lastTime = std::chrono::steady_clock::now();   //현재 시각 가져오기
    double fps = 0.0;

    while(true){
        cap.read(frame);
        if(frame.empty()){
            std::cerr << "Empty frame" << std::endl;
            break;
        }
        
        frameCount++; // empty()다음에 있어 제대로 읽힌 프레임만 셈
        auto now = std::chrono::steady_clock::now(); 
        double elapsed = std::chrono::duration<double>(now - lastTime).count();
        // 시간 간격을 초단위 소수로 바꿔 단위를 빼고 숫자만 꺼내옴
        if(elapsed >= 1.0){ //세는건 매바퀴고 경과 시간이 1초를 넘었을때 계산하고 초기화
            fps = frameCount / elapsed; // 정수/소수 = 소수
            frameCount = 0;
            lastTime = now;
        }

        cv::putText(frame, cv::format("FPS: %.1F", fps), cv::Point(10, 30), //fps 표시
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, cv::format("%dx%d", frame.cols, frame.rows), cv::Point(10, 470), //해상도 표시
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Webcam", frame);
        if(cv::waitKey(1) == 27){ // esc 누를 경우 반복탈출
            break;
        }
    }
    return 0;

}