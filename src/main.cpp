#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <onnxruntime_cxx_api.h>

int main(){
    std::cout << "OpenCV version: " << CV_VERSION << std::endl; // OpenCV 버전 출력

    std::cout << "ORT providers:";
    for (const std::string& p : Ort::GetAvailableProviders()) {    //문자열 여러개가 담긴 vector
        std::cout << " " << p;
    }
    std::cout << std::endl;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "depth_filter");    // 전체 초기화
        Ort::SessionOptions options;    // 설정상자
        OrtCUDAProviderOptions cuda_options; // CUDA설정, 기본생성자가 기본값을 채워줌
        options.AppendExecutionProvider_CUDA(cuda_options); // CUDA 우선순위 등록

        Ort::Session session(env, L"models/depth_anything_v2_small.onnx", options); //모델 읽기&실행준비
        std::cout << "CUDA session created" << std::endl;

        Ort::AllocatorWithDefaultOptions allocator; //ONNX Runtime의 메모리 대여창구 (이름 문자열을 담는 메모리)
        // 입력
        for (size_t i = 0; i < session.GetInputCount(); ++i){ //데이터를 넣을 구멍 개수 만큼 반복
            Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator); // i번째 입력 구멍의 이름 확인
            //   =unique_ptr : name이 사라질때 메모리 이름 자동 반납(누수x 현대 c++의 메모리 자동관리)
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i); // i번째 입력에 들어가는 데이터의 종류 정보 묶음을 받음
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo(); // 종류 정보중 텐서 정보만 좁혀서 본다 (type_info보다 오래쓸경우 댕글링 발생)
                                                                      //tensor_info는 가르키기만 하는데 type_info가 먼저 사라지면 반납된 메모리의 번호를 들고있음 그게 댕글링포인터

            std::cout << "Output " << name.get();   // unique_ptr에 든 실제 글자 주소를 잠깐 꺼낸다 (소유는 여전히 name이 한다)
            if (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { //데이터 타입이 float32일 경우
                std::cout << "  type=" << "float32";
            } 
            else {
               std::cout << "  type=" << tensor_info.GetElementType(); // 각 칸에 든 숫자 형식 (정수?, 소수?)
            }
            std::cout << "  shape=[";

            for (int64_t d : tensor_info.GetShape()) {  // 몇 차원이며, 각 차원의 크기 (vector 복사본으로 반환)
                std::cout << " " << d;
            }
            std::cout << " ]" << std::endl;
        }
        // 출력
        for (size_t i = 0; i < session.GetOutputCount(); ++i){
            Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            std::cout << "Output " << name.get();
            if (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {  //데이터 타입이 float32일 경우
                std::cout << "  type=" << "float32";
            } 
            else {
                std::cout << "  type=" << tensor_info.GetElementType();
            }
            std::cout << "  shape=[";

            for (int64_t d : tensor_info.GetShape()) {
                std::cout << " " << d;
            }
            std::cout << " ]" << std::endl;
        }
    }   
    catch (const Ort::Exception& e) { //try catch로 실패시 원인을 파악하기 위해 사용
        std::cerr << "ORT error: " << e.what() << std::endl;
        return 1;
    }

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