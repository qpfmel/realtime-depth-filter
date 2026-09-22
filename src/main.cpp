#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <array>

void setAndCheck(cv::VideoCapture& cap, const char* label, int prop, double value) {
    bool ok = cap.set(prop, value);
    double back = cap.get(prop);
    std::cout << label << ": set(" << value << ") -> ok=" << (ok ? "true" : "false") << ", get=" << back << std::endl;
}

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

        setAndCheck(cap, "AUTO_EXPOSURE", cv::CAP_PROP_AUTO_EXPOSURE, 0);
        setAndCheck(cap, "EXPOSURE", cv::CAP_PROP_EXPOSURE, -6);
        setAndCheck(cap, "AUTO_WB", cv::CAP_PROP_AUTO_WB, 0);
        setAndCheck(cap, "AUTOFOCUS", cv::CAP_PROP_AUTOFOCUS, 0);

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




        //=================반복문 밖의 변할필요 없는 고정된 값들==========================

        const int kInputW = 518; //14의 배수중 웹캠 비율에 가장 유사한 숫자
        const int kInputH = 392;

        std::array<int64_t, 4> input_shape{1, 3, kInputH, kInputW};// 입력형식 고정(1, 3, 392, 518)

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //데이터가 cpu에 있다는것을 알려줌

        const char* input_names[] = {"pixel_values"};
        const char* output_names[] = {"predicted_depth"};
        
        int frameCount = 0;                                 //fps 측정용 변수
        auto lastTime = std::chrono::steady_clock::now();   //현재 시각 가져오기
        double fps = 0.0;                                   //fps
        double inferSumMs = 0.0;    //1초동안 추론에 쓴 시간의 합
        double inferMs = 0.0;       //화면에 보여줄 평균 추론 시간

        cv::Mat prevDepth;       //직전 프레임의 깊이맵
        double jitter = 0.0;     //화면에 보여줄 흔들림 값
        double jitterSum = 0.0;  //1초 동안의 합   
        double jitterCentered = 0.0;
        double jitterCenteredSum = 0.0;

        double capSumMs = 0.0;
        double capMs = 0.0;
        double otherMs = 0.0;

        
        while(true){
            auto capStart = std::chrono::steady_clock::now();
            cap.read(frame); //frame에 이미지 채워넣기
            auto capEnd = std::chrono::steady_clock::now();
            capSumMs += std::chrono::duration<double, std::milli>(capEnd - capStart).count();

            if(frame.empty()){ //첫 frame이 비어져있을 경우 카메라 종료
                std::cerr << "First frame is empty" << std::endl;
                break;
            }

            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);    // 색 순서를 bgr에서 rgb로(OpenCV는 bgr이지만 해당 모델은 rgb여서)
            cv::Mat resized;
            cv::resize(rgb, resized, cv::Size(kInputW, kInputH));   // 모델이 14x14 조각으로 보기때문에 14배수로 진행
            cv::Mat f32;
            resized.convertTo(f32, CV_32F, 1.0 / 255.0); // 0~1(소수)로 숫자형식 변경
            cv::Mat norm = (f32 - cv::Scalar(0.485, 0.456, 0.406)) / cv::Scalar(0.229, 0.224, 0.225);
            //학습할때 쓴 평균/표준편차로 맞춘다 (ImageNet 기준)
            cv::Mat blob = cv::dnn::blobFromImage(norm); // 배치순서 바꾸기(h,w,c -> p(장수), c, h, w)

            
            Ort::Value input = Ort::Value::CreateTensor<float>(
                mem, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size()); //텐서 생성(얕은 복사)
            
            auto inferStart = std::chrono::steady_clock::now(); // 추론 시작 전 시각
            std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);   // 실제 모델을 실행하는 코드
            auto inferEnd = std::chrono::steady_clock::now(); //추론 직후 시각
            
            inferSumMs += std::chrono::duration<double, std::milli>(inferEnd - inferStart).count(); // 이번 프레임의 추론시간을 합계에 더한다 (밀리초 변환)

            std::vector<int64_t> out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape(); //출력 모양 읽기
            float* depth_data = outputs[0].GetTensorMutableData<float>(); //깊이 숫자들의 시작주소 받기

            cv::Mat depth(static_cast<int>(out_shape[1]), static_cast<int>(out_shape[2]), CV_32F, depth_data);
            //위에서 받은 숫자들을 Mat로 감싸고 static_cast로 타입변환(Out_shape는 int64_t(8바이트)이고 Mat는 int(4바이트)로 받음)

            // =====흔들림 측정=========
            if (!prevDepth.empty()) {
                cv::Mat diff;
                cv::absdiff(depth, prevDepth, diff); //픽셀마다 |(이번) - (직전)|(절댓값)을 구해 diff에 넣음
                jitterSum += cv::mean(diff)[0];      //전체 픽셀의 평균


                cv::Mat a = depth - cv::Scalar(cv::mean(depth)[0]);    //프레임마다 자기 평균을 빼기
                cv::Mat b = prevDepth - cv::Scalar(cv::mean(prevDepth)[0]);

                cv::absdiff(a, b, diff);                     //픽셀마다 |(이번-이번평균) - (직전 - 직전평균)|(절댓값)을 구해 diff에 넣음
                jitterCenteredSum += cv::mean(diff)[0];      //전체 픽셀의 평균
            }
            prevDepth = depth.clone(); //depth는 ONNX Runtime의 출력 메모리를 가르켜줄뿐이라 clone()을 해서 완전히 값을 복사해야한다
            

            //시각화
            cv::Mat depth_vis;
            cv::normalize(depth, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U); //최대, 최솟값을 0~255로 펼쳐버린다
            cv::Mat depth_color;
            cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_INFERNO);

            double mn = 0, mx = 0;
            cv::minMaxLoc(depth, &mn, &mx); // 변수의 주소를 넘겨서 주소에 직접적으로 값을 새겨넣게함

            std::cout << "depth shape=[" << out_shape[0] << " " << out_shape[1] << " " << out_shape[2] //min, max 값 표시
                    << " ] min=" << mn << " max=" << mx <<std::endl;


            frameCount++; // empty()다음에 있어 제대로 읽힌 프레임만 셈
            auto now = std::chrono::steady_clock::now(); 
            double elapsed = std::chrono::duration<double>(now - lastTime).count();
            // 시간 간격을 초단위 소수로 바꿔 단위를 빼고 숫자만 꺼내옴
            if(elapsed >= 1.0){  //세는건 매바퀴고 경과 시간이 1초를 넘었을때 계산하고 초기화
                fps = frameCount / elapsed; // 정수/소수 = 소수
                
                inferMs = inferSumMs / frameCount;  //frameCount가 0이 되기전에 계산
                jitter = jitterSum / frameCount;    
                jitterCentered = jitterCenteredSum / frameCount;
                capMs = capSumMs / frameCount;

                otherMs = (1000.0 / fps) - capMs - inferMs;

                std::cout << cv::format("fps=%.1f cap=%.1fms  infer=%.1fms  J=%.4f  Jc=%.4f  d=%.2f~%.2f", //fps, infer, j, jc, d 값 표시
                                    fps, capMs, inferMs, jitter, jitterCentered, mn, mx) << std::endl;
                
                jitterCenteredSum = 0.0;
                jitterSum = 0.0;
                frameCount = 0;
                inferSumMs = 0.0;
                capSumMs = 0.0;

                lastTime = now;  
            }

            cv::putText(frame, cv::format("FPS: %.1F", fps), cv::Point(10, 30), //fps 표시
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("Infer: %.1f ms", inferMs), cv::Point(10, 65), //infer 표시
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("%dx%d", frame.cols, frame.rows), cv::Point(10, 470), //해상도 표시
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("J: %.4f  Jc: %.3f", jitter, jitterCentered), cv::Point(10, 100), //jitter 표시
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("d: %.2f ~ %.2f", mn, mx), cv::Point(10, 135),    //깊이맵 최소~최대 표시
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("cap: %.1fms", capMs), cv::Point(10, 170),  // cap표시
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, cv::format("otherMs: %.2fms", otherMs), cv::Point(10, 205),  // 전체 소요시간 표시
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Webcam", frame);
            cv::imshow("Depth", depth_color);
            
            if(cv::waitKey(1) == 27){ // esc 누를 경우 반복탈출
                break;
            }
        }
    }
    catch (const Ort::Exception& e) { //try catch로 실패시 원인을 파악하기 위해 사용
        std::cerr << "ORT error: " << e.what() << std::endl;
        return 1;
    }
    return 0;

}