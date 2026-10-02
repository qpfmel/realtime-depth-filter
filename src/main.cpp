#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <condition_variable>

void setAndCheck(cv::VideoCapture& cap, const char* label, int prop, double value) {
    bool ok = cap.set(prop, value);
    double back = cap.get(prop);
    std::cout << label << ": set(" << value << ") -> ok=" << (ok ? "true" : "false") << ", get=" << back << std::endl;
}

struct ThreadGuard {
    std::thread& t;
    std::atomic<bool>& running;
    std::mutex& m;
    std::condition_variable& cv;
    ~ThreadGuard() {
        {
            std::lock_guard<std::mutex> lock(m);
            running = false;
        }
        cv.notify_all();
        if (t.joinable()) t.join();
    }
    
};

struct Result {
    cv::Mat frame;
    cv::Mat depth;      // 원본 깊이맵
    cv::Mat depthStab;  // 안정화 깊이맵
    double inferMs = 0, J = 0, Jc = 0, Jg = 0;
    double Js = 0;  //안정화결과의 J
    float a = 1.0f;
};

int main(){
    std::cout << "OpenCV version: " << CV_VERSION << std::endl; // OpenCV 버전 출력

    std::cout << "ORT providers:";
    for (const std::string& p : Ort::GetAvailableProviders()) {    //문자열 여러개가 담긴 vector
        std::cout << " " << p;
    }
    std::cout << std::endl;

    try {
        // ============================================================
        // 1. 모델 준비
        // ============================================================
        
        //------모델 입출력 정보 확인----------
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "depth_filter");    // 전체 초기화
        Ort::SessionOptions options;    // 설정상자
        OrtCUDAProviderOptions cuda_options; // CUDA설정, 기본생성자가 기본값을 채워줌
        options.AppendExecutionProvider_CUDA(cuda_options); // CUDA 우선순위 등록

        Ort::Session session(env, L"models/depth_anything_v2_small.onnx", options); //모델 읽기&실행준비
        std::cout << "CUDA session created" << std::endl;

        Ort::AllocatorWithDefaultOptions allocator; //ONNX Runtime의 메모리 대여창구 (이름 문자열을 담는 메모리)

        //------------------------모델 입출력 정보 확인--------------------------------
        // 입력
        for (size_t i = 0; i < session.GetInputCount(); ++i){ //데이터를 넣을 구멍 개수 만큼 반복
            Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator); // i번째 입력 구멍의 이름 확인
            //   =unique_ptr : name이 사라질때 메모리 이름 자동 반납(누수x 현대 c++의 메모리 자동관리)
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i); // i번째 입력에 들어가는 데이터의 종류 정보 묶음을 받음
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo(); // 종류 정보중 텐서 정보만 좁혀서 본다 (type_info보다 오래쓸경우 댕글링 발생)
                                                                      //tensor_info는 가르키기만 하는데 type_info가 먼저 사라지면 반납된 메모리의 번호를 들고있음 그게 댕글링포인터

            std::cout << "Input " << name.get();   // unique_ptr에 든 실제 글자 주소를 잠깐 꺼낸다 (소유는 여전히 name이 한다)
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

        // ============================================================
        // 2. 카메라 준비
        // ============================================================
    
        std::vector<int> params = {
            cv::CAP_PROP_FOURCC,    cv::VideoWriter::fourcc('Y', 'U', 'Y', '2'), // 포맷
            cv::CAP_PROP_FRAME_WIDTH, 640,
            cv::CAP_PROP_FRAME_HEIGHT, 480,
            cv::CAP_PROP_FPS, 30
        };

        cv::VideoCapture cap(0, cv::CAP_DSHOW, params); // 카메라 열기
        if(!cap.isOpened()){    //카메라가 안열릴경우 카메라 종류
            std::cerr << "Failed to open camera" << std::endl;
            return 1;
        }

        //---------자동노출, wb, 초점 잠금--------------
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

        // ============================================================
        // 3. 스레드 준비 — 캡처를 분리한다
        // ============================================================

        cv::Mat sharedFrame;    //최신 프레임 한장
        int frameId = 0;        //방 번호표, 캡처가 넣을때마다 +1
        std::mutex frameMutex;  // 자물쇠
        std::condition_variable frameCv; // 새프레임이 왔다는 신호
        std::atomic<bool> running{true};    // 종료 신호
        std::atomic<int> captureCount{0};

        Result sharedResult;
        int resultId = 0;
        std::mutex resultMutex;
        std::atomic<int> displayCount{0};

        std::atomic<float> alpha{1.0f};

        std::thread captureThread([&]() { // 람다함수, [&] - 바깥의 변수들을 원본 그대로 쓰겠다는 뜻
            cv::Mat local;  //스레드 버퍼
            while (running) {
                if (!cap.read(local) || local.empty()) break; // 첫프레임 방지
                {
                    std::lock_guard<std::mutex> lock(frameMutex);   //RAII - 만들어질때 잠금, 블록을 벗어나면 자동 해제 (데드락 방지)
                    sharedFrame = local.clone(); // 깊은 복사로 칸이 local을 가르키게 되는것을 방지함 (main이 프레임을 읽는도중 내용이 바뀔 수 있음)
                    frameId++;
                }
                frameCv.notify_one(); // 초인종, 기다리는 쪽 하나를 깨움
                captureCount++;
            }

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                running = false;
            }
            frameCv.notify_all();
        });
        ThreadGuard captureGuard{captureThread, running, frameMutex, frameCv};

        // ================================
        // 4. 반복문 밖의 변할필요 없는 고정된 값들
        // ================================

        const int kInputW = 518; //14의 배수중 웹캠 비율에 가장 유사한 숫자
        const int kInputH = 392;

        std::array<int64_t, 4> input_shape{1, 3, kInputH, kInputW};// 입력형식 고정(1, 3, 392, 518)

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //데이터가 cpu에 있다는것을 알려줌

        const char* input_names[] = {"pixel_values"};
        const char* output_names[] = {"predicted_depth"};


        std::thread inferThread([&]() {
            try{
                // ---------추론 스레드에서만 쓰는 변수------
                int lastId = 0;
                cv::Mat prevDepth;
                int frameCount = 0;
                auto lastTime = std::chrono::steady_clock::now();
                double waitSumMs = 0, inferSumMs = 0, jSum = 0, jcSum = 0, jgSum = 0;
                double inferMs = 0, J = 0, Jc =0, Jg = 0;
                cv::Mat ema; //EMA의 이전 출력
                double jsSum = 0, Js = 0;
                
                while(running) {
                    auto waitStart = std::chrono::steady_clock::now();
                    cv::Mat frame; //캡쳐한 최신 프레임을 추론스레드가 가져오는 역할 (복사만 빠르게)
                    {   // 중괄호가 잠금을 잠깐동안만 유지하는 역할
                        std::unique_lock<std::mutex> lock(frameMutex); //스레드 잠금
                        frameCv.wait(lock, [&] { return frameId != lastId || !running; }); //조건이 참이면 바로 복사, 거짓이면 잠금을 풀고 잠듬, 캡처가 신호를 보내면 다시 깨서 조건확인
                        if (!running) break; //종료중이면 루프 탈출
                        frame = sharedFrame.clone();
                        lastId = frameId; //받은 번호 기억
                    }
                    auto waitEnd =  std::chrono::steady_clock::now();
                    waitSumMs += std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();

                    //-----------전처리-----------
                    cv::Mat rgb;
                    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);    // 색 순서를 bgr에서 rgb로(OpenCV는 bgr이지만 해당 모델은 rgb여서)
                    cv::Mat resized;
                    cv::resize(rgb, resized, cv::Size(kInputW, kInputH));   // 모델이 14x14 조각으로 보기때문에 14배수로 진행
                    cv::Mat f32;
                    resized.convertTo(f32, CV_32F, 1.0 / 255.0); // 0~1(소수)로 숫자형식 변경
                    cv::Mat norm = (f32 - cv::Scalar(0.485, 0.456, 0.406)) / cv::Scalar(0.229, 0.224, 0.225);
                    //학습할때 쓴 평균/표준편차로 맞춘다 (ImageNet 기준)
                    cv::Mat blob = cv::dnn::blobFromImage(norm); // 배치순서 바꾸기(h,w,c -> p(장수), c, h, w)

                    //-----------추론--------------
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
                    

                    // =======================
                    //      흔들림 측정
                    // =======================
                    if (!prevDepth.empty()) {
                        double meanNow = cv::mean(depth)[0];        //이번 프레임 평균
                        double meanPrev = cv::mean(prevDepth)[0];   //직전 프레임 평균

                        // Jg - 통째로 밀린양. 두 평균의 차이를 직접 잰다
                        jgSum += std::abs(meanNow - meanPrev); 

                        // J - 전체 흔들림
                        cv::Mat diff;
                        cv::absdiff(depth, prevDepth, diff);
                        jSum += cv::mean(diff)[0];

                        // Jc - 형태변형. 각자 자기 평균을 뺀 뒤 비교 (J를 밝기 보정한 값)
                        cv::Mat a = depth - cv::Scalar(meanNow);
                        cv::Mat b = prevDepth - cv::Scalar(meanPrev);
                        cv::absdiff(a, b, diff);
                        jcSum += cv::mean(diff)[0];
                    }
                    prevDepth = depth.clone(); //depth는 ONNX Runtime의 출력 메모리를 가르켜줄뿐이라 clone()을 해서 완전히 값을 복사해야한다

                    //----------안정화 (EMA)----------
                    float a = alpha.load();     //이번 프레임에 쓸 alpa
                    if(ema.empty()) {           
                        ema = depth.clone();    //첫프레임은 그대로
                    } 
                    else {
                        cv::Mat next;
                        cv::addWeighted(depth, a, ema, 1.0 - a, 0.0, next); //ema 공식
                        cv::Mat d;
                        cv::absdiff(next, ema, d); // J 안정화 구하기 ()
                        jsSum += cv::mean(d)[0];
                        ema = next; // 다음번에 쓸 이번출력 저장
                    }

                    
                    frameCount++;
                    auto now = std::chrono::steady_clock::now();

                    //------------1초 통계-----------
                    double elapsed = std::chrono::duration<double>(now - lastTime).count();
                    // 시간 간격을 초단위 소수로 바꿔 단위를 빼고 숫자만 꺼내옴
                    if(elapsed >= 1.0){  //세는건 매바퀴고 경과 시간이 1초를 넘었을때 계산하고 초기화

                        double capFps = captureCount.exchange(0) / elapsed;
                        double dispFps = displayCount.exchange(0) / elapsed;
                        double inferFps = frameCount / elapsed;
                        double waitMs = waitSumMs / frameCount;

                        inferMs = inferSumMs / frameCount;  //frameCount가 0이 되기전에 계산
                        J = jSum / frameCount;
                        Jc = jcSum / frameCount;
                        Jg = jgSum / frameCount;
                        Js = jsSum / frameCount;

                        std::cout << cv::format("capFps=%.0f inferFps=%.1f dispFps=%.1f wait=%.1fms  infer=%.1fms  J=%.4f  Jc=%.4f  Jg=%.4f  a=%.1f  Js=%.4f",
                                                capFps, inferFps, dispFps, waitMs, inferMs, J, Jc, Jg, a, Js) << std::endl;

                        frameCount = 0;
                        waitSumMs = inferSumMs = jSum = jcSum = jgSum = jsSum = 0;
                        lastTime = now;  
                    }
                    
                    // Result 구조체에 넣기
                    {   // 추론스레드가 넣고 main스레드가 꺼냄
                        std::lock_guard<std::mutex> lock(resultMutex);
                        sharedResult.frame = frame.clone();
                        sharedResult.depth = depth.clone();
                        sharedResult.depthStab = ema.clone();
                        sharedResult.inferMs = inferMs;
                        sharedResult.J = J;
                        sharedResult.Jc = Jc;
                        sharedResult.Jg = Jg;
                        sharedResult.Js = Js;
                        sharedResult.a = a;
                        resultId++;
                    }
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Infer thread error: " << e.what() << std::endl;
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    running = false;
                }
                frameCv.notify_all();
            }
        });
        ThreadGuard inferGuard{inferThread, running, frameMutex, frameCv};

        bool showStab = false;

        // ============================================================
        // 표시만
        // ============================================================
            int lastResultId = 0;
            Result shown; // main스레드 전용

            while(running) {
                bool fresh = false;
                {
                    std::lock_guard<std::mutex> lock(resultMutex);
                    if (resultId != lastResultId) {
                        shown = sharedResult;
                        shown.frame = sharedResult.frame.clone();
                        shown.depth = sharedResult.depth.clone();
                        shown.depthStab = sharedResult.depthStab.clone();
                        lastResultId = resultId;
                        fresh = true;
                    }
                }

                if (fresh) {
                    //시각화
                    double mn = 0, mx = 0;
                    cv::minMaxLoc(shown.depth, &mn, &mx);

                    cv::Mat depth_vis;
                    const cv::Mat& toShow = showStab ? shown.depthStab : shown.depth;
                    cv::normalize(toShow, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
                    cv::Mat depth_color;
                    cv::applyColorMap(depth_vis, depth_color, cv::COLORMAP_INFERNO);

                    //  ==========================================
                    //                  화면 표시
                    //  ==========================================
                    cv::putText(shown.frame, cv::format("Infer: %.1f ms", shown.inferMs), cv::Point(10, 30),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    cv::putText(shown.frame, cv::format("J: %.4f  Jc: %.4f Jg: %.4f", shown.J, shown.Jc, shown.Jg), cv::Point(10, 65),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    cv::putText(shown.frame, cv::format("d: %.2f ~ %.2f", mn, mx), cv::Point(10, 100),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    cv::putText(shown.frame, cv::format("%s  a=%.1f  J=%.4f  Js=%.4f", showStab ? "STAB" : "RAW", shown.a, shown.J, shown.Js),
                                cv::Point(10, 135), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

                    cv::imshow("Webcam", shown.frame);
                    cv::imshow("Depth", depth_color);
                    displayCount++;
                }

                int key = cv::waitKey(1);
                if (key == 27) break;
                if (key >= '1' && key <= '9') alpha = (key - '0') / 10.0f;
                if (key == '0') alpha = 1.0f;
                if (key == 's') showStab = !showStab;
            }
    }
    catch (const Ort::Exception& e) { //try catch로 실패시 원인을 파악하기 위해 사용
        std::cerr << "ORT error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;

}