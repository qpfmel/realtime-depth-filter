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
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <future>
#include <opencv2/ximgproc.hpp>

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

// guided filter: 안내자 I의 경계를 따라 p를 매끄럽게 한다
// I = 안내자 (흑백, 0~1 float), p = 고칠 깊이, r = 창 반지름, eps = "진짜 경계" 기준
// 이 작업은 흑백 영상이 단순히 늘어지고 옮겨진것 뿐이라서 경계위치가 뚜렷한 흑백 영상에서 가져오기때문에 결과영상의 깊이맵이 흑백영상의 뚜렷한 경계를 가질수있어
cv::Mat guidedFilter (const cv::Mat& I, const cv::Mat& p, int r, double eps) {
    cv::Size win(2 * r + 1, 2 * r + 1);
    cv::Mat meanI, meanP, corrIP, corrII;
    cv::boxFilter(I, meanI, CV_32F, win);           // 창 안의 흑백 평균
    cv::boxFilter(p, meanP, CV_32F, win);           // 창 안의 깊이 평균
    cv::boxFilter(I.mul(p), corrIP, CV_32F, win);   // 창 안의 (흑백X깊이) 평균
    cv::boxFilter(I.mul(I), corrII, CV_32F, win);   // 창 안의 (흑백X흑백) 평균

    cv::Mat varI = corrII - meanI.mul(meanI);   // 흑백 분산 (창 안에서 흑백이 변하는 정도)
    cv::Mat covIP = corrIP - meanI.mul(meanP);  // 공분산: 흑백과 깊이가 얼마나 "같이" 변하나

    cv::Mat a = covIP / (varI + eps);   // 직선의 기울기 (기울기 = 공분산 / 분산) : 결과 깊이맵의 경계면을 뚜렷하게 하기위한 장치 (흑백영상의 경계면을 빌려옴)
    // eps = 경계로 인정하는 합격선. 크면 뿌옇게 뭉개고, 작으면 경계를 또렷하게 살림 (+ 0 나누기 방지)
    cv::Mat b = meanP - a.mul(meanI);   // 직선의 절편 (원래 깊이맵의 값과 유사하게 높이기위한 장치)

    cv::Mat meanA, meanB;                   // 여러 창들의 직선을 평균 내서 가장 유사한 절편과 기울기를 구함
    cv::boxFilter(a, meanA, CV_32F, win);   
    cv::boxFilter(b, meanB, CV_32F, win);
    return meanA.mul(I) + meanB;    // 깊이 = a × 흑백 + b
}

struct Sums {   // 1초동안 모으는 합계들
    int frames = 0;     // 1초 동안 처리한 프레임 수
    double wait = 0, infer = 0, ema = 0, flow = 0, flowWait = 0, gf = 0;   // 시간 (ms)
    double j = 0, jc = 0, jg = 0, jf = 0, js = 0, l = 0;  // 흔들림, 잔상
    double mot = 0, gfDiff = 0;     // 기타
};

struct Result {
    cv::Mat frame;
    cv::Mat depth;      // 원본 깊이맵
    cv::Mat depthStab;  // 안정화 깊이맵
    double inferMs = 0, J = 0, Jc = 0, Jg = 0, L = 0;
    double Js = 0;  //안정화결과의 J
    float a = 1.0f;

    double motionPct = 0;   // 움직였다고 판단한 픽셀 비율
    int thr = 0;            // 이 결과에 쓴 임계값

    cv::Mat lag;    // 픽셀별 잔상 |출력 - 원본|
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

        std::atomic<int> mode{1};   // 1=s1, 2=s2, 3=s3
        std::atomic<int> motionThr{17};     // 움직임 임계값

        std::atomic<int> s4Mode{0}; // s4 모드 전환 3가지 상태라 bool이 아닌 int
        std::atomic<int> gfR{8};    // 창 반지름 R (8이면 창크기는 17X17)
        const double kEps[] = {1e-4, 1e-3, 1e-2, 1e-1}; // eps 후보 4개
        std::atomic<int> epsIdx{2}; // 몇번째 eps후보를 사용할지 (3번째 부터 시작)

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
                int lastId = 0; // 마지막으로 받은 프레임 번호
                cv::Mat prevDepth;  // 이전 깊이맵
                auto lastTime = std::chrono::steady_clock::now();   // 1초 통계내기 위한 변수
                double inferMs = 0, J = 0, Jc =0, Jg = 0;   // sum이 안붙은 변수는 1초 평균 (다음 1초가 끝날때까지 값 유지)
                cv::Mat ema; //EMA의 이전 출력
                double Js = 0;
                double emaT = 0;
                double L = 0;
                cv::Mat prevGray;   //직전 카메라 영상
                double motionPct = 0;
                double flowT = 0, flowWaitT = 0;
                Sums sum;

                cv::Mat prevDepthF; // 이전 프레임의 s4결과
                double gfT = 0, Jf = 0, gfDiff = 0;


                cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)); //동그라미 모양 도장을 만드는 함수
                
                cv::Mat grid(kInputH, kInputW, CV_32FC2);   // 칸(x, y)에 자기 좌표(x, y)를 적어서 그리기
                for (int y = 0; y < grid.rows; ++y) {
                    for (int x=0; x < grid.cols; ++x) {
                        grid.at<cv::Vec2f>(y, x) = cv::Vec2f((float)x, (float)y);   //코드 기준인 x,y를 y,x(행,열)로 바꾸기
                    }
                }

                while(running) {
                    auto waitStart = std::chrono::steady_clock::now();
                    cv::Mat frame; //캡쳐한 최신 프레임을 추론스레드가 가져오는 역할 (복사만 빠르게)
                    {   // 중괄호가 잠금을 잠깐동안만 유지하는 역할
                        std::unique_lock<std::mutex> lock(frameMutex); //스레드 잠금
                        frameCv.wait(lock, [&] { return frameId != lastId || !running; }); //조건이 참이면 바로 복사, 거짓이면 잠금을 풀고 잠듬, 캡처가 신호를 보내면 다시 깨서 조건확인
                        if (!running) break; //종료중이면 루프 탈출
                        frame = sharedFrame.clone(); //캡쳐스레드에서 공유한 새프레임을 내가 쓸 프레임저장 공간에 복사
                        lastId = frameId; //받은 번호 기억
                    }
                    auto waitEnd = std::chrono::steady_clock::now();
                    sum.wait += std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();    // 새 프레임을 가져오는 시간 구하기 (wait)

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
                    
                    
                    cv::Mat gray;
                    cv::cvtColor(resized, gray, cv::COLOR_RGB2GRAY);
                    // 전처리때 resized한 카메라 영상을 흑백으로
                    // 깊이맵과의 크기가 같음
                    int m = mode.load();  // S? (?=1, 2, 3 (모드변경)

                    bool useFlow = (m == 3 && !prevGray.empty());   // s3모드임과 동시에 전흑백프레임이 존재할경우 광학흐름(useFlow) 계산
                    double flowMs = 0;  // 임시 스레드가 사용하는 변수 (추론 스레드는 .get()뒤에서만 읽음)
                    std::future<cv::Mat> flowJob; // ? 영수증이 뭐지?

                    if (useFlow) {
                        flowJob = std::async(std::launch::async, [&](){ // 임시 스레드 생성
                            auto t0 = std::chrono::steady_clock::now();
                            cv::Mat flow;
                            cv::calcOpticalFlowFarneback(gray, prevGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0); // 광학 흐름 계산 (지금과 이전 결과를 비교해서 flow에 채워줌)
                            cv::Mat map = grid + flow; // 주소 = 내 좌표 + 이동량
                            flowMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); // 광학 흐름 계산 시간
                            return map;
                        });
                    }

                    //-----------추론--------------
                    Ort::Value input = Ort::Value::CreateTensor<float>(
                        mem, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size()); //텐서 생성(얕은 복사)
                    auto inferStart = std::chrono::steady_clock::now(); // 추론 시작 전 시각
                    std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);   // 실제 모델을 실행하는 코드
                    auto inferEnd = std::chrono::steady_clock::now(); //추론 직후 시각
                    sum.infer += std::chrono::duration<double, std::milli>(inferEnd - inferStart).count(); // 이번 프레임의 추론시간을 합계에 더한다 (밀리초 변환)

                    std::vector<int64_t> out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape(); //출력 모양 읽기
                    float* depth_data = outputs[0].GetTensorMutableData<float>(); //깊이 숫자들의 시작주소 받기

                    cv::Mat depth(static_cast<int>(out_shape[1]), static_cast<int>(out_shape[2]), CV_32F, depth_data);
                    //위에서 받은 숫자들을 Mat로 감싸고 static_cast로 타입변환(Out_shape는 int64_t(8바이트)이고 Mat는 int(4바이트)로 받음)
                    

                    // =======================
                    //      흔들림 측정
                    // =======================
                    if (!prevDepth.empty()) {   // 첫 프레임은 이전 깊이가 없어서 건너뛰기
                        double meanNow = cv::mean(depth)[0];        //이번 프레임 평균
                        double meanPrev = cv::mean(prevDepth)[0];   //직전 프레임 평균

                        // Jg - 통째로 밀린양. 두 평균의 차이를 직접 잰다
                        sum.jg += std::abs(meanNow - meanPrev); 

                        // J - 전체 흔들림
                        cv::Mat diff;
                        cv::absdiff(depth, prevDepth, diff);
                        sum.j += cv::mean(diff)[0];

                        // Jc - 형태변형. 각자 자기 평균을 뺀 뒤 비교 (J를 밝기 보정한 값)
                        cv::Mat a = depth - cv::Scalar(meanNow);
                        cv::Mat b = prevDepth - cv::Scalar(meanPrev);
                        cv::absdiff(a, b, diff);
                        sum.jc += cv::mean(diff)[0];
                    }
                    prevDepth = depth.clone(); //depth는 ONNX Runtime의 출력 메모리를 가르켜줄뿐이라 clone()을 해서 완전히 값을 복사해야한다 (깊은복사)

                    //-------------s4: 경계보존--------------------------
                    int s4 = s4Mode.load(); // 프레임을 처음 한번만 꺼냄
                    int r = gfR.load(); // 반지름 길이 꺼냄
                    double eps = kEps[epsIdx.load()];   // 실제 eps 값을 꺼냄

                    cv::Mat depthF = depth;
                    if (s4 != 0) {  // s4가 켜져있을때만 실행
                        cv::Mat guide;
                        gray.convertTo(guide, CV_32F, 1.0 / 255.0); // eps 전처리 (eps후보는 0~1기준)
                        auto g0 = std::chrono::steady_clock::now(); // s4 비용 측정을 위한 코드
                        if (s4 == 1) depthF = guidedFilter(guide, depth, r, eps);   // 직접 구현한 가이드 필터를 사용함
                        else cv::ximgproc::guidedFilter(guide, depth, depthF, r, eps);  //contrib의 ximgproc 모듈 안에 있다는 뜻 직접짠 가이드필터와 구분해줌
                        sum.gf += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g0).count(); // 걸린 시간을 ms로 변환해 합계를 구함

                        if (s4 == 2) {  // s4가 contrib모드일때 직접구현한 함수도 한번더 돌림
                            cv::Mat mine = guidedFilter(guide, depth, r, eps), d2;
                            cv::absdiff(mine, depthF, d2); // 픽셀마다 |mine - contrib|를 구해 평균을 내서 비교함 (0에 가까우면 mine이 옳음)
                            cv::Rect inner(r, r, d2.cols - 2 * r, d2.rows - 2 * r);   // 가장자리 r픽셀을 뺀 안쪽 사각형
                            sum.gfDiff += cv::mean(d2(inner))[0];                      // 안쪽만 평균
                        }
                    }

                    if(!prevDepthF.empty()) {   // s4의 j(깊이맵 흔들림)계산
                        cv::Mat d3;
                        cv::absdiff(depthF, prevDepthF, d3);
                        sum.jf += cv::mean(d3)[0];
                    }
                    prevDepthF = depthF.clone();

                    


                    auto emaStart = std::chrono::steady_clock::now();
                    //----------안정화 (EMA)----------
                    float a = alpha.load();     //이번 프레임에 쓸 alpa
                    int thr = motionThr.load(); //임계값


                    cv::Mat motion = cv::Mat::zeros(gray.size(), CV_8U); //첫프레임은 움직임맵 계산을 안하지만 첫 motion을 세기때문에 일단 전부 0으로 채움

                    cv::Mat lagMap; // 첫프레임은 비어있음

                    if(ema.empty()) {           
                        ema = depthF.clone();    //첫프레임은 그대로, 출력 = 원본
                    } 
                    else {

                        cv::Mat ref = prevGray; // 전 흑백 영상
                        cv::Mat base = ema; // 전 출력 영상

                        if(m == 3) {
                            if (useFlow){
                                auto w0 = std::chrono::steady_clock::now();
                                cv::Mat map = flowJob.get();    // 광학 흐름 결과를 받음 (덜 끝났으면 여기서 기다림, 두 스레드의 동기화 지점)
                                sum.flowWait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count(); // 기다린 시간 (wait2)
                                sum.flow += flowMs; // .get 뒤라서 flowMs에 안전하게 접근 가능

                                cv::Mat warpedGray, warpedEma;
                                cv::remap(prevGray, warpedGray, map, cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REPLICATE); // 주소표대로 이전 흑백영상을 옮김
                                cv::remap(ema, warpedEma, map, cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REPLICATE); // 주소표대로 이전 출력영상을 옮김
                                ref = warpedGray; // s3일때 ref에 옮긴 값을 넣음
                                base = warpedEma; // s3일때 base에 옮긴 값을 넣음
                            }
                        }

                        cv::Mat diffImg;
                        cv::absdiff(gray, ref, diffImg); // 픽셀마다 |지금밝기 - 직전밝기|, 정지상태는 거의 0 / 움직임은 숫자가 커짐
                        motion = diffImg > thr; // 픽셀마다의 차이가 임계값보다 크면 255 아니면 0
                        
                        cv::Mat motionWide;
                        cv::dilate(motion, motionWide, dilateKernel);   
                        // dilate = 팽창 (움직임 맵을 도장 크기만큼 넓힘) 
                        //깊이 모델을 경계를 흐릿하게 출력하기때문에 실제 움직임 맵에서는 안움직였다고 판단할 수 있기때문에 움직임 맵을 넓혀서 즉시 따라가게 함
                        motion = motionWide;

                        cv::Mat alphaMap(depth.size(), CV_32F, cv::Scalar(a)); // 깊이맵과 같은 크기의 그림을 그리고, 모든칸을 알파로 채움
                        if (m >= 2) alphaMap.setTo(1.0f, motion); // s2가 참일때만 motion이 255인 칸만 알파를 1.0으로 바꿈
                        cv::blur(alphaMap, alphaMap, cv::Size(15, 15)); // 알파맵을 15x15 평균으로 흐리게 (도장 테두리 방지)

                        cv::Mat next = base + alphaMap.mul(depthF - base); // ema공식

                        cv::Mat d;
                        cv::absdiff(next, ema, d); // J 안정화 구하기 ()
                        sum.js += cv::mean(d)[0];
                        ema = next; // 다음번에 쓸 이번출력 저장

                        cv::absdiff(ema, depthF, d);
                        lagMap = d; // L의 평균을 내기전, 픽셀별 값 그대로
                        sum.l += cv::mean(d)[0];
                    }
                    prevGray = gray.clone(); // .get 뒤에서만 사용해야 한다 (임시스레드가 읽는 도중 바뀌면 데이터 레이스 발생 가능)

                    sum.mot += cv::countNonZero(motion) * 100.0 / motion.total(); // 0이 아닌 칸수 * 100 / 전체 칸 수 = 움직인 비율% (움직였다고 판단한 픽셀비율을 더함)
                    
                    auto emaEnd = std::chrono::steady_clock::now();
                    sum.ema += std::chrono::duration<double, std::milli>(emaEnd - emaStart).count(); // 안정화 시간 계산
                    

                    sum.frames++;   // 프레임 세기
                    auto now = std::chrono::steady_clock::now();    // 현재 시간 기록 (프레임을 센 시간)

                    //------------1초 통계-----------
                    double elapsed = std::chrono::duration<double>(now - lastTime).count();
                    // 시간 간격을 초단위 소수로 바꿔 단위를 빼고 숫자만 꺼내옴
                    if(elapsed >= 1.0){  //세는건 매바퀴고 경과 시간이 1초를 넘었을때 계산하고 초기화

                        double capFps = captureCount.exchange(0) / elapsed; // 값을 읽음과 동시에 0으로 만듬
                        double dispFps = displayCount.exchange(0) / elapsed;
                        double inferFps = sum.frames / elapsed;
                        double waitMs = sum.wait / sum.frames;

                        //frameCount가 0이 되기전에 계산 (1초동안 모은 합계 / 프레임 수)
                        inferMs = sum.infer / sum.frames;  
                        J = sum.j / sum.frames;
                        Jc = sum.jc / sum.frames;
                        Jg = sum.jg / sum.frames;
                        Js = sum.js / sum.frames;
                        emaT = sum.ema / sum.frames;
                        L = sum.l / sum.frames;
                        motionPct = sum.mot / sum.frames;
                        flowT = sum.flow / sum.frames;
                        flowWaitT = sum.flowWait / sum.frames;
                        gfT = sum.gf / sum.frames;  
                        Jf = sum.jf / sum.frames;  
                        gfDiff = sum.gfDiff / sum.frames;

                        std::cout << cv::format("[speed] fps=%.1f infer=%.1f ema=%.2f flow=%.1f wait2=%.1f gf=%.2f | "
                        "[jitter] J=%.4f Jf=%.4f Js=%.4f L=%.3f | "
                        "[set] S%d a=%.1f thr=%d mot=%.1f%% S4=%d r=%d eps=%.4f diff=%.5f",
                        inferFps, inferMs, emaT, flowT, flowWaitT, gfT,
                        J, Jf, Js, L,
                        m, a, thr, motionPct, s4, r, eps, gfDiff) << std::endl;

                        // 다음 1초 계산을 위해 값을 0으로 초기화
                        sum = {};
                        lastTime = now;  
                    }
                    
                    // Result 구조체에 넣기
                    {   // 추론스레드가 넣고 main스레드가 꺼냄
                        std::lock_guard<std::mutex> lock(resultMutex);
                        sharedResult.frame = frame.clone();
                        sharedResult.depth = depthF.clone();
                        sharedResult.depthStab = ema.clone();
                        sharedResult.inferMs = inferMs;
                        sharedResult.J = J;
                        sharedResult.Jc = Jc;
                        sharedResult.Jg = Jg;
                        sharedResult.Js = Js;
                        sharedResult.a = a;
                        sharedResult.L = L;

                        sharedResult.motionPct = motionPct;
                        sharedResult.thr = thr;

                        sharedResult.lag = lagMap.clone();

                        resultId++; // 새 결과가 들어왔다는 증표
                    }
                }
            }
            catch (const std::exception& e) {   // 추론스레드에서 예외가 나면 running을 끄고 다른 스레드를 깨움 (예외가 스레드 밖으로 나가면 강제종료)
                std::cerr << "Infer thread error: " << e.what() << std::endl;
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    running = false;
                }
                frameCv.notify_all();
            }
        });
        ThreadGuard inferGuard{inferThread, running, frameMutex, frameCv};

        bool showStab = false; // main만 읽고 쓰는 변수
        bool showLag = false;

        // ============================================================
        // 표시만
        // ============================================================
            int lastResultId = 0;
            Result shown; // main스레드 전용

            while(running) {
                bool fresh = false;
                {
                    std::lock_guard<std::mutex> lock(resultMutex);
                    if (resultId != lastResultId) { // 새 결과가 들어왔는지 확인
                        shown = sharedResult;
                        shown.lag = sharedResult.lag.clone();
                        shown.frame = sharedResult.frame.clone();   // 깊은 복사로 putTxet에서 마음대로 그리기 위해 .clone()사용
                        shown.depth = sharedResult.depth.clone();
                        shown.depthStab = sharedResult.depthStab.clone();
                        lastResultId = resultId; // 새결과 판별 번호와 이전결과 판별번호를 같게 함 (다음 결과 판별 위함)
                        fresh = true; // 새 결과를 받았다는 표시 (루프 안에서만)
                    }
                }

                if (fresh) {
                    //시각화
                    double mn = 0, mx = 0;
                    cv::minMaxLoc(shown.depth, &mn, &mx);

                    cv::Mat depth_vis;

                    if (showLag && !shown.lag.empty()) {    
                        shown.lag.convertTo(depth_vis, CV_8U, 255.0 / (0.2 * (mx - mn))); 
                        // 고정 배율: 깊이 범위의 20% 이상 어긋나면 최대 밝기 → S2와 S3의 밝기를 직접 비교할 수 있음
                    }
                    else {
                        const cv::Mat& toShow = showStab ? shown.depthStab : shown.depth;
                        cv::normalize(toShow, depth_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
                    }

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
                if (key == 'm') mode = mode % 3 + 1; // 1 -> 2 -> 3
                if (key == '[') motionThr = std::min(motionThr + 2, 255); // 임계값 올리기
                if (key == ']') motionThr = std::max(motionThr - 2, 1);     // 내리기
                if (key == 'v') showLag = !showLag;
                if (key == 'g') s4Mode = (s4Mode + 1) % 3;  // 0, 1, 2 전환
                if (key == '-') gfR = std::max(gfR - 2, 1);     // 창 줄이기
                if (key == '=') gfR = std::min(gfR + 2, 32);    // 창 키우기
                if (key == 'e') epsIdx = (epsIdx + 1) % 4;      // 0.01, 0.1, 0.0001, 0.01 전환 
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