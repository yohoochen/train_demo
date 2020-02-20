//
//
#include <Object_detect.h>
#include <ObjectLayer.h>
#include <assert.h>
#include <fstream>
static Object_Detection::Logger gLogger;

namespace Object_Detection
{

    Object_detect::Object_detect(const std::string &onnxFile, const std::string &calibFile,
            Object_Detection::RUN_MODE mode):forwardFace(false),mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),
                                  runMode(mode),runIters(0)
    {
        gUseDLACore = 1;
        const int maxBatchSize = 1;
        nvinfer1::IHostMemory *modelStream{nullptr};
        int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        nvinfer1::INetworkDefinition* network = builder->createNetwork();


        auto parser = nvonnxparser::createParser(*network, gLogger);
        std::cout << "Begin parsing model..." << std::endl;
        if (!parser->parseFromFile(onnxFile.c_str(), verbosity))
        {
            std::string msg("failed to parse onnx file");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 30);// 1G


        if (runMode== RUN_MODE::INT8)
        {
            //nvinfer1::IInt8Calibrator* calibrator;
            std::cout <<"setInt8Mode"<<std::endl;
            if (!builder->platformHasFastInt8())
                std::cout << "Notice: the platform do not has fast for int8" << std::endl;
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(nullptr);

            enableDLA(builder, gUseDLACore, true);

        }
        else if (runMode == RUN_MODE::FLOAT16)
        {
            std::cout <<"setFp16Mode"<<std::endl;
            if (!builder->platformHasFastFp16())
                std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
            builder->setFp16Mode(true);

            enableDLA(builder, gUseDLACore, true);
        }
        // config input shape

        std::cout << "Begin building engine..." << std::endl;
        nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
        if (!engine){
            std::string error_message ="Unable to create engine";
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, error_message.c_str());
            exit(-1);
        }
        std::cout << "End building engine..." << std::endl;

        // We don't need the network any more, and we can destroy the parser.

        parser->destroy();
        // Serialize the engine, then close everything down.
        modelStream = engine->serialize();
        engine->destroy();
        network->destroy();
        builder->destroy();
        assert(modelStream != nullptr);
        mRunTime = nvinfer1::createInferRuntime(gLogger);
        assert(mRunTime != nullptr);

        if ( (runMode== RUN_MODE::INT8) || (runMode == RUN_MODE::FLOAT16))
        {
            if (gUseDLACore >= 0)
            {
                mRunTime->setDLACore(gUseDLACore);
            }
        }
        
        mEngine= mRunTime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);

        assert(mEngine != nullptr);
        modelStream->destroy();
        InitEngine();

    }

    Object_detect::Object_detect(const std::string &engineFile)
            :forwardFace(false),mContext(nullptr),mEngine(nullptr),mRunTime(nullptr),runMode(RUN_MODE::FLOAT32),runIters(0)
    {
        using namespace std;
        fstream file;

        file.open(engineFile,ios::binary | ios::in);
        if(!file.is_open())
        {
            cout << "read engine file" << engineFile <<" failed" << endl;
            return;
        }
        file.seekg(0, ios::end);
        int length = file.tellg();
        file.seekg(0, ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);

        file.close();

        std::cout << "deserializing" << std::endl;
        mRunTime = nvinfer1::createInferRuntime(gLogger);
        assert(mRunTime != nullptr);
        mEngine= mRunTime->deserializeCudaEngine(data.get(), length, nullptr);
        assert(mEngine != nullptr);
        InitEngine();

    }

    void Object_detect::InitEngine() {
        contours.emplace_back(cv::Point(300, 720));
        contours.emplace_back(cv::Point(550, 80));
        contours.emplace_back(cv::Point(730, 80));
        contours.emplace_back(cv::Point(980, 720));
        contours2.emplace_back(cv::Point(0,80));
        contours2.emplace_back(cv::Point(1280, 80));
        contours2.emplace_back(cv::Point(1280, 720));
        contours2.emplace_back(cv::Point(0, 720));
        const int maxBatchSize = 1;
        mContext = mEngine->createExecutionContext();
        assert(mContext != nullptr);
        mContext->setProfiler(&mProfiler);
        int nbBindings = mEngine->getNbBindings();
        std::cout<<"nbBindings:"<<nbBindings<<std::endl;

        if (nbBindings > 4) forwardFace= true;

        mCudaBuffers.resize(nbBindings);
        mBindBufferSizes.resize(nbBindings);
        int64_t totalSize = 0;
        for (int i = 0; i < nbBindings; ++i)
        {
            nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
            nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
            totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
            mBindBufferSizes[i] = totalSize;
            std::cout << "mBindBufferSizes[0] :" << mBindBufferSizes[i] <<std::endl;
            std::cout << "totalSize :" << totalSize <<std::endl;
            mCudaBuffers[i] = safeCudaMalloc(totalSize);
        }
        outputBufferSize = mBindBufferSizes[1] * 6 ;

        cudaOutputBuffer = safeCudaMalloc(outputBufferSize);
        CUDA_CHECK(cudaStreamCreate(&mCudaStream));


        //inference_outputData = new float[outputBufferSize];

        inference_outputData.reset(new float[outputBufferSize]);
    }

    void Object_detect::doInference(const void *inputData, void *outputData)
    {
        const int batchSize = 1;
        int inputIndex = 0 ;

        //std::cout << "mCudaBuffers[inputIndex]:" <<mCudaBuffers[inputIndex]<< std::endl;
        CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[inputIndex], inputData, mBindBufferSizes[inputIndex], cudaMemcpyHostToDevice, mCudaStream));
                
        mContext->execute(batchSize, &mCudaBuffers[inputIndex]);
        
        CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));

        CTdetforward_gpu(static_cast<const float *>(mCudaBuffers[1]),static_cast<const float *>(mCudaBuffers[2]),
                         static_cast<const float *>(mCudaBuffers[3]),static_cast<float *>(cudaOutputBuffer),
                         ouputSize,ouputSize,classNum,kernelSize,visThresh);
        
        CUDA_CHECK(cudaMemcpyAsync(outputData, cudaOutputBuffer, outputBufferSize, cudaMemcpyDeviceToHost, mCudaStream));

        runIters++ ;
    }
    
    void Object_detect::saveEngine(const std::string &fileName)
    {
        if(mEngine)
        {
            nvinfer1::IHostMemory* data = mEngine->serialize();
            std::ofstream file;
            file.open(fileName,std::ios::binary | std::ios::out);
            if(!file.is_open())
            {
                std::cout << "read create engine file" << fileName <<" failed" << std::endl;
                return;
            }
            file.write((const char*)data->data(), data->size());
            file.close();
        }

    }

    void Object_detect::enableDLA(nvinfer1::IBuilder* b, int useDLACore, bool allowGPUFallback = true)
    {
        if (useDLACore >= 0)
        {
            if (b->getNbDLACores() == 0)
            {
                std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores" << std::endl;
                assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
            }
            b->allowGPUFallback(allowGPUFallback);
            if (!b->getInt8Mode())
            {
                // User has not requested INT8 Mode.
                // By default run in FP16 mode. FP32 mode is not permitted.
                b->setFp16Mode(true);
            }
            b->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            b->setDLACore(useDLACore);
            b->setStrictTypeConstraints(true);
        }
    }



    void Object_detect::detect(cv::Mat& img, std::vector<Object> &Obj_pool)
    {
        int start = cv::getTickCount();


        auto inputData = prepareImage(img);

        int end_tick = cv::getTickCount();
        std::cout << " prepareImage time:" << 1000.0 * (end_tick - start) / cv::getTickFrequency() << "ms" << std::endl << std::endl;


        start = cv::getTickCount();

        doInference(inputData.data(), inference_outputData.get());

        end_tick = cv::getTickCount();
        std::cout << " doInference time:" << 1000.0 * (end_tick - start) / cv::getTickFrequency() << "ms" << std::endl << std::endl;


        start = cv::getTickCount();

        int num_det = static_cast<int>(inference_outputData[0]);
        std::vector<Detection> result;
        result.resize(num_det);
        memcpy(result.data(), &inference_outputData[1], num_det * sizeof(Object_Detection::Detection));
        Object_Detection::postProcess(result,img);
        Object_Detection::Result_convert(result,Obj_pool);

        end_tick = cv::getTickCount();
        std::cout << " postProcess time:" << 1000.0 * (end_tick - start) / cv::getTickFrequency() << "ms" << std::endl << std::endl;


    }


    void draw_bb_top(cv::Mat &img_, std::string &name, cv::Point &pt_lt, cv::Point &pt_br, cv::Scalar &color){
      //cv::rectangle(img_,cv::Point(pt_lt.x,pt_lt.y-15),pt_br,color,2);
        cv::rectangle(img_,cv::Point(pt_lt.x,pt_lt.y),pt_br,color,2);
        cv::rectangle(img_,cv::Point(pt_lt.x,pt_lt.y-15),cv::Point(pt_lt.x + name.length()*15,pt_lt.y),color,-1);
        cv::putText(img_,name,cv::Point(pt_lt.x+7,pt_lt.y-4), cv::FONT_HERSHEY_SIMPLEX, 0.6,cv::Scalar(255,255,255),1,8);
//        cv::Point a = cv::Point(pt_lt.x,pt_br.y);
//        cv::Point b = cv::Point(pt_br.x,pt_lt.y);
//        double result1 = pointPolygonTest(contours, pt_lt, false);
//        double result2 = pointPolygonTest(contours, pt_br, false);
//        double result3 = pointPolygonTest(contours, a, false);
//        double result4 = pointPolygonTest(contours, b, false);
//        if(result1 == 1 || result2 == 1 || result3 == 1 || result4 == 1){
//            return true;
//        }
//        return false;
//        std::cout<<result<<std::endl;
    }

    bool area(cv::Mat &img_, std::string &name, cv::Point &pt_lt, cv::Point &pt_br, cv::Scalar &color) {
        cv::Point a = cv::Point(pt_lt.x,pt_br.y);
        cv::Point b = cv::Point(pt_br.x,pt_lt.y);
        double result1 = pointPolygonTest(contours2, pt_lt, false);
        double result2 = pointPolygonTest(contours2, pt_br, false);
        double result3 = pointPolygonTest(contours2, a, false);
        double result4 = pointPolygonTest(contours2, b, false);
        if(result1 == 1 || result2 == 1 || result3 == 1 || result4 == 1){
            return true;
        }
        return false;
    }

    void getResult(std::pair<std::string, int> &result, cv::Mat &img, std::vector<Object> &Obj_pool) {
//        int count = 0;
        int count = Obj_pool.size();
//        int count2 = 0;
        int num;
        //cv::line(img,cv::Point(0, 80),cv::Point(1280, 80),cv::Scalar(0, 0, 255),3);
        //cv::line(img,cv::Point(350, 720),cv::Point(550, 80),cv::Scalar(0, 0, 255),3);
//        cv::line(img_,cv::Point(550, 80),cv::Point(730, 80),cv::Scalar(0, 0, 255),3);
        //cv::line(img,cv::Point(730, 80),cv::Point(930, 720),cv::Scalar(0, 0, 255),3);
        for (size_t i = 0; i < Obj_pool.size(); i++){
            cv::Point tl = Obj_pool[i].boundingbox.tl();
            cv::Point br = cv::Point(tl.x + Obj_pool[i].boundingbox.width,tl.y + Obj_pool[i].boundingbox.height);
//            if(draw_bb_top(img, Object_Detection::CLASSES[Obj_pool[i].label], tl, br, Object_Detection::COLOR[Obj_pool[i].label])){
//                count++;
//            }
//            if(area(img, Object_Detection::CLASSES[Obj_pool[i].label], tl, br, Object_Detection::COLOR[Obj_pool[i].label])){
//                count2++;
//            }
            draw_bb_top(img, Object_Detection::CLASSES[Obj_pool[i].label], tl, br, Object_Detection::COLOR[Obj_pool[i].label]);
        }
        if(count <= 10){
            getMemory(0,num);
            result.first = Object_Detection::STATUS[num];
        }else if(count < 15){
            getMemory(1,num);
            result.first = Object_Detection::STATUS[num];
        }else{
            getMemory(2,num);
            result.first = Object_Detection::STATUS[num];
        }
        result.second = count;
    }

    void getMemory(int i, int &num) {
        std::cout<<"memory_size  "<<memory.size()<<std::endl;
        if(memory.size()==10){
            memory.pop_front();
        }
        memory.emplace_back(i);
        int p[3] = {0, 0, 0};
        int size = memory.size();
        for(int i = 0; i< size; i++){
            p[memory[i]]++;
        }
        int max = 0;
        for(int n=0;n<3;n++){
            if(p[n]>max){
                max = p[n];
                num = n;
            }
        }
    }

}
