//
// 
//

#include <Object_detect.h>
#include <string>
#include <ObjProcess.h>
#include <iostream>

int main(int argc, const char** argv)
{

    Object_Detection::RUN_MODE mode = Object_Detection::RUN_MODE::FLOAT32;
    //if(options["mode"] == "0" ) mode = Objece_detect::RUN_MODE::FLOAT32;
    //if(options["mode"] == "1" ) mode = Objece_detect::RUN_MODE::FLOAT16;
    //if(options["mode"] == "2" ) mode = Objece_detect::RUN_MODE::INT8;

    Object_Detection::Object_detect net("model/model_head.onnx", "" ,mode);
    net.saveEngine("model/model_head.engine");

}