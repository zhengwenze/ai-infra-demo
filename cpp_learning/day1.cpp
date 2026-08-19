#include <iostream>
#include <string>
#include <vector>

using namespace std;

// ============================================================
// 1. 定义一个推理请求
// ============================================================

struct Request {
    int id;             // 请求 ID
    string prompt;      // 用户输入
    int max_tokens;     // 最大生成 token 数
};


// ============================================================
// 2. 函数 + const 引用
// ============================================================

// const Request& 表示：
// 1. 不复制 Request 对象
// 2. 函数内部不能修改 req
void printRequest(const Request& req) {
    cout << "Request ID: " << req.id << endl;
    cout << "Prompt: " << req.prompt << endl;
    cout << "Max Tokens: " << req.max_tokens << endl;
}


// ============================================================
// 3. 引用
// ============================================================

// Request& 是引用。
// 修改 req 就相当于修改原来的 Request。
void updateMaxTokens(Request& req, int new_max_tokens) {
    req.max_tokens = new_max_tokens;
}


// ============================================================
// 4. 指针
// ============================================================

// Request* 是指针。
// 指针保存的是 Request 对象的内存地址。
void updatePrompt(Request* req, const string& new_prompt) {
    // 指针可能为空，因此通常要检查
    if (req != nullptr) {
        req->prompt = new_prompt;
    }
}


// ============================================================
// main
// ============================================================

int main() {

    // ========================================================
    // 一、变量
    // ========================================================

    int request_count = 3;

    string model_name = "Qwen2.5-7B";

    const int MAX_BATCH_SIZE = 16;

    cout << "Model: " << model_name << endl;
    cout << "Request count: " << request_count << endl;
    cout << "Max batch size: " << MAX_BATCH_SIZE << endl;

    cout << "--------------------------------" << endl;


    // ========================================================
    // 二、栈对象
    // ========================================================

    // req1 是局部变量。
    // 一般情况下存放在栈上。
    //
    // main 函数结束后，它会自动销毁。
    Request req1{
        1,
        "你好，请介绍一下 AI Infra",
        128
    };

    printRequest(req1);

    cout << "--------------------------------" << endl;


    // ========================================================
    // 三、引用
    // ========================================================

    Request& reference = req1;

    // reference 和 req1 实际上代表同一个对象
    reference.max_tokens = 256;

    cout << "修改引用后：" << endl;

    cout << "req1.max_tokens = "
         << req1.max_tokens
         << endl;

    cout << "--------------------------------" << endl;


    // ========================================================
    // 四、指针
    // ========================================================

    Request* pointer = &req1;

    cout << "req1 地址：" << &req1 << endl;

    cout << "pointer 保存的地址："
         << pointer
         << endl;

    cout << "通过指针读取 prompt：" << endl;

    cout << pointer->prompt << endl;


    // 通过指针修改对象

    pointer->prompt = "解释一下 KV Cache";

    cout << "修改之后：" << endl;

    cout << req1.prompt << endl;

    cout << "--------------------------------" << endl;


    // ========================================================
    // 五、函数参数：引用和指针
    // ========================================================

    updateMaxTokens(req1, 512);

    updatePrompt(
        &req1,
        "解释 vLLM Continuous Batching"
    );

    printRequest(req1);

    cout << "--------------------------------" << endl;


    // ========================================================
    // 六、vector<Request>
    // ========================================================

    vector<Request> requests;

    requests.push_back(
        {
            1,
            "什么是 KV Cache？",
            128
        }
    );

    requests.push_back(
        {
            2,
            "什么是 Continuous Batching？",
            256
        }
    );

    requests.push_back(
        {
            3,
            "什么是 PagedAttention？",
            512
        }
    );


    cout << "当前共有 "
         << requests.size()
         << " 个推理请求"
         << endl;


    // ========================================================
    // 七、遍历 vector
    // ========================================================

    for (const Request& req : requests) {

        cout << "--------------------------------" << endl;

        printRequest(req);
    }


    // ========================================================
    // 八、堆对象
    // ========================================================

    cout << "================================" << endl;
    cout << "创建堆对象" << endl;


    Request* heap_request = new Request{
        100,
        "这是一个堆上的推理请求",
        1024
    };


    cout << "堆对象地址："
         << heap_request
         << endl;


    printRequest(*heap_request);


    // ========================================================
    // 九、释放堆内存
    // ========================================================

    delete heap_request;

    // 避免悬空指针
    heap_request = nullptr;


    cout << "heap_request 已释放" << endl;


    return 0;
}