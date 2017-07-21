/**
  ******************************************************************************
  * File Name          : nn.h
  * Description        : Neural Network 
  ******************************************************************************
  */

class nLayer{
  private:
    int input_n;
    int output_n;
    float *w;
    float *x;
    float *a;
    void fwd(float* in);
    void bwd();
    float maxa();

  public:
    nLayer(int input, int output);  
    ~nLayer();
    float* get_w();
    float* get_a();
    void set_w(float* ws);
    void init();
    void sigmoid(float *x);
    void softmax(float *x);
    void relu(float *x);
    void step(float *x);

};

