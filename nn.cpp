/**
  ******************************************************************************
  * File Name          : nn.cpp
  * Description        : Neural Network
  ******************************************************************************
  */
#include <math.h>
#include <stdlib.h>
#include "nn.h"

nLayer::nLayer(int input, int output){
  
    nLayer::input_n = input;
    nLayer::output_n = output;
    nLayer::x = (float *)malloc(input * sizeof(float));
    nLayer::w = (float *)malloc(input * output * sizeof(float));
    nLayer::a = (float *)malloc(output * sizeof(float));      
  
}

nLayer::~nLayer(){
    
  free(nLayer::x);
  free(nLayer::w);
  free(nLayer::a);
  
}

void nLayer::set_w(float* ws){
  nLayer::w = ws;
}

float* nLayer::get_a(){
  return nLayer::a;
}

float* nLayer::get_w(){
  return nLayer::w;
}

void nLayer::init(){
  int i;
  for(i=0;i<nLayer::input_n*nLayer::output_n;i++){
    nLayer::w[i] = (float)rand()/RAND_MAX;
  }
}

float nLayer::maxa(){
    int i;
    float c=0;
    for(i=0;i<nLayer::output_n;i++){
      if(c<nLayer::a[i]) c = nLayer::a[i];
    }
    return c;
}

void nLayer::fwd(float* in){
    int i,j;
    int size;

    size = sizeof(in)/sizeof(float);
    if(size == nLayer::input_n){
      for(i=0;i<nLayer::output_n;i++){
        *(nLayer::a+i)=0;
        for(j=0;j<nLayer::input_n;j++){
          *(nLayer::a+i)+=*(nLayer::w+i+(j*nLayer::input_n)) * *(nLayer::x+j);
        }
      }
    }
}

void nLayer::sigmoid(float* x){
    int i;
    nLayer::fwd(x);
    for(i=0;i<nLayer::output_n;i++){
      nLayer::a[i]=1/(1+exp(-nLayer::a[i]));
    }
}

void nLayer::step(float* x){
    int i;
    nLayer::fwd(x);
    for(i=0;i<nLayer::output_n;i++){
      if(nLayer::a[i] > 0) nLayer::a[i]=1;
      else nLayer::a[i] =0;
    }
}

void nLayer::softmax(float* x){
    int i;
    float c;
    float sum_exp_a=0;
    nLayer::fwd(x);
    c = nLayer::maxa();
    for(i=0;i<nLayer::output_n;i++){
      sum_exp_a+=exp(nLayer::a[i]-c);
    }
    for(i=0;i<nLayer::output_n;i++){
      nLayer::a[i]=exp(nLayer::a[i]-c)/sum_exp_a;
    }
}

void nLayer::relu(float* x){
    int i;
    nLayer::fwd(x);
    for(i=0;i<nLayer::output_n;i++){
      if(nLayer::a[i] < 0) nLayer::a[i]=0;
    }  
}

