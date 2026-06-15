#include "frontend/matcha_frontend.h"
#include <cstdio>
int main(int argc, char** argv){
  MatchaFrontend fe;
  if(!fe.Load(argv[1], argv[2])){ printf("LOAD FAIL\n"); return 1; }
  for(int i=3;i<argc;i++){
    int sk=0; auto ids=fe.TextToIds(argv[i],&sk);
    printf("TEXT %s\nIDS", argv[i]);
    for(auto v:ids) printf(" %d", v);
    printf("\nSKIP %d\n", sk);
  }
  return 0;
}
