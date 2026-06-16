// WebSocket streaming ASR server for the X-ASR zh-en zipformer2 transducer,
// protocol-compatible with the sherpa-onnx X-ASR deployment server
// (https://github.com/Gilgamesh-J/X-ASR .../deployment/README.md#websocket-protocol).
//
// Client -> server:
//   {"type":"start","sample_rate":16000}   binary int16-LE PCM 16k mono   {"type":"end"}
//   {"type":"reset"}                       {"type":"ping"}
// Server -> client:
//   {"type":"started","sample_rate":16000}  {"type":"partial","text":"..."}
//   {"type":"final","text":"...","first_partial_latency":N}
//   {"type":"reset_ok"}                     {"type":"pong"}
//
// Zero external deps: raw POSIX sockets + a minimal WebSocket (RFC 6455) layer.
//   Usage:  rs-xasr-ws-server <model.gguf> [port]      (default port 6006)
#include "arch/xasr_zipformer2.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <arpa/inet.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// ----------------------------------------------------------------- SHA1 + base64
struct SHA1 {
  uint32_t h[5]; uint64_t len = 0; uint8_t buf[64]; int n = 0;
  SHA1() { h[0]=0x67452301; h[1]=0xEFCDAB89; h[2]=0x98BADCFE; h[3]=0x10325476; h[4]=0xC3D2E1F0; }
  static uint32_t rol(uint32_t v, int b){ return (v<<b)|(v>>(32-b)); }
  void block(const uint8_t *p){
    uint32_t w[80];
    for(int i=0;i<16;++i) w[i]=(p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];
    for(int i=16;i<80;++i) w[i]=rol(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
    for(int i=0;i<80;++i){ uint32_t f,k;
      if(i<20){f=(b&c)|(~b&d);k=0x5A827999;}
      else if(i<40){f=b^c^d;k=0x6ED9EBA1;}
      else if(i<60){f=(b&c)|(b&d)|(c&d);k=0x8F1BBCDC;}
      else {f=b^c^d;k=0xCA62C1D6;}
      uint32_t t=rol(a,5)+f+e+k+w[i]; e=d;d=c;c=rol(b,30);b=a;a=t; }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;
  }
  void add(const void*data,size_t l){ const uint8_t*p=(const uint8_t*)data; len+=l;
    while(l){ buf[n++]=*p++; --l; if(n==64){block(buf);n=0;} } }
  void final(uint8_t out[20]){ uint64_t bits=len*8; uint8_t pad=0x80; add(&pad,1);
    uint8_t z=0; while(n!=56) add(&z,1);
    uint8_t lb[8]; for(int i=0;i<8;++i) lb[i]=bits>>((7-i)*8); block_finish(lb);
    for(int i=0;i<5;++i){ out[i*4]=h[i]>>24;out[i*4+1]=h[i]>>16;out[i*4+2]=h[i]>>8;out[i*4+3]=h[i]; } }
  void block_finish(const uint8_t lb[8]){ for(int i=0;i<8;++i) buf[n++]=lb[i]; block(buf); }
};
static std::string b64(const uint8_t *d, int n){
  static const char *t="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o; for(int i=0;i<n;i+=3){ int v=d[i]<<16; if(i+1<n)v|=d[i+1]<<8; if(i+2<n)v|=d[i+2];
    o+=t[(v>>18)&63]; o+=t[(v>>12)&63]; o+= i+1<n? t[(v>>6)&63]:'='; o+= i+2<n? t[v&63]:'='; }
  return o;
}

// ----------------------------------------------------------------- socket I/O
static bool send_all(int fd,const void*b,size_t l){ const char*p=(const char*)b;
  while(l){ ssize_t k=send(fd,p,l,0); if(k<=0) return false; p+=k; l-=k; } return true; }
static bool recv_all(int fd,void*b,size_t l){ char*p=(char*)b;
  while(l){ ssize_t k=recv(fd,p,l,0); if(k<=0) return false; p+=k; l-=k; } return true; }

// minimal WS frame: returns opcode, fills payload. false on close/error.
static bool ws_read(int fd, int &opcode, std::vector<uint8_t> &payload){
  uint8_t h[2]; if(!recv_all(fd,h,2)) return false;
  opcode = h[0]&0x0f; bool masked = h[1]&0x80; uint64_t len = h[1]&0x7f;
  if(len==126){ uint8_t e[2]; if(!recv_all(fd,e,2))return false; len=(e[0]<<8)|e[1]; }
  else if(len==127){ uint8_t e[8]; if(!recv_all(fd,e,8))return false; len=0; for(int i=0;i<8;++i)len=(len<<8)|e[i]; }
  uint8_t mask[4]={0,0,0,0}; if(masked && !recv_all(fd,mask,4)) return false;
  payload.resize(len); if(len && !recv_all(fd,payload.data(),len)) return false;
  if(masked) for(uint64_t i=0;i<len;++i) payload[i]^=mask[i&3];
  return true;
}
static bool ws_send(int fd,int opcode,const void*data,size_t len){
  uint8_t h[10]; int n=0; h[n++]=0x80|opcode;
  if(len<126) h[n++]=len;
  else if(len<65536){ h[n++]=126; h[n++]=len>>8; h[n++]=len; }
  else { h[n++]=127; for(int i=7;i>=0;--i) h[n++]=(len>>(i*8))&0xff; }
  return send_all(fd,h,n) && (len==0 || send_all(fd,data,len));
}
static bool ws_text(int fd,const std::string&s){ return ws_send(fd,1,s.data(),s.size()); }

// ----------------------------------------------------------------- gguf load
struct Model {
  ggml_context *ctx=nullptr; gguf_context *gg=nullptr;
  ggml_backend_t backend=nullptr; ggml_backend_buffer_t buf=nullptr;
  std::map<std::string,ggml_tensor*> w;
  std::unordered_map<int,std::string> id2tok;
  XAsrHParams hp;
};
static bool load_model(const char*path, Model &m){
  gguf_init_params gp{true,&m.ctx}; m.gg=gguf_init_from_file(path,gp);
  if(!m.gg){ fprintf(stderr,"gguf load failed\n"); return false; }
  m.backend=ggml_backend_cpu_init();
  m.buf=ggml_backend_alloc_ctx_tensors(m.ctx,m.backend);
  FILE*f=fopen(path,"rb"); size_t off=gguf_get_data_offset(m.gg);
  int n=gguf_get_n_tensors(m.gg); std::vector<char> rb;
  for(int i=0;i<n;++i){ const char*nm=gguf_get_tensor_name(m.gg,i);
    ggml_tensor*t=ggml_get_tensor(m.ctx,nm); if(!t)continue;
    size_t to=gguf_get_tensor_offset(m.gg,i), sz=ggml_nbytes(t);
    if(rb.size()<sz) rb.resize(sz); fseek(f,off+to,SEEK_SET);
    if(fread(rb.data(),1,sz,f)!=sz){ fclose(f); return false; }
    ggml_backend_tensor_set(t,rb.data(),0,sz); m.w[nm]=t; }
  fclose(f);
  int tk=gguf_find_key(m.gg,"tokenizer.ggml.tokens");
  if(tk>=0){ int nt=gguf_get_arr_n(m.gg,tk);
    for(int i=0;i<nt;++i) m.id2tok[i]=gguf_get_arr_str(m.gg,tk,i); }
  xasr_read_hparams(m.gg, m.hp);
  return true;
}

// detokenize hyp ids -> text (▁ -> space, drop spaces between CJK).
static bool is_cjk(unsigned c){ return c>=0x3400; }
static std::string detok(const Model&m,const std::vector<int32_t>&hyp,int ctx){
  std::string s;
  for(size_t i=ctx;i<hyp.size();++i){ auto it=m.id2tok.find(hyp[i]);
    if(it==m.id2tok.end())continue; std::string t=it->second;
    size_t p; while((p=t.find("\xe2\x96\x81"))!=std::string::npos) t.replace(p,3," ");
    s+=t; }
  // trim leading space + collapse " " between two CJK runs
  std::string o; for(size_t i=0;i<s.size();++i){
    if(s[i]==' '){ // drop leading, and any space adjacent to a CJK/fullwidth char
      bool lcjk=!o.empty() && (unsigned char)o.back()>=0x80;
      bool rcjk=i+1<s.size() && (unsigned char)s[i+1]>=0x80;
      if(o.empty() || lcjk || rcjk) continue; }
    o+=s[i]; }
  (void)is_cjk; return o;
}

// crude JSON field reads (sufficient for this protocol).
static std::string json_type(const std::string&j){
  size_t p=j.find("\"type\""); if(p==std::string::npos) return "";
  p=j.find(':',p); p=j.find('"',p); if(p==std::string::npos) return "";
  size_t q=j.find('"',p+1); return j.substr(p+1,q-p-1);
}

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: %s <model.gguf> [port]\n",argv[0]); return 1; }
  int port = argc>2? atoi(argv[2]) : 6006;
  Model m; if(!load_model(argv[1],m)) return 1;
  fprintf(stderr,"loaded %zu tensors, %zu tokens\n",m.w.size(),m.id2tok.size());

  int srv=socket(AF_INET,SOCK_STREAM,0); int one=1;
  setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
  sockaddr_in a{}; a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
  if(bind(srv,(sockaddr*)&a,sizeof a)<0){ perror("bind"); return 1; }
  listen(srv,4);
  fprintf(stderr,"X-ASR WebSocket server on ws://0.0.0.0:%d\n",port);

  for(;;){
    int c=accept(srv,nullptr,nullptr); if(c<0) continue;
    // --- HTTP upgrade handshake ---
    std::string req; char tmp[1024]; bool ok=false;
    while(req.find("\r\n\r\n")==std::string::npos){ ssize_t k=recv(c,tmp,sizeof tmp,0); if(k<=0)break; req.append(tmp,k); }
    size_t kp=req.find("Sec-WebSocket-Key:");
    if(kp!=std::string::npos){ kp=req.find_first_not_of(" ",kp+18);
      size_t ke=req.find("\r\n",kp); std::string key=req.substr(kp,ke-kp);
      key+="258EAFA5-E914-47DA-95CA-C5AB0DC85B11"; SHA1 s; s.add(key.data(),key.size());
      uint8_t dg[20]; s.final(dg);
      std::string resp="HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "+b64(dg,20)+"\r\n\r\n";
      ok=send_all(c,resp.data(),resp.size()); }
    if(!ok){ close(c); continue; }

    XAsrOnlineStream stream(m.w,m.backend,m.hp);
    std::string last_partial;
    int op; std::vector<uint8_t> pl;
    while(ws_read(c,op,pl)){
      if(op==0x8){ break; }                       // close
      if(op==0x9){ ws_send(c,0xA,pl.data(),pl.size()); continue; } // ping->pong
      if(op==0x2){ // binary PCM int16-LE
        stream.AcceptPcm((const int16_t*)pl.data(), (int)(pl.size()/2));
        std::string t=detok(m,stream.Hyp(),stream.ContextSize());
        if(t!=last_partial){ last_partial=t; ws_text(c,"{\"type\":\"partial\",\"text\":\""+t+"\"}"); }
        continue; }
      if(op==0x1){ // text control
        std::string j((char*)pl.data(),pl.size()); std::string ty=json_type(j);
        if(ty=="start"){ stream.Reset(); last_partial.clear(); ws_text(c,"{\"type\":\"started\",\"sample_rate\":16000}"); }
        else if(ty=="end"){ stream.InputFinished();
          std::string t=detok(m,stream.Hyp(),stream.ContextSize());
          char fpl[64]; snprintf(fpl,sizeof fpl,"%.3f",stream.FirstPartialLatencySec()<0?0.0:stream.FirstPartialLatencySec());
          ws_text(c,"{\"type\":\"final\",\"text\":\""+t+"\",\"first_partial_latency\":"+fpl+"}"); }
        else if(ty=="reset"){ stream.Reset(); last_partial.clear(); ws_text(c,"{\"type\":\"reset_ok\"}"); }
        else if(ty=="ping"){ ws_text(c,"{\"type\":\"pong\"}"); }
        continue; }
    }
    close(c);
  }
  return 0;
}
