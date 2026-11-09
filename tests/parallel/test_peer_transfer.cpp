#include "core/peer_transfer.h"
#include "core/device_group.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
// GNU linker wrappers are test-only fault injection and allocation accounting.
// No production hook or invalid CUDA handle is required.
namespace {
bool account_pinned=false;
void* slot_pointer=nullptr;
int slot_allocations=0, slot_releases=0, slot_release_successes=0;
cudaStream_t fail_query_stream=nullptr;
bool injected=false, d2h_submitted=false;
void account_allocation(void* pointer, cudaError_t status) {
    if (account_pinned && status==cudaSuccess) { slot_pointer=pointer; ++slot_allocations; }
}
}
extern "C" cudaError_t CUDARTAPI __real_cudaMallocHost(void**,size_t);
extern "C" cudaError_t CUDARTAPI __real_cudaHostAlloc(void**,size_t,unsigned int);
extern "C" cudaError_t CUDARTAPI __real_cudaFreeHost(void*);
extern "C" cudaError_t CUDARTAPI __real_cudaStreamGetDevice(cudaStream_t,int*);
extern "C" cudaError_t CUDARTAPI __real_cudaMemcpyAsync(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t);
extern "C" cudaError_t CUDARTAPI __wrap_cudaMallocHost(void** pointer,size_t bytes) {
    auto status=__real_cudaMallocHost(pointer,bytes); if(status==cudaSuccess) account_allocation(*pointer,status); return status;
}
extern "C" cudaError_t CUDARTAPI __wrap_cudaHostAlloc(void** pointer,size_t bytes,unsigned int flags) {
    auto status=__real_cudaHostAlloc(pointer,bytes,flags); if(status==cudaSuccess) account_allocation(*pointer,status); return status;
}
extern "C" cudaError_t CUDARTAPI __wrap_cudaFreeHost(void* pointer) {
    const bool tracked=account_pinned && pointer==slot_pointer;
    if (tracked) ++slot_releases;
    auto status=__real_cudaFreeHost(pointer);
    if (tracked && status==cudaSuccess) ++slot_release_successes;
    return status;
}
extern "C" cudaError_t CUDARTAPI __wrap_cudaStreamGetDevice(cudaStream_t stream,int* device) {
    if (fail_query_stream && stream==fail_query_stream) {
        fail_query_stream=nullptr; injected=true;
        return cudaErrorInvalidResourceHandle;
    }
    return __real_cudaStreamGetDevice(stream,device);
}
extern "C" cudaError_t CUDARTAPI __wrap_cudaMemcpyAsync(void* destination,const void* source,
    size_t bytes,cudaMemcpyKind kind,cudaStream_t stream) {
    auto status=__real_cudaMemcpyAsync(destination,source,bytes,kind,stream);
    if (account_pinned && kind==cudaMemcpyDeviceToHost && status==cudaSuccess) d2h_submitted=true;
    return status;
}
using namespace ninfer;
static void require(bool b) { if(!b) throw std::runtime_error("transfer assertion"); }
int main() {
 try {
    int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count<2) return 77;
    DeviceGroup group({0,1});
    constexpr size_t size=4097;
    DeviceBuffer source, destination;
    { auto bind=group.bind(0); source=DeviceBuffer(size); }
    { auto bind=group.bind(1); destination=DeviceBuffer(size); }
    for(auto policy:{TransferPolicy::pinned_host,TransferPolicy::automatic}) {
      PeerTransfer transfer(group.context(0),group.context(1),size,policy);
      constexpr int rounds=8;
      PinnedHostBuffer input(size*rounds), result(size*rounds);
      auto* payload=static_cast<unsigned char*>(input.data());
      for(int round=0;round<rounds;++round)
        for(size_t i=0;i<size;++i) payload[round*size+i]=(i*37+round*19)&255;
      // No host synchronization between rounds: exercise source, destination and
      // pinned-slot reuse dependencies with consumers still queued.
      for(int round=0;round<rounds;++round) {
        transfer.source_complete().wait(group.context(0).stream);
        { auto bind=group.bind(0); CUDA_CHECK(cudaMemcpyAsync(source.p,payload+round*size,size,cudaMemcpyHostToDevice,group.context(0).stream)); }
        transfer.enqueue(destination.p,source.p,size,group.context(0).stream,group.context(1).stream);
        { auto bind=group.bind(1); CUDA_CHECK(cudaMemcpyAsync(static_cast<unsigned char*>(result.data())+round*size,destination.p,size,cudaMemcpyDeviceToHost,group.context(1).stream)); }
      }
      group.context(1).synchronize();
      require(std::equal(payload,payload+size*rounds,static_cast<unsigned char*>(result.data())));
      bool rejected=false;
      try { transfer.enqueue(source.p,destination.p,size,group.context(0).stream,group.context(1).stream); }
      catch(const std::invalid_argument&) { rejected=true; } require(rejected);
      rejected=false;
      try { transfer.enqueue(destination.p,source.p,size+1,group.context(0).stream,group.context(1).stream); }
      catch(const std::out_of_range&) { rejected=true; } require(rejected);
      rejected=false;
      try { transfer.enqueue(destination.p,source.p,size,nullptr,group.context(1).stream); }
      catch(const std::invalid_argument&) { rejected=true; } require(rejected);
      transfer.enqueue(destination.p,source.p,size,group.context(0).stream,group.context(1).stream);
      CUDA_CHECK(cudaSetDevice(1)); // destructor must drain both endpoints before freeing staging.
    }
    account_pinned=true;
    {
      PeerTransfer failed(group.context(0),group.context(1),size,TransferPolicy::pinned_host);
      require(slot_allocations==1 && slot_releases==0);
      fail_query_stream=group.context(1).transfer_stream;
      bool caught=false;
      try { failed.enqueue(destination.p,source.p,size,group.context(0).stream,group.context(1).stream); }
      catch(const std::runtime_error&) { caught=true; }
      require(caught && injected && d2h_submitted);
      require(failed.source_complete().ready());
      { auto bind=group.bind(0); require(cudaStreamQuery(group.context(0).transfer_stream)==cudaSuccess); }
      (void)cudaGetLastError();
      caught=false;
      try { failed.enqueue(destination.p,source.p,size,group.context(0).stream,group.context(1).stream); }
      catch(const std::logic_error&) { caught=true; }
      require(caught && slot_releases==0);
    }
    require(slot_allocations==1 && slot_releases==1 && slot_release_successes==1);
    account_pinned=false;
    {
      PeerTransfer fresh(group.context(0),group.context(1),size,TransferPolicy::pinned_host);
      fresh.enqueue(destination.p,source.p,size,group.context(0).stream,group.context(1).stream);
      fresh.synchronize();
      std::vector<unsigned char> left(size),right(size);
      source.copy_to_host(left.data(),size); destination.copy_to_host(right.data(),size);
      require(left==right);
    }
    return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
