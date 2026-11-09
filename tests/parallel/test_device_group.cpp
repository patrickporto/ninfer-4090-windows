#include "core/device_group.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include <stdexcept>
#include <utility>
#include <iostream>
using namespace ninfer;
static void require(bool b) { if (!b) throw std::runtime_error("device lifetime assertion"); }
static int current() { int d; CUDA_CHECK(cudaGetDevice(&d)); return d; }
int main() {
    try {
        bool rejected = false;
        try { DeviceGroup invalid({0, 0}); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected);
        int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count<2) return 77;
        CUDA_CHECK(cudaSetDevice(1));
        DeviceGroup group({0,1}); require(current()==1);
        {
            auto binding=group.bind(0); require(current()==0);
            DeviceBuffer a(4096); DeviceArena arena(4096);
            CudaCompletionEvent event(group.context(0));
            CudaEventTimer timer(group.context(0));
            DecodeGraphDefinition graph;
            graph.capture(group.context(0).stream,[&] { CUDA_CHECK(cudaMemsetAsync(a.p,7,a.bytes,group.context(0).stream)); });
            DecodeGraphExecutable executable; executable.instantiate(graph);
            executable.launch(group.context(0).stream);
            CUDA_CHECK(cudaSetDevice(1));
            DeviceBuffer moved(std::move(a)); require(moved.device()==0);
            DeviceArena moved_arena(std::move(arena)); require(moved_arena.device()==0);
            CudaCompletionEvent moved_event(std::move(event));
            // Default-stream sentinels belong to the caller's current device,
            // before the event owner is bound. Never redirect them to device 0.
            for (auto sentinel : {cudaStreamPerThread, cudaStreamLegacy}) {
                bool mismatched_default=false;
                try { moved_event.record(sentinel); }
                catch(const std::invalid_argument&) { mismatched_default=true; }
                require(mismatched_default);
                require(current()==1);
            }
            moved_event.record(group.context(0).stream); moved_event.synchronize();
            moved_event.wait(group.context(1).stream);
            bool wrong_stream=false;
            try { moved_event.record(group.context(1).stream); }
            catch(const std::invalid_argument&) { wrong_stream=true; }
            require(wrong_stream);
            DecodeGraphDefinition moved_graph(std::move(graph));
            DecodeGraphExecutable moved_executable(std::move(executable));
            wrong_stream=false;
            try { moved_executable.launch(group.context(1).stream); }
            catch(const std::invalid_argument&) { wrong_stream=true; }
            require(wrong_stream);
            moved_executable.launch(group.context(0).stream);
            timer.start(); timer.stop_ms(); require(current()==1);
            group.context(0).synchronize(); require(current()==1);
        }
        require(current()==1);
        { auto binding=group.bind(0); DeviceBuffer a(64); CUDA_CHECK(cudaSetDevice(1));
          a.fill(42); unsigned char out[64]; a.copy_to_host(out,64);
          for(auto v:out) { require(v==42); }
          require(current()==1); }
        {
            auto binding=group.bind(0);
            bool null_timer=false;
            try { CudaEventTimer invalid(group.context(0),nullptr); }
            catch(const std::invalid_argument&) { null_timer=true; }
            require(null_timer); // Original timer contract rejected the default stream.
            CudaCompletionEvent event(group.context(0));
            bool null_record=false, null_wait=false;
            try { event.record(nullptr); } catch(const std::logic_error&) { null_record=true; }
            try { event.wait(nullptr); } catch(const std::logic_error&) { null_wait=true; }
            require(null_record && null_wait); // Preserve original completion-event contract.
            DeviceBuffer buffer(64);
            DecodeGraphDefinition graph;
            // CUDA prohibits capture on the legacy default stream. The original
            // wrapper accepted an explicit per-thread default-stream sentinel.
            graph.capture(cudaStreamPerThread,[&] {
                CUDA_CHECK(cudaMemsetAsync(buffer.p,19,buffer.bytes,cudaStreamPerThread));
            });
            DecodeGraphExecutable executable;
            executable.instantiate(graph);
            executable.upload(nullptr);
            executable.launch(nullptr);
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
            unsigned char result[64]; buffer.copy_to_host(result,64);
            for(auto value:result) { require(value==19); }
            require(cuda_stream_device(nullptr)==0);
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
