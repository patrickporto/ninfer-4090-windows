#include "core/cuda_error_scope.h"
#include "core/device.h"

#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#ifdef __linux__
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
void throws_with_context() {
    try {
        ninfer::cuda_check(cudaErrorInvalidValue, "injected operation", "fake.cu", 17);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("fake.cu:17") == std::string::npos ||
            message.find("injected operation") == std::string::npos ||
            message.find("cudaErrorInvalidValue") == std::string::npos)
            throw std::runtime_error("CUDA exception lost its diagnostic context");
        return;
    }
    throw std::runtime_error("CUDA failure was ignored");
}

#ifdef __linux__
void default_policy_aborts(bool new_thread) {
    const auto pid = fork();
    if (pid < 0) throw std::runtime_error("fork failed");
    if (pid == 0) {
        const rlimit limit{0, 0};
        (void)setrlimit(RLIMIT_CORE, &limit);
        const auto fail = [] {
            try {
                ninfer::cuda_check(cudaErrorInvalidValue, "default policy", "fake.cu", 21);
            } catch (...) {
                // An uncaught C++ exception also aborts. Distinguish it from the
                // required fatal policy to detect leaked or process-global scopes.
                _exit(97);
            }
        };
        if (new_thread) { std::thread worker(fail); worker.join(); }
        else fail();
        _exit(99);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT)
        throw std::runtime_error("default CUDA policy did not preserve abort semantics");
}
#endif
}

int main() {
    try {
        ninfer::cuda_check(cudaSuccess, "success", "fake.cu", 1);
#ifdef __linux__
        default_policy_aborts(false);
#endif
        {
            ninfer::ScopedCudaErrors outer;
            throws_with_context();
            {
                ninfer::ScopedCudaErrors inner;
                throws_with_context();
                ninfer::ScopedCudaErrors inert(false);
                throws_with_context();
            }
            throws_with_context();
#ifdef __linux__
            // A new rank thread must opt in explicitly; caller TLS is not inherited.
            default_policy_aborts(true);
#endif
        }
#ifdef __linux__
        default_policy_aborts(false);
#endif
        std::exception_ptr worker_error;
        std::thread worker([&] {
            try { ninfer::ScopedCudaErrors scope; throws_with_context(); }
            catch (...) { worker_error = std::current_exception(); }
        });
        worker.join();
        if (worker_error) std::rethrow_exception(worker_error);
#ifdef __linux__
        default_policy_aborts(false);
#endif
        std::cout << "PASS scoped CUDA failures, nesting, thread isolation and default policy\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
