#pragma once

#include <cstddef>
#include <experimental/mdspan>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace stdex = std::experimental;

namespace egg
{

/// @name Non-owning multidimensional views
/// Trivially copyable, so capturable by value into a SYCL kernel.
/// @{
template <class T> using View1 = stdex::mdspan<T, stdex::dextents<std::size_t, 1>>;
template <class T> using View2 = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;
template <class T> using View3 = stdex::mdspan<T, stdex::dextents<std::size_t, 3>>;
/// @}

/// Move-only RAII owner of a device USM array of T on a given queue.
template <class T> class UsmBuffer
{
  public:
    UsmBuffer() = default;

    UsmBuffer(sycl::queue q, std::size_t n) :
        q_(q), n_(n), ptr_(n ? sycl::malloc_device<T>(n, q) : nullptr)
    {
    }

    /// Allocate and upload a host vector in one step.
    UsmBuffer(sycl::queue q, const std::vector<T>& host) : UsmBuffer(q, host.size())
    {
        if (n_) { q_.memcpy(ptr_, host.data(), n_ * sizeof(T)).wait(); }
    }

    ~UsmBuffer() { reset(); }

    UsmBuffer(UsmBuffer&& o) noexcept { swap(o); }
    UsmBuffer& operator=(UsmBuffer&& o) noexcept
    {
        if (this != &o) {
            reset();
            swap(o);
        }
        return *this;
    }
    UsmBuffer(const UsmBuffer&) = delete;
    UsmBuffer& operator=(const UsmBuffer&) = delete;

    T* data() const { return ptr_; }
    std::size_t size() const { return n_; }

    void upload(const std::vector<T>& host)
    {
        if (host.size()) { q_.memcpy(ptr_, host.data(), host.size() * sizeof(T)).wait(); }
    }
    void download(T* host)
    {
        if (n_) { q_.memcpy(host, ptr_, n_ * sizeof(T)).wait(); }
    }

  private:
    void reset()
    {
        if (ptr_) { sycl::free(ptr_, q_); }
        ptr_ = nullptr;
        n_ = 0;
    }
    void swap(UsmBuffer& o) noexcept
    {
        std::swap(q_, o.q_);
        std::swap(n_, o.n_);
        std::swap(ptr_, o.ptr_);
    }

    // `sycl::queue` is a reference-counted handle; enqueueing work (memcpy in the
    // const download path) does not change this buffer's logical state.
    mutable sycl::queue q_ {};
    std::size_t n_ {};
    T* ptr_ {};
};

}  // namespace egg
