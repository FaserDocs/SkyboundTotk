// boost::container::static_vector for suyu's code: a std::vector (capacity and options are not enforced).
#pragma once
#include <cstddef>
#include <vector>
namespace boost::container {
template <std::size_t A>
struct inplace_alignment {};
template <bool B>
struct throw_on_overflow {};
template <typename... Options>
struct static_vector_options {
    using type = static_vector_options;
};

template <typename T, std::size_t N, typename Options = void>
class static_vector : public std::vector<T> {
public:
    using std::vector<T>::vector;
    static_vector() { this->reserve(N); }
};
} // namespace boost::container
