// boost::container::small_vector for suyu's shader_recompiler: a std::vector with the inline capacity reserved.
#pragma once
#include <cstddef>
#include <initializer_list>
#include <vector>
namespace boost::container {
template <typename T, std::size_t N, typename Alloc = void>
class small_vector : public std::vector<T> {
public:
    using std::vector<T>::vector;
    small_vector() { this->reserve(N); }
    small_vector(std::initializer_list<T> list) : std::vector<T>(list) {}
};
} // namespace boost::container
