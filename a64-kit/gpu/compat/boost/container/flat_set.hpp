// boost::container::flat_set for suyu's shader_recompiler: a sorted vector.
#pragma once
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
namespace boost::container {
template <typename T, typename Compare = std::less<T>, typename Container = std::vector<T>>
class flat_set {
    Container v;
    Compare cmp;

public:
    using value_type = T;
    using iterator = typename Container::iterator;
    using const_iterator = typename Container::const_iterator;
    std::pair<iterator, bool> insert(const T& x) {
        auto it = std::lower_bound(v.begin(), v.end(), x, cmp);
        if (it != v.end() && !cmp(x, *it)) return {it, false};
        return {v.insert(it, x), true};
    }
    template <typename... A>
    std::pair<iterator, bool> emplace(A&&... a) { return insert(T(std::forward<A>(a)...)); }
    iterator find(const T& x) {
        auto it = std::lower_bound(v.begin(), v.end(), x, cmp);
        return it != v.end() && !cmp(x, *it) ? it : v.end();
    }
    const_iterator find(const T& x) const {
        auto it = std::lower_bound(v.begin(), v.end(), x, cmp);
        return it != v.end() && !cmp(x, *it) ? it : v.end();
    }
    bool contains(const T& x) const { return find(x) != v.end(); }
    std::size_t index_of(const_iterator it) const { return static_cast<std::size_t>(it - v.begin()); }
    std::size_t index_of(iterator it) { return static_cast<std::size_t>(it - v.begin()); }
    const T& operator[](std::size_t i) const { return v[i]; }
    T& operator[](std::size_t i) { return v[i]; }
    std::size_t count(const T& x) const { return contains(x) ? 1 : 0; }
    iterator erase(const_iterator it) { return v.erase(it); }
    std::size_t erase(const T& x) {
        auto it = find(x);
        if (it == v.end()) return 0;
        v.erase(it);
        return 1;
    }
    iterator begin() { return v.begin(); }
    iterator end() { return v.end(); }
    const_iterator begin() const { return v.begin(); }
    const_iterator end() const { return v.end(); }
    std::size_t size() const { return v.size(); }
    bool empty() const { return v.empty(); }
    void clear() { v.clear(); }
    void reserve(std::size_t n) { v.reserve(n); }
};
} // namespace boost::container
