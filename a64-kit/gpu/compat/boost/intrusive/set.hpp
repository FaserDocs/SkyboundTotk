// boost::intrusive::set (the subset suyu's shader_recompiler uses): an ordered set of element references, ordered by
// the element's operator<. Built on std::multiset of pointers, whose iterators survive inserts as boost's do.
// Lookups with a separate key and comparator are linear: a shader's control-flow graph has few blocks.
#pragma once
#include <boost/intrusive/list.hpp>
#include <cstddef>
#include <iterator>
#include <set>
#include <type_traits>
#include <utility>

namespace boost::intrusive {
template <typename... Options>
struct set_base_hook {};

template <typename T, typename... Options>
class set {
    struct less {
        bool operator()(const T* a, const T* b) const { return *a < *b; }
    };
    using base = std::multiset<T*, less>;
    base s;

public:
    template <typename It, typename Ref>
    class iter {
        friend class set;
        It it{};

    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = std::remove_reference_t<Ref>*;
        using reference = Ref;
        iter() = default;
        explicit iter(It i) : it(i) {}
        reference operator*() const { return **it; }
        pointer operator->() const { return *it; }
        iter& operator++() {
            ++it;
            return *this;
        }
        iter operator++(int) {
            iter t = *this;
            ++it;
            return t;
        }
        iter& operator--() {
            --it;
            return *this;
        }
        iter operator--(int) {
            iter t = *this;
            --it;
            return t;
        }
        friend bool operator==(const iter& a, const iter& b) { return a.it == b.it; }
    };
    using iterator = iter<typename base::iterator, T&>;
    using const_iterator = iter<typename base::const_iterator, const T&>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type = std::size_t;
    using value_type = T;

    std::pair<iterator, bool> insert(T& v) { return {iterator(s.insert(&v)), true}; }
    iterator insert(iterator hint, T& v) { return iterator(s.insert(hint.it, &v)); }
    iterator erase(iterator it) { return iterator(s.erase(it.it)); }
    template <typename Key, typename Compare>
    iterator find(const Key& key, Compare comp) {
        for (auto i = s.begin(); i != s.end(); ++i)
            if (!comp(**i, key) && !comp(key, **i)) return iterator(i);
        return end();
    }
    template <typename Key, typename Compare>
    iterator upper_bound(const Key& key, Compare comp) {
        for (auto i = s.begin(); i != s.end(); ++i)
            if (comp(key, **i)) return iterator(i);
        return end();
    }
    template <typename Key, typename Compare>
    iterator lower_bound(const Key& key, Compare comp) {
        for (auto i = s.begin(); i != s.end(); ++i)
            if (!comp(**i, key)) return iterator(i);
        return end();
    }
    iterator begin() { return iterator(s.begin()); }
    iterator end() { return iterator(s.end()); }
    const_iterator begin() const { return const_iterator(s.begin()); }
    const_iterator end() const { return const_iterator(s.end()); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    bool empty() const { return s.empty(); }
    size_type size() const { return s.size(); }
    void clear() { s.clear(); }
};
} // namespace boost::intrusive
