// boost::intrusive::list (the subset suyu's shader_recompiler uses): a circular doubly linked list threaded through
// a hook the element type derives from. Elements are never owned or copied by the list.
#pragma once
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace boost::intrusive {
struct normal_link {};
struct safe_link {};
template <typename Mode>
struct link_mode {};
template <typename Hook>
struct base_hook {};
template <bool B>
struct constant_time_size {};

namespace detail {
struct list_node {
    list_node* prev_ = nullptr;
    list_node* next_ = nullptr;
};
} // namespace detail

template <typename... Options>
struct list_base_hook : detail::list_node {};

template <typename T, typename... Options>
class list {
    using node = detail::list_node;
    node root;
    std::size_t count = 0;

    static T& value_of(node* n) { return static_cast<T&>(*n); }
    static node* node_of(T& v) { return static_cast<node*>(&v); }
    static node* node_of(const T& v) { return const_cast<node*>(static_cast<const node*>(&v)); }
    void reset() {
        root.prev_ = root.next_ = &root;
        count = 0;
    }
    static void link_before(node* pos, node* n) {
        n->prev_ = pos->prev_;
        n->next_ = pos;
        pos->prev_->next_ = n;
        pos->prev_ = n;
    }
    static void unlink(node* n) {
        n->prev_->next_ = n->next_;
        n->next_->prev_ = n->prev_;
        n->prev_ = n->next_ = nullptr;
    }

public:
    template <bool Const>
    class iter {
        friend class list;
        node* n = nullptr;

    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = std::conditional_t<Const, const T*, T*>;
        using reference = std::conditional_t<Const, const T&, T&>;
        iter() = default;
        explicit iter(node* n_) : n(n_) {}
        // boost's iterators convert from an element pointer (structured_control_flow assigns `node = stmt->up`)
        iter(std::conditional_t<Const, const T*, T*> p) : n(p ? node_of(*p) : nullptr) {}
        template <bool C2, typename = std::enable_if_t<Const && !C2>>
        iter(const iter<C2>& o) : n(o.raw()) {}
        reference operator*() const { return value_of(n); }
        pointer operator->() const { return &value_of(n); }
        iter& operator++() {
            n = n->next_;
            return *this;
        }
        iter operator++(int) {
            iter t = *this;
            n = n->next_;
            return t;
        }
        iter& operator--() {
            n = n->prev_;
            return *this;
        }
        iter operator--(int) {
            iter t = *this;
            n = n->prev_;
            return t;
        }
        friend bool operator==(const iter& a, const iter& b) { return a.n == b.n; }
        node* raw() const { return n; }
    };
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator = iter<false>;
    using const_iterator = iter<true>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    list() { reset(); }
    list(const list&) = delete;
    list& operator=(const list&) = delete;
    list(list&& o) noexcept {
        reset();
        take(o);
    }
    list& operator=(list&& o) noexcept {
        if (this != &o) {
            clear();
            take(o);
        }
        return *this;
    }
    // Elements are not touched: they may already be gone (suyu frees them through ObjectPool), as boost allows.
    ~list() = default;

    iterator begin() { return iterator(root.next_); }
    iterator end() { return iterator(&root); }
    const_iterator begin() const { return const_iterator(root.next_); }
    const_iterator end() const { return const_iterator(const_cast<node*>(&root)); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const_reverse_iterator crbegin() const { return rbegin(); }
    const_reverse_iterator crend() const { return rend(); }

    bool empty() const { return root.next_ == &root; }
    size_type size() const { return count; }
    T& front() { return value_of(root.next_); }
    T& back() { return value_of(root.prev_); }
    const T& front() const { return value_of(root.next_); }
    const T& back() const { return value_of(root.prev_); }

    void push_back(T& v) {
        link_before(&root, node_of(v));
        ++count;
    }
    void push_front(T& v) {
        link_before(root.next_, node_of(v));
        ++count;
    }
    void pop_front() {
        unlink(root.next_);
        --count;
    }
    void pop_back() {
        unlink(root.prev_);
        --count;
    }
    iterator insert(const_iterator pos, T& v) {
        link_before(pos.raw(), node_of(v));
        ++count;
        return iterator(node_of(v));
    }
    iterator erase(const_iterator pos) {
        node* next = pos.raw()->next_;
        unlink(pos.raw());
        --count;
        return iterator(next);
    }
    iterator erase(const_iterator first, const_iterator last) {
        while (first != last) first = erase(first);
        return iterator(last.raw());
    }
    void clear() { reset(); }   // elements keep stale links, as boost's normal_link mode leaves them
    // Moves [first, last) of x in front of pos.
    void splice(const_iterator pos, list& x, const_iterator first, const_iterator last) {
        if (first == last) return;
        std::size_t moved = 0;
        for (const_iterator i = first; i != last; ++i) ++moved;
        node* f = first.raw();
        node* l = last.raw()->prev_;
        f->prev_->next_ = last.raw();
        last.raw()->prev_ = f->prev_;
        node* p = pos.raw();
        f->prev_ = p->prev_;
        l->next_ = p;
        p->prev_->next_ = f;
        p->prev_ = l;
        x.count -= moved;
        count += moved;
    }
    void splice(const_iterator pos, list& x) { splice(pos, x, x.begin(), x.end()); }

    static iterator s_iterator_to(T& v) { return iterator(node_of(v)); }
    static const_iterator s_iterator_to(const T& v) { return const_iterator(node_of(v)); }
    iterator iterator_to(T& v) { return iterator(node_of(v)); }
    const_iterator iterator_to(const T& v) const { return const_iterator(node_of(v)); }

private:
    void take(list& o) {
        if (o.empty()) return;
        root.next_ = o.root.next_;
        root.prev_ = o.root.prev_;
        root.next_->prev_ = &root;
        root.prev_->next_ = &root;
        count = o.count;
        o.reset();
    }
};
} // namespace boost::intrusive
