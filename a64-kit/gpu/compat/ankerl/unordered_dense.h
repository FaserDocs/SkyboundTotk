// ankerl::unordered_dense::map for suyu's shader_recompiler: std::unordered_map has the API it uses.
#pragma once
#include <unordered_map>
#include <unordered_set>
namespace ankerl::unordered_dense {
template <class K, class V, class H = std::hash<K>, class E = std::equal_to<K>>
using map = std::unordered_map<K, V, H, E>;
template <class K, class H = std::hash<K>, class E = std::equal_to<K>>
using set = std::unordered_set<K, H, E>;
}
