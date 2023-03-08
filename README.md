# Seckoo

A concurrent Cuckoo Hash Table.

## Current State

Currently Seckoo's `CuckooHash` is limited to a Single-Writer-Multi-Reader setup where keys and values both must implement `Copy` along with the usual `Hash` and `Eq` trait for keys. This is likely to change in the future, so that any key that implements `Hash + Eq` may be used.

Seckoo is also a fixed-capacity hash table, but this could also change in future iterations.

This crate and its API is thus still quite unstable and should be used with **caution**!

## TODO

- [ ] Vital
  - [ ] `insert` currently allows for multiple equal keys to exist in the table at the same time. Fix this!
    - Does it actually? If we search for a key, an empty bucket should mean that it is not in the list, as a keys should always try to insert themselves as early as possible. Once we allow for removing elements, then this will be a concern!
 - [ ] Changing bucket group items to `*mut Bucket<K, V>` would reduce the minimum allocation size, as the bucket array/`Vec` would be smaller.
- [ ] Upgrade from SeqLock to a Pointer setup.
  - [ ] Change `Bucket<K, V>` to `AtomicPointer<Bucket<K, V>>`
  - [ ] Add garbage collection ([`crossbeam_epoch::Epoch`](https://crates.io/crates/crossbeam-epoch)).
  - [ ] Add `Entry<'c, K, V, ...>` type to give concurrent access to keys.
  - [ ] Remove `WriteGuard<'c, K, V, ...>` as it should not be needed anymore
- [ ] Add additional methods (in _pseudo Rust_):
  - [ ] `len(&self) -> usize`
  - [ ] `get_or_insert(&self, key: K, f: impl FnOnce() -> V) -> Entry<'c, K, V, ...>;`
  - [ ] `get_cloned(&self, key: &K) -> Option<V> where V: Clone;`
  - [ ] `contains(&self, key: &K) -> bool;`
- [ ] Improve code coverage
- [ ] Add fuzzing
- [ ] Add benches
