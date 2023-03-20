#![deny(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]
//! This is a simple lock free cache, implemented via a hash table following a cuckoo hashing
//! scheme.
use crossbeam_epoch::{Atomic, Collector, CompareExchangeError, Guard, Owned, Shared};
use std::alloc::Layout;
use std::collections::hash_map;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::{align_of, size_of};
use std::ops::Deref;
use std::sync::atomic;

const DEFAULT_LENGTH: usize = 4096;

#[inline]
fn buckets_layout<K, V, const N: usize>() -> Layout {
    let size = size_of::<[Atomic<Bucket<K, V>>; N]>();
    let align = align_of::<[Atomic<Bucket<K, V>>; N]>();
    std::alloc::Layout::from_size_align(size, align).expect("Amount of buckets to not exceed isize")
}

/// A lock free cache. Cached values must implement `Clone` in order to be retrieved.
pub type Cache<K, V, H = hash_map::RandomState, const N: usize = DEFAULT_LENGTH> =
    CuckooHash<K, V, H, N>;

/// A lock free hash table following a cuckoo hashing scheme.
pub struct CuckooHash<K, V, H, const N: usize> {
    hash_builders: [H; 2],
    buckets: BucketArray<K, V, N>,
    collector: Collector,
}

impl<K: Debug, V: Debug, H, const N: usize> Debug for CuckooHash<K, V, H, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.collector.register().pin();

        let mut list = f.debug_list();
        for bucket in &*self.buckets {
            let ptr = bucket.load(atomic::Ordering::Relaxed, &guard);

            if ptr.is_null() {
                continue;
            }

            // Safety: While we hold the `Shared` we can safely deref the pointer.
            #[allow(unsafe_code)]
            unsafe {
                list.entry(ptr.deref());
            }
        }
        list.finish()
    }
}

#[allow(unsafe_code)]
unsafe impl<K, V, H, const N: usize> Send for CuckooHash<K, V, H, N>
where
    K: Send,
    V: Send,
{
}

#[allow(unsafe_code)]
unsafe impl<K, V, H, const N: usize> Sync for CuckooHash<K, V, H, N>
where
    K: Sync,
    V: Sync,
{
}

#[derive(Debug)]
#[repr(align(2))]
struct Bucket<K, V> {
    key: K,
    val: V,
}

#[derive(Debug)]
struct BucketArray<K, V, const N: usize> {
    buckets: *mut [Atomic<Bucket<K, V>>; N],
}

impl<K, V, const N: usize> BucketArray<K, V, N> {
    fn new() -> Self {
        let layout = buckets_layout::<K, V, N>();

        #[allow(unsafe_code)]
        let buckets = unsafe { std::alloc::alloc_zeroed(layout) };

        Self {
            buckets: buckets.cast(),
        }
    }
}

impl<K, V, const N: usize> Deref for BucketArray<K, V, N> {
    type Target = [Atomic<Bucket<K, V>>; N];

    fn deref(&self) -> &Self::Target {
        // Safety: No one besides us will deallocate the array or alter it in an undefined way.
        #[allow(unsafe_code)]
        unsafe {
            &mut (*self.buckets)
        }
    }
}

impl<K, V, const N: usize> Drop for BucketArray<K, V, N> {
    fn drop(&mut self) {
        let layout = buckets_layout::<K, V, N>();
        // Safety:
        // 1. We know no one else will have deallocated the buckets before we drop.
        // 2. After `BucketArray` is dropped the array will no longer be accessed.
        #[allow(unsafe_code)]
        unsafe {
            std::alloc::dealloc(self.buckets.cast(), layout);
        }
    }
}

impl<K, V, const N: usize> CuckooHash<K, V, hash_map::RandomState, N> {
    /// Create a new `CuckooHash` with the standards library's `RandomState`.
    pub fn std() -> Self {
        // Ensure `Bucket`s leave enough space to store tag.
        assert!(align_of::<Bucket<K, V>>() >= 2);

        CuckooHash {
            hash_builders: [
                std::collections::hash_map::RandomState::new(),
                std::collections::hash_map::RandomState::new(),
            ],
            buckets: BucketArray::new(),
            collector: Collector::new(),
        }
    }
}

impl<K, V, H, const N: usize> CuckooHash<K, V, H, N>
where
    H: BuildHasher + Clone,
    K: Hash + Eq,
    V: Clone,
    <H as BuildHasher>::Hasher: Hasher,
{
    /// Get a clone of the value corresponding to a key, if it exists.
    pub fn get(&self, k: &K) -> Option<V> {
        let mut h1 = self.hash_builders[0].build_hasher();
        let mut h2 = self.hash_builders[1].build_hasher();

        k.hash(&mut h1);
        k.hash(&mut h2);
        let first_index = h1.finish() as usize;
        let hash2 = h2.finish() as usize;

        // XOR the first index with the tag to get the second index and truncate.
        let second_index = first_index ^ hash2;

        let handle = self.collector.register();
        let guard = handle.pin();
        guard.flush();

        for i in [first_index, second_index] {
            let ptr = &self.buckets[i % N].load(atomic::Ordering::Relaxed, &guard);

            if ptr.is_null() {
                break;
            }

            #[allow(unsafe_code)]
            unsafe {
                if ptr.deref().key.eq(k) {
                    return Some((&ptr.deref().val).clone());
                }
            }
        }

        None
    }

    /// Insert a new key-value pair into the hash table.
    #[inline]
    pub fn insert(&self, k: K, v: V) {
        let mut h1 = self.hash_builders[0].build_hasher();
        let mut h2 = self.hash_builders[1].build_hasher();

        k.hash(&mut h1);
        k.hash(&mut h2);
        let index = h1.finish() as usize;

        let handle = self.collector.register();
        let guard = handle.pin();
        let new_bucket = Owned::from(Bucket { key: k, val: v })
            .with_tag(1)
            .into_shared(&guard);

        self.insert_internal(index, new_bucket, &guard, 0);
    }

    fn insert_internal(
        &self,
        index: usize,
        new_bucket: Shared<Bucket<K, V>>,
        guard: &Guard,
        depth: usize,
    ) {
        // If retry depth is exceeded, defer deallocation of the current bucket and return.
        if depth >= 8 {
            // Safety: This bucket is no longer inserted anywhere in the table, so we can safely
            // defer deallocation.
            #[allow(unsafe_code)]
            unsafe {
                guard.defer_destroy(new_bucket);
                guard.flush();
            }
            return;
        }

        // Search all the buckets at this index for the tag and then value.
        let bucket = &self.buckets[index % N];

        // Try inserting expecting a null.
        let Err( CompareExchangeError { mut current, new: _ } ) = bucket.compare_exchange(
            Shared::null(),
            new_bucket,
            atomic::Ordering::Relaxed,
            atomic::Ordering::Relaxed,
            &guard,
        ) else {
            return;
        };

        loop {
            // Keys are not equal.
            // Safety: We have a valid `Shared` to the bucket, thus we can safely deref.
            #[allow(unsafe_code)]
            unsafe {
                if current.deref().key.ne(&new_bucket.deref().key) {
                    break;
                }
            }
            // Same key, but the other bucket is has a tag of one while we have a tag of 0.
            if current.tag() & !new_bucket.tag() == 1 {
                // Safety: We have not inserted `new_bucket` anywhere, so we can safely defer its
                // deallocation.
                #[allow(unsafe_code)]
                unsafe {
                    guard.defer_destroy(new_bucket);
                    guard.flush();
                }
                return;
            }

            // Try swapping out equal key.
            current = match bucket.compare_exchange(
                current,
                new_bucket,
                atomic::Ordering::Relaxed,
                atomic::Ordering::Relaxed,
                &guard,
            ) {
                Ok(_) => {
                    // Succeeded to swap with an equal key.
                    // We defer deallocation.
                    // Safety: We removed the old bucket from the array and can thus safely defer
                    // its deallocation.
                    #[allow(unsafe_code)]
                    unsafe {
                        guard.defer_destroy(current);
                        guard.flush();
                    }
                    return;
                }
                Err(CompareExchangeError { current, new: _ }) => current,
            }
        }

        // No empty location was found, so we insert depending on recursion depth into this bucket and re-insert
        // the replaced bucket.
        let mut h2 = self.hash_builders[1].build_hasher();

        let old = bucket.swap(new_bucket, atomic::Ordering::Relaxed, &guard);

        // Set the tag to 0 to signal that it has been moved.
        let old = old.with_tag(0);

        // XOR the index with the next_tag to get the other index for this key and truncate.
        // Safety: We hold a valid `Shared`.
        #[allow(unsafe_code)]
        unsafe {
            old.deref().key.hash(&mut h2);
        }
        let hash2 = h2.finish();
        let next_index = index ^ hash2 as usize;

        self.insert_internal(next_index, old, guard, depth + 1);
    }

    /// Returns a cloning iterator over the hash table.
    pub fn iter(&self) -> Iter<'_, K, V, H, N> {
        Iter {
            cuckoo: &self,
            index: 0,
        }
    }
}

/// A cloning iterator over the hash table.
#[derive(Debug)]
pub struct Iter<'c, K, V, H, const N: usize> {
    cuckoo: &'c CuckooHash<K, V, H, N>,
    index: usize,
}

impl<'a, K, V, H, const N: usize> Iterator for Iter<'a, K, V, H, N>
where
    H: BuildHasher + Clone,
    K: Hash + Eq,
    V: Clone,
    <H as BuildHasher>::Hasher: Hasher,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        let handle = self.cuckoo.collector.register();
        let guard = handle.pin();

        while self.index < N
            && self.cuckoo.buckets[self.index]
                .load(atomic::Ordering::Relaxed, &guard)
                .is_null()
        {
            self.index += 1
        }

        if self.index >= N {
            return None;
        }

        // We get a `Shared` and deref it. This is safe, as we have ensured no threads will
        // deallocate a `Bucket` that is still in the hash table.
        #[allow(unsafe_code)]
        unsafe {
            self.index += 1;
            return Some(
                self.cuckoo.buckets[self.index - 1]
                    .load(atomic::Ordering::Relaxed, &guard)
                    .deref()
                    .val
                    .clone(),
            );
        }
    }
}

impl<'a, K, V, H, const N: usize> IntoIterator for &'a CuckooHash<K, V, H, N>
where
    H: BuildHasher + Clone,
    K: Hash + Eq,
    V: Clone,
    <H as BuildHasher>::Hasher: Hasher,
{
    type Item = V;
    type IntoIter = Iter<'a, K, V, H, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V, H, const N: usize> Drop for CuckooHash<K, V, H, N> {
    fn drop(&mut self) {
        for bucket in &*self.buckets {
            // Safety: We have a mutable reference to the CuckooHash, so we are the only thread
            // that can currently accessed the pointers.
            #[allow(unsafe_code)]
            unsafe {
                let node = bucket.load(atomic::Ordering::Relaxed, crossbeam_epoch::unprotected());

                let Some(node) = node.try_into_owned()  else {
                    continue;
                };

                drop(node.into_box())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::{atomic::AtomicUsize, Arc};

    use super::*;

    #[test]
    fn test_std_constructor() {
        let _ = Cache::<i32, i32>::std();
    }

    #[test]
    fn test_insert_get() {
        let hashmap = Cache::<String, ()>::std();

        hashmap.insert("Hello There!".to_string(), ());

        assert!(hashmap.get(&"Hello There!".to_string()).is_some());
    }

    #[test]
    fn test_insert_updates_equal_key() {
        let hashmap = Cache::<String, i32>::std();

        hashmap.insert("Hello There!".to_string(), 0);
        hashmap.insert("Hello There!".to_string(), 1);

        assert!(hashmap.get(&"Hello There!".to_string()).is_some());
        println!("returning 1");
        assert_eq!(1, hashmap.get(&"Hello There!".to_string()).unwrap());
        println!("returning 2");
    }

    #[test]
    fn test_insert_will_remove_oldest() {
        'test: loop {
            let hashmap = Cache::<i32, i32>::std();

            for i in 0..1024 {
                hashmap.insert(i, 0);
            }

            for i in 0..1024 {
                if !hashmap.get(&i).is_some() {
                    continue 'test;
                }
            }

            for i in 0..1024 {
                hashmap.insert(i, 1);
            }

            for i in 0..1024 {
                assert_eq!(hashmap.get(&i).unwrap(), 1);
            }

            break;
        }
    }

    #[test]
    fn stops_retrying_on_max_depth() {
        let hashmap = Cache::<String, i32>::std();

        hashmap.insert("1".to_string(), 0);
        hashmap.insert("2".to_string(), 1);
        hashmap.insert("3".to_string(), 2);
        hashmap.insert("4".to_string(), 3);

        println!("{}", hashmap.get(&"1".to_string()).is_some());
        println!("{}", hashmap.get(&"2".to_string()).is_some());
        println!("{}", hashmap.get(&"3".to_string()).is_some());
        println!("{}", hashmap.get(&"4".to_string()).is_some());
    }

    #[test]
    fn test_iter() {
        let hashmap = Cache::<String, i32>::std();

        hashmap.insert("1".to_string(), 0);
        hashmap.insert("2".to_string(), 1);
        hashmap.insert("3".to_string(), 2);
        hashmap.insert("4".to_string(), 3);

        println!("iter test");

        for value in &hashmap {
            println!("{value}")
        }
    }

    #[test]
    fn test_threads() {
        use std::thread;

        let hashmap = Cache::<[u8; 12], [u8; 12]>::std();

        thread::scope(|s| {
            let hashmap = &hashmap;

            for _ in 0..8 {
                s.spawn(move || {
                    for _ in 0..16 {
                        hashmap.insert(rand::random(), rand::random());
                    }
                });
            }
        });
    }

    #[test]
    fn test_drop() {
        struct CountOnDrop {
            key: i32,
            counter: Arc<AtomicUsize>,
        }

        impl CountOnDrop {
            fn new(key: i32, counter: Arc<AtomicUsize>) -> Self {
                CountOnDrop { key, counter }
            }
        }

        impl Hash for CountOnDrop {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.key.hash(state);
            }
        }

        impl PartialEq for CountOnDrop {
            fn eq(&self, other: &Self) -> bool {
                self.key.eq(&other.key)
            }
        }

        impl Eq for CountOnDrop {}

        impl Drop for CountOnDrop {
            fn drop(&mut self) {
                println!("Dropping: {}", self.key);
                self.counter.fetch_add(1, atomic::Ordering::Release);
            }
        }

        impl Clone for CountOnDrop {
            fn clone(&self) -> Self {
                CountOnDrop {
                    key: self.key.clone(),
                    counter: self.counter.clone(),
                }
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        let hashmap = Cache::<i32, CountOnDrop, _, 4>::std();

        hashmap.insert(0, CountOnDrop::new(0, counter.clone()));
        hashmap.insert(0, CountOnDrop::new(1, counter.clone()));
        assert_eq!(hashmap.get(&0).unwrap().key, 1);

        drop(hashmap);
        assert_eq!(counter.load(atomic::Ordering::Relaxed), 3);
    }
}
