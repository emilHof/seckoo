use crossbeam_epoch::{Atomic, Collector, CompareExchangeError, Guard, Owned, Shared};
use std::collections::hash_map;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::{align_of, size_of};
use std::sync::atomic;

const DEFAULT_LENGTH: usize = 4096;
const DEFAULT_BUCKET_CT: usize = 1;

#[inline]
fn low_bits<T>() -> usize {
    (1 << align_of::<T>().trailing_zeros()) - 1
}

pub type HashMap<
    K,
    V,
    H = hash_map::RandomState,
    const M: usize = DEFAULT_LENGTH,
    const B: usize = DEFAULT_BUCKET_CT,
> = CuckooHash<K, V, H, M, B>;

#[derive(Debug)]
pub struct CuckooHash<K, V, H, const M: usize, const B: usize> {
    hash_builders: [H; 2],
    buckets: [[Atomic<Bucket<K, V>>; B]; M],
    collector: Collector,
}

unsafe impl<K, V, H, const M: usize, const B: usize> Send for CuckooHash<K, V, H, M, B>
where
    K: Send,
    V: Send,
{
}

unsafe impl<K, V, H, const M: usize, const B: usize> Sync for CuckooHash<K, V, H, M, B>
where
    K: Sync,
    V: Sync,
{
}

#[derive(Debug)]
#[repr(align(8))]
struct Bucket<K, V> {
    key: K,
    val: V,
}

impl<K, V, const M: usize, const B: usize> CuckooHash<K, V, hash_map::RandomState, M, B> {
    pub fn std() -> Self {
        // Ensure `Bucket`s leave enough space to store tag.
        assert!(align_of::<Bucket<K, V>>() >= 8);

        let buckets = [(); M].map(|_| [(); B].map(|_| Atomic::<Bucket<K, V>>::null()));

        CuckooHash {
            hash_builders: [
                std::collections::hash_map::RandomState::new(),
                std::collections::hash_map::RandomState::new(),
            ],
            buckets,
            collector: Collector::new(),
        }
    }
}

impl<K, V, H, const M: usize, const B: usize> CuckooHash<K, V, H, M, B>
where
    H: BuildHasher + Clone,
    K: Hash + Eq,
    V: Clone,
    <H as BuildHasher>::Hasher: Hasher,
{
    pub fn get(&self, k: &K) -> Option<V> {
        let mut h1 = self.hash_builders[0].build_hasher();
        let mut h2 = self.hash_builders[1].build_hasher();

        k.hash(&mut h1);
        k.hash(&mut h2);
        let first_index = h1.finish() as usize;
        let tag = h2.finish() as usize & low_bits::<Bucket<K, V>>();

        // XOR the first index with the tag to get the second index and truncate.
        tag.hash(&mut h1);
        let second_index = first_index ^ h1.finish() as usize;

        let handle = self.collector.register();
        let guard = handle.pin();
        guard.flush();

        for i in [first_index, second_index] {
            for b in &self.buckets[i % M] {
                let ptr = b.load(atomic::Ordering::Relaxed, &guard);
                if ptr.is_null() {
                    break;
                }

                unsafe {
                    if ptr.tag() & low_bits::<Bucket<K, V>>() == tag && ptr.deref().key.eq(k) {
                        return Some(ptr.deref().val.clone());
                    }
                }
            }
        }

        None
    }

    pub fn insert(&self, k: K, v: V) {
        let mut h1 = self.hash_builders[0].build_hasher();
        let mut h2 = self.hash_builders[1].build_hasher();

        k.hash(&mut h1);
        k.hash(&mut h2);
        let index = h1.finish() as usize;
        let tag = h2.finish() as usize & low_bits::<Bucket<K, V>>();

        let handle = self.collector.register();
        let guard = handle.pin();
        let new_bucket = Owned::from(Bucket { key: k, val: v })
            .with_tag(tag)
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
            unsafe {
                guard.defer_destroy(new_bucket);
            }
            return;
        }

        // Search all the buckets at this index for the tag and then value.
        for b in &self.buckets[index % M] {
            loop {
                // Try inserting expecting a null.
                let Err( CompareExchangeError { current, new: _ } ) = b.compare_exchange(
                        Shared::null(),
                        new_bucket,
                        atomic::Ordering::Relaxed,
                        atomic::Ordering::Relaxed,
                        &guard,
                    ) else {
                        return;
                    };

                unsafe {
                    if current.tag() != new_bucket.tag()
                        || current.deref().key.ne(&new_bucket.deref().key)
                    {
                        break;
                    }
                }

                // If keys are equal, try inserting a new value.
                let Err( CompareExchangeError { current, new: _ } ) = b.compare_exchange(
                        current,
                        new_bucket,
                        atomic::Ordering::Relaxed,
                        atomic::Ordering::Relaxed,
                        &guard,
                    ) else {
                        return;
                    };

                unsafe {
                    if current.tag() != new_bucket.tag()
                        || current.deref().key.ne(&new_bucket.deref().key)
                    {
                        break;
                    }
                }
            }
        }

        // No empty location was found, so we insert depending on recursion depth into this bucket and re-insert
        // the replaced bucket.
        let mut h1 = self.hash_builders[0].build_hasher();
        let bucket = &self.buckets[index % M][B - (depth % B) - 1];

        let old = bucket.swap(new_bucket, atomic::Ordering::Relaxed, &guard);
        let next_tag = old.tag() & low_bits::<Bucket<K, V>>();

        // XOR the index with the next_tag to get the second index and truncate.
        next_tag.hash(&mut h1);
        let next_index = index ^ h1.finish() as usize;

        self.insert_internal(next_index, old, guard, depth + 1);
    }

    pub fn iter(&self) -> Iter<'_, K, V, H, M, B> {
        Iter {
            cuckoo: &self,
            index: 0,
        }
    }
}

pub struct Iter<'c, K, V, H, const M: usize, const B: usize> {
    cuckoo: &'c CuckooHash<K, V, H, M, B>,
    index: usize,
}

impl<'a, K, V, H, const M: usize, const B: usize> Iterator for Iter<'a, K, V, H, M, B>
where
    H: BuildHasher + Clone,
    K: Hash + std::cmp::Eq,
    V: Clone,
    <H as BuildHasher>::Hasher: Hasher,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        let handle = self.cuckoo.collector.register();
        let guard = handle.pin();

        while self.index < M * B
            && self.cuckoo.buckets[self.index / B][self.index % B]
                .load(atomic::Ordering::Relaxed, &guard)
                .is_null()
        {
            self.index += 1
        }

        if self.index >= M * B {
            return None;
        }

        unsafe {
            self.index += 1;
            return Some(
                self.cuckoo.buckets[(self.index - 1) / B][(self.index - 1) % B]
                    .load(atomic::Ordering::Relaxed, &guard)
                    .deref()
                    .val
                    .clone(),
            );
        }
    }
}

impl<'a, K, V, H, const M: usize, const B: usize> IntoIterator for &'a CuckooHash<K, V, H, M, B>
where
    H: BuildHasher + Clone,
    K: Hash + std::cmp::Eq,
    V: Clone,
    <H as BuildHasher>::Hasher: Hasher,
{
    type Item = V;
    type IntoIter = Iter<'a, K, V, H, M, B>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_std_constructor() {
        let _ = HashMap::<i32, i32>::std();
    }

    #[test]
    fn test_insert_get() {
        let hashmap = HashMap::<String, ()>::std();

        hashmap.insert("Hello There!".to_string(), ());

        assert!(hashmap.get(&"Hello There!".to_string()).is_some());
    }

    #[test]
    fn test_insert_updates_equal_key() {
        let hashmap = HashMap::<String, i32>::std();

        hashmap.insert("Hello There!".to_string(), 0);
        hashmap.insert("Hello There!".to_string(), 1);

        assert!(hashmap.get(&"Hello There!".to_string()).is_some());
        assert_eq!(1, hashmap.get(&"Hello There!".to_string()).unwrap());
    }

    #[test]
    fn stops_retrying_on_max_depth() {
        let hashmap = HashMap::<String, i32, _, 4, 1>::std();

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
        let hashmap = HashMap::<String, i32>::std();

        hashmap.insert("1".to_string(), 0);
        hashmap.insert("2".to_string(), 1);
        hashmap.insert("3".to_string(), 2);
        hashmap.insert("4".to_string(), 3);

        for value in &hashmap {
            println!("{value}")
        }
    }

    #[test]
    fn test_threads() {
        use std::thread;

        let hashmap = HashMap::<[u8; 12], [u8; 12], _, 64, 2>::std();

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

        /*
        for value in &hashmap {
            println!("{}", String::from_utf8_lossy(&value));
        }
        */
    }
}
