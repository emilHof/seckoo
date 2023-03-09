use crossbeam_epoch::{Atomic, Collector, CompareExchangeError, Guard, LocalHandle, Owned, Shared};
use std::hash::BuildHasher;
use std::sync::atomic;

const DEFAULT_LENGTH: usize = 1024;
const DEFAULT_HASHER_CT: usize = 2;
const DEFAULT_BUCKET_CT: usize = 4;

pub type HashMap<
    K,
    V,
    H = std::collections::hash_map::DefaultHasher,
    const M: usize = DEFAULT_LENGTH,
    const N: usize = DEFAULT_HASHER_CT,
    const B: usize = DEFAULT_BUCKET_CT,
> = CuckooHash<K, V, H, M, N, B>;

#[derive(Debug)]
pub struct CuckooHash<K, V, H, const M: usize, const N: usize, const B: usize> {
    hashers: [H; N],
    buckets: [[Atomic<Bucket<K, V>>; B]; M],
    collector: Collector,
}

unsafe impl<K, V, H, const M: usize, const N: usize, const B: usize> Send
    for CuckooHash<K, V, H, M, N, B>
where
    K: Send,
    V: Send,
{
}

unsafe impl<K, V, H, const M: usize, const N: usize, const B: usize> Sync
    for CuckooHash<K, V, H, M, N, B>
where
    K: Sync,
    V: Sync,
{
}
#[derive(Debug)]
struct Bucket<K, V> {
    key: K,
    val: V,
}

impl<K, V, const M: usize, const N: usize, const B: usize>
    CuckooHash<K, V, std::collections::hash_map::DefaultHasher, M, N, B>
{
    // #[cfg(feature = "nightly")]
    pub fn std() -> Self {
        let buckets = [(); M].map(|_| [(); B].map(|_| Atomic::<Bucket<K, V>>::null()));

        CuckooHash {
            hashers: [(); N].map(|_| std::collections::hash_map::RandomState::new().build_hasher()),
            buckets,
            collector: Collector::new(),
        }
    }
}

impl<K, V, H, const M: usize, const N: usize, const B: usize> CuckooHash<K, V, H, M, N, B>
where
    H: std::hash::Hasher + Clone,
    K: std::hash::Hash + std::cmp::Eq,
    V: Clone,
{
    pub fn get(&self, k: &K) -> Option<V> {
        let handle = self.collector.register();

        for hasher in &self.hashers {
            let mut hasher = hasher.clone();
            k.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            if let Some(val) = self.search_buckets(hash % M, k, &handle) {
                return Some(val);
            }
        }
        None
    }

    // Searches a bucket for a key. Rereads when version after read is wrong, yet does not re-read
    // prior buckets.
    fn search_buckets<'a>(&'a self, i: usize, k: &K, handle: &LocalHandle) -> Option<V>
    where
        V: Clone,
    {
        for bucket_ptr in &self.buckets[i] {
            let guard = handle.pin();
            guard.flush();

            let bucket = bucket_ptr.load(atomic::Ordering::Relaxed, &guard);
            unsafe {
                // If we have an empty bucket, then the key is not in the map.
                if bucket.is_null() {
                    return None;
                }

                if bucket.deref().key.eq(&k) {
                    return Some(bucket.deref().val.clone());
                }
            }
        }

        None
    }

    pub fn insert(&self, k: K, v: V) {
        let handle = self.collector.register();
        let guard = handle.pin();
        let new_bucket = Owned::from(Bucket { key: k, val: v }).into_shared(&guard);

        self.insert_internal(new_bucket, &guard, 0);
    }

    fn insert_internal(&self, new_bucket: Shared<Bucket<K, V>>, guard: &Guard, depth: usize) {
        // If retry depth is exceeded, defer deallocation of the current bucket and return.
        if depth >= 8 {
            unsafe {
                guard.defer_destroy(new_bucket);
            }
            return;
        }

        for hasher in &self.hashers {
            let mut hasher = hasher.clone();
            unsafe {
                new_bucket.deref().key.hash(&mut hasher);
            }
            let hash = hasher.finish() as usize;
            for bucket in self.buckets[hash % M].iter() {
                // If an empty bucket is found, we insert here.
                loop {
                    // Try inserting expecting a null.
                    let Err( CompareExchangeError { current, new: _ } ) = bucket.compare_exchange(
                        Shared::null(),
                        new_bucket,
                        atomic::Ordering::Relaxed,
                        atomic::Ordering::Relaxed,
                        &guard,
                    ) else {
                        return;
                    };

                    unsafe {
                        if current.deref().key.ne(&new_bucket.deref().key) {
                            break;
                        }
                    }

                    // If keys are equal, try inserting a new value.
                    let Err( CompareExchangeError { current, new: _ } ) = bucket.compare_exchange(
                        current,
                        new_bucket,
                        atomic::Ordering::Relaxed,
                        atomic::Ordering::Relaxed,
                        &guard,
                    ) else {
                        return;
                    };

                    unsafe {
                        if current.deref().key.ne(&new_bucket.deref().key) {
                            break;
                        }
                    }
                }
            }
        }

        // No empty location was found, so we insert it in our first choice bucket and re-insert
        // the old value.
        let mut hasher = self.hashers[0].clone();
        unsafe {
            new_bucket.deref().key.hash(&mut hasher);
        }
        let hash = hasher.finish() as usize;
        let bucket = &self.buckets[hash % M][B - (depth % B) - 1];

        let old = bucket.swap(new_bucket, atomic::Ordering::Relaxed, &guard);

        self.insert_internal(old, guard, depth + 1);
    }

    pub fn iter(&self) -> Iter<'_, K, V, H, M, N, B> {
        Iter {
            cuckoo: &self,
            index: 0,
        }
    }
}

pub struct Iter<'c, K, V, H, const M: usize, const N: usize, const B: usize> {
    cuckoo: &'c CuckooHash<K, V, H, M, N, B>,
    index: usize,
}

impl<'a, K, V, H, const M: usize, const N: usize, const B: usize> Iterator
    for Iter<'a, K, V, H, M, N, B>
where
    H: std::hash::Hasher + Clone,
    K: std::hash::Hash + std::cmp::Eq,
    V: Clone,
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

impl<'a, K, V, H, const M: usize, const N: usize, const B: usize> IntoIterator
    for &'a CuckooHash<K, V, H, M, N, B>
where
    H: std::hash::Hasher + Clone,
    K: std::hash::Hash + std::cmp::Eq,
    V: Clone,
{
    type Item = V;
    type IntoIter = Iter<'a, K, V, H, M, N, B>;

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
        let hashmap = HashMap::<String, i32, _, 4, 2, 1>::std();

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

        let hashmap = HashMap::<[u8; 12], [u8; 12], _, 64, 2, 2>::std();

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

        for value in &hashmap {
            println!("{}", String::from_utf8_lossy(&value));
        }
    }
}
