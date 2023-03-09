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
    locked: atomic::AtomicBool,
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
            locked: atomic::AtomicBool::new(false),
        }
    }
}

impl<K, V, H: std::hash::Hasher, const M: usize, const N: usize, const B: usize>
    CuckooHash<K, V, H, M, N, B>
{
    pub fn try_lock(&self) -> Result<WriteGuard<K, V, H, M, N, B>, ()> {
        if !self.locked.swap(true, atomic::Ordering::Acquire) {
            return Ok(WriteGuard { cuckoo: self });
        };

        Err(())
    }
}

impl<K, V, H, const M: usize, const N: usize, const B: usize> CuckooHash<K, V, H, M, N, B>
where
    H: std::hash::Hasher + Clone,
    K: Clone + std::hash::Hash + std::cmp::Eq,
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
}

pub struct WriteGuard<'c, K, V, H, const M: usize, const N: usize, const B: usize> {
    cuckoo: &'c CuckooHash<K, V, H, M, N, B>,
}

impl<'c, K, V, H, const M: usize, const N: usize, const B: usize> WriteGuard<'c, K, V, H, M, N, B>
where
    H: std::hash::Hasher + Clone,
    K: std::hash::Hash + std::cmp::Eq,
    V: Clone,
{
    pub fn insert(&self, k: K, v: V) {
        let handle = self.cuckoo.collector.register();
        let guard = handle.pin();
        let new_bucket = Owned::from(Bucket { key: k, val: v }).into_shared(&guard);

        self.insert_internal(new_bucket, &guard, 0);
    }

    fn insert_internal(&self, new_bucket: Shared<Bucket<K, V>>, guard: &Guard, depth: usize) {
        // If retry depth is exceeded, defer deallocation of the current bucket and return.
        if depth >= B {
            unsafe {
                guard.defer_destroy(new_bucket);
            }
            return;
        }

        for hasher in &self.cuckoo.hashers {
            let mut hasher = hasher.clone();
            unsafe {
                new_bucket.deref().key.hash(&mut hasher);
            }
            let hash = hasher.finish() as usize;
            for bucket in self.cuckoo.buckets[hash % M].iter() {
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
        let mut hasher = self.cuckoo.hashers[0].clone();
        unsafe {
            new_bucket.deref().key.hash(&mut hasher);
        }
        let hash = hasher.finish() as usize;
        let bucket = &self.cuckoo.buckets[hash % M][B - 1 - depth];

        let old = bucket.swap(new_bucket, atomic::Ordering::Relaxed, &guard);

        self.insert_internal(old, guard, depth + 1);
    }
}

impl<'c, K, V, H, const M: usize, const N: usize, const B: usize> Drop
    for WriteGuard<'c, K, V, H, M, N, B>
{
    fn drop(&mut self) {
        self.cuckoo.locked.store(false, atomic::Ordering::Release)
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
    fn test_lock() {
        let hashmap = HashMap::<i32, i32>::std();

        let lock = hashmap.try_lock();

        assert!(lock.is_ok());

        assert!(hashmap.try_lock().is_err());
    }

    #[test]
    fn test_insert_get() {
        let hashmap = HashMap::<String, ()>::std();

        let lock = hashmap.try_lock().unwrap();

        lock.insert("Hello There!".to_string(), ());

        assert!(hashmap.get(&"Hello There!".to_string()).is_some());
    }

    #[test]
    fn test_insert_updates_equal_key() {
        let hashmap = HashMap::<String, i32>::std();

        let lock = hashmap.try_lock().unwrap();

        lock.insert("Hello There!".to_string(), 0);
        lock.insert("Hello There!".to_string(), 1);

        assert!(hashmap.get(&"Hello There!".to_string()).is_some());
        assert_eq!(1, hashmap.get(&"Hello There!".to_string()).unwrap());
    }

    #[test]
    fn stops_retrying_on_max_depth() {
        let hashmap = HashMap::<String, i32, _, 4, 2, 1>::std();

        let lock = hashmap.try_lock().unwrap();

        lock.insert("1".to_string(), 0);
        lock.insert("2".to_string(), 1);
        lock.insert("3".to_string(), 2);
        lock.insert("4".to_string(), 3);

        println!("{}", hashmap.get(&"1".to_string()).is_some());
        println!("{}", hashmap.get(&"2".to_string()).is_some());
        println!("{}", hashmap.get(&"3".to_string()).is_some());
        println!("{}", hashmap.get(&"4".to_string()).is_some());
    }
}
