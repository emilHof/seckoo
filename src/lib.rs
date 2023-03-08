use std::hash::BuildHasher;
use std::mem::MaybeUninit;
use std::ptr::write_bytes;
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
    buckets: [[Bucket<K, V>; B]; M],
    locked: atomic::AtomicBool,
}

#[derive(Debug)]
struct Bucket<K, V> {
    key: K,
    val: V,
    version: atomic::AtomicUsize,
}

impl<K, V> Bucket<K, V> {
    fn is_empty(&self) -> bool {
        self.version.load(atomic::Ordering::Acquire) == 0
    }
    /// Increments the sequence at the current index by 1, making it odd, prohibiting reads.
    #[inline]
    fn start_write(&self) {
        let version = self.version.fetch_add(1, atomic::Ordering::Relaxed);
        assert!(version & 1 == 0);
    }

    /// Increments the sequence at the current index by 1, making it even and allowing reads.
    #[inline]
    fn end_write(&self) {
        let version = self.version.fetch_add(1, atomic::Ordering::Relaxed);
        assert!(version & 1 == 1);
    }
}

impl<K, V, const M: usize, const N: usize, const B: usize>
    CuckooHash<K, V, std::collections::hash_map::DefaultHasher, M, N, B>
{
    // #[cfg(feature = "nightly")]
    pub fn std() -> Self {
        let buckets: [[Bucket<K, V>; B]; M] = unsafe {
            let mut buckets: [[MaybeUninit<Bucket<K, V>>; B]; M] =
                MaybeUninit::uninit().assume_init();
            write_bytes(&mut buckets, 0, 1);

            // This workaround is currently necessary, as `core::mem::transmute()` is not available
            // for arrays whose length is specified by Const Generics.
            let init = core::ptr::read(
                (&buckets as *const [[MaybeUninit<Bucket<K, V>>; B]; M])
                    .cast::<[[Bucket<K, V>; B]; M]>(),
            );
            core::mem::forget(buckets);
            init
        };

        CuckooHash {
            hashers: [(); N].map(|_| std::collections::hash_map::RandomState::new().build_hasher()),
            buckets,
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
    K: Copy + std::hash::Hash + std::cmp::Eq,
    V: Copy,
{
    pub fn get(&self, k: &K) -> Option<V> {
        for hasher in &self.hashers {
            let mut hasher = hasher.clone();
            k.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            if let Some(val) = self.search_buckets(hash % M, k) {
                return Some(val);
            }
        }
        None
    }

    // Searches a bucket for a key. Rereads when version after read is wrong, yet does not re-read
    // prior buckets.
    fn search_buckets(&self, i: usize, k: &K) -> Option<V> {
        for bucket in &self.buckets[i] {
            loop {
                let v1 = bucket.version.load(atomic::Ordering::Acquire);
                let t_k = unsafe { std::ptr::read_volatile(&bucket.key as *const K as *mut K) };
                let v = unsafe { std::ptr::read_volatile(&bucket.val as *const V as *mut V) };
                let v2 = bucket.version.load(atomic::Ordering::Relaxed);

                if v1 != v2 {
                    continue;
                }

                if k.eq(&t_k) {
                    return Some(v);
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
    K: Copy + std::hash::Hash + std::cmp::Eq,
    V: Copy,
{
    pub fn insert(&self, k: K, v: V) {
        for hasher in &self.cuckoo.hashers {
            let mut hasher = hasher.clone();
            k.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            for bucket in self.cuckoo.buckets[hash % M].iter() {
                // If an empty bucket is found, we insert here.
                if bucket.is_empty() {
                    bucket.start_write();
                    unsafe { std::ptr::write_volatile(&bucket.key as *const K as *mut _, k) }
                    unsafe { std::ptr::write_volatile(&bucket.val as *const V as *mut _, v) }
                    bucket.end_write();

                    return;
                }

                // If the key is the same, we insert the new value.
                if k.eq(&bucket.key) {
                    bucket.start_write();
                    unsafe { std::ptr::write_volatile(&bucket.val as *const V as *mut _, v) }
                    bucket.end_write();

                    return;
                }
            }
        }

        // No empty location was found, so we insert it in our first choice bucket and re-insert
        // the old value.
        //
        let mut hasher = self.cuckoo.hashers[0].clone();
        k.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        let bucket = &self.cuckoo.buckets[hash % M][0];
        let old_key = bucket.key;
        let old_val = bucket.val;

        bucket.start_write();
        unsafe { std::ptr::write_volatile(&bucket.key as *const K as *mut _, k) }
        unsafe { std::ptr::write_volatile(&bucket.val as *const V as *mut _, v) }
        bucket.start_write();

        self.insert(old_key, old_val);
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
        let hashmap = HashMap::<[u8; 12], ()>::std();

        let lock = hashmap.try_lock().unwrap();

        lock.insert(*b"Hello There!", ());

        assert!(hashmap.get(b"Hello There!").is_some());
    }
}
