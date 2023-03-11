use crossbeam_epoch::{Atomic, Collector, CompareExchangeError, Guard, Owned, Shared};
use std::collections::hash_map;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::align_of;
use std::sync::atomic;

const DEFAULT_LENGTH: usize = 4096;

#[inline]
fn low_bits<T>() -> usize {
    (1 << align_of::<T>().trailing_zeros()) - 1
}

pub type HashMap<K, V, H = hash_map::RandomState, const N: usize = DEFAULT_LENGTH> =
    CuckooHash<K, V, H, N>;

pub struct CuckooHash<K, V, H, const N: usize> {
    hash_builders: [H; 2],
    buckets: [Atomic<Bucket<K, V>>; N],
    collector: Collector,
}

impl<K: Debug, V: Debug, H, const N: usize> Debug for CuckooHash<K, V, H, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.collector.register().pin();

        let mut list = f.debug_list();
        for bucket in &self.buckets {
            let ptr = bucket.load(atomic::Ordering::Relaxed, &guard);

            if ptr.is_null() {
                continue;
            }

            unsafe {
                list.entry(ptr.deref());
            }
        }
        list.finish()
    }
}

unsafe impl<K, V, H, const N: usize> Send for CuckooHash<K, V, H, N>
where
    K: Send,
    V: Send,
{
}

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

impl<K, V, const N: usize> CuckooHash<K, V, hash_map::RandomState, N> {
    pub fn std() -> Self {
        // Ensure `Bucket`s leave enough space to store tag.
        assert!(align_of::<Bucket<K, V>>() >= 2);

        let buckets = [(); N].map(|_| Atomic::<Bucket<K, V>>::null());

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

impl<K, V, H, const N: usize> CuckooHash<K, V, H, N>
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
        let hash2 = h2.finish() as usize;

        // XOR the first index with the tag to get the second index and truncate.
        let second_index = first_index ^ hash2 as usize;

        let handle = self.collector.register();
        let guard = handle.pin();
        guard.flush();

        for i in [first_index, second_index] {
            let ptr = &self.buckets[i % N].load(atomic::Ordering::Relaxed, &guard);

            if ptr.is_null() {
                break;
            }

            unsafe {
                if ptr.deref().key.eq(k) {
                    return Some((&ptr.deref().val).clone());
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
            /*
            println!("in this loop with index: {}", index % N);

            println!("comparing if equal");
            */
            // Keys are not equal.
            unsafe {
                if current.deref().key.ne(&new_bucket.deref().key) {
                    break;
                }
            }
            /*
            println!("are equal!");
            */

            /*
            println!(
                "comparing tags c: {} and n: {}",
                current.tag(),
                new_bucket.tag()
            );
            */
            // Same key, but the other bucket is has a tag of one while we have a tag of 0.
            if current.tag() & !new_bucket.tag() == 1 {
                /*
                println!("existing value is more current!");
                */
                unsafe {
                    guard.defer_destroy(new_bucket);
                    guard.flush();
                }
                return;
            }

            /*
            println!("trying to swap new value and replace old!");
            */

            // Try swapping out equal key.
            current = match bucket.compare_exchange(
                current,
                new_bucket,
                atomic::Ordering::Relaxed,
                atomic::Ordering::Relaxed,
                &guard,
            ) {
                Ok(_) => {
                    /*
                    println!("success!");
                    */
                    // Succeeded to swap with an equal key.
                    // We defer deallocation.
                    unsafe {
                        guard.defer_destroy(current);
                        guard.flush();
                    }
                    return;
                }
                Err(CompareExchangeError { current, new: _ }) => current,
            }
            /*
            println!("failed!");
            */
        }

        // No empty location was found, so we insert depending on recursion depth into this bucket and re-insert
        // the replaced bucket.
        let mut h2 = self.hash_builders[1].build_hasher();

        let old = bucket.swap(new_bucket, atomic::Ordering::Relaxed, &guard);

        // Set the tag to 0 to signal that it has been moved.
        let old = old.with_tag(0);

        // XOR the index with the next_tag to get the other index for this key and truncate.
        unsafe {
            old.deref().key.hash(&mut h2);
        }
        let hash2 = h2.finish();
        let next_index = index ^ hash2 as usize;

        self.insert_internal(next_index, old, guard, depth + 1);
    }

    pub fn iter(&self) -> Iter<'_, K, V, H, N> {
        Iter {
            cuckoo: &self,
            index: 0,
        }
    }
}

pub struct Iter<'c, K, V, H, const N: usize> {
    cuckoo: &'c CuckooHash<K, V, H, N>,
    index: usize,
}

impl<'a, K, V, H, const N: usize> Iterator for Iter<'a, K, V, H, N>
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
    K: Hash + std::cmp::Eq,
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
        for bucket in &self.buckets {
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
        println!("returning 1");
        assert_eq!(1, hashmap.get(&"Hello There!".to_string()).unwrap());
        println!("returning 2");
    }

    #[test]
    fn test_insert_will_remove_oldest() {
        'test: loop {
            let hashmap = HashMap::<i32, i32>::std();

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
        let hashmap = HashMap::<String, i32>::std();

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

        println!("iter test");

        for value in &hashmap {
            println!("{value}")
        }
    }

    #[test]
    fn test_threads() {
        use std::thread;

        let hashmap = HashMap::<[u8; 12], [u8; 12]>::std();

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

        let hashmap = HashMap::<i32, CountOnDrop, _, 4>::std();

        hashmap.insert(0, CountOnDrop::new(0, counter.clone()));
        hashmap.insert(0, CountOnDrop::new(1, counter.clone()));
        assert_eq!(hashmap.get(&0).unwrap().key, 1);

        drop(hashmap);
        assert_eq!(counter.load(atomic::Ordering::Relaxed), 3);
    }
}
