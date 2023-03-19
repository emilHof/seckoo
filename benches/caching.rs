use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;
use std::thread;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{thread_rng, Rng};

use moka::sync::Cache as MokaCache;
use seckoo::HashMap as SeckooCache;

const THREADS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

pub trait Cache: Send + Sync {
    type Key;
    type Value;

    fn insert(&self, k: Self::Key, v: Self::Value);
    fn get(&self, k: &Self::Key) -> Option<Self::Value>;
}

impl<T> Cache for &T
where
    T: Cache,
{
    type Key = T::Key;
    type Value = T::Value;

    fn get(&self, k: &Self::Key) -> Option<Self::Value> {
        (*self).get(k)
    }

    fn insert(&self, k: Self::Key, v: Self::Value) {
        (*self).insert(k, v)
    }
}

impl<K, V, const N: usize> Cache for SeckooCache<K, V, RandomState, N>
where
    K: Hash + Eq + Send + Sync,
    V: Clone + Send + Sync,
{
    type Key = K;
    type Value = V;

    fn insert(&self, k: Self::Key, v: Self::Value) {
        SeckooCache::insert(&self, k, v);
    }

    fn get(&self, k: &Self::Key) -> Option<Self::Value> {
        SeckooCache::get(&self, k)
    }
}

impl<K, V> Cache for MokaCache<K, V, RandomState>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    type Key = K;
    type Value = V;

    fn insert(&self, k: Self::Key, v: Self::Value) {
        MokaCache::insert(&self, k, v);
    }

    fn get(&self, k: &Self::Key) -> Option<Self::Value> {
        MokaCache::get(&self, k)
    }
}

impl<K, V> Cache for Mutex<HashMap<K, V>>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    type Key = K;
    type Value = V;

    fn insert(&self, k: Self::Key, v: Self::Value) {
        HashMap::insert(&mut self.lock().unwrap(), k, v);
    }

    fn get(&self, k: &Self::Key) -> Option<Self::Value> {
        HashMap::get(&self.lock().unwrap(), k).cloned()
    }
}

pub fn insert_get_random<C: Cache<Key = i32, Value = i32> + Send + Sync, F: Fn() -> C>(
    t: usize,
    f: F,
) {
    let cache = f();

    thread::scope(|s| {
        for _ in 0..t {
            let cache = &cache;
            s.spawn(move || {
                let mut rng = thread_rng();

                for i in 0..black_box(256) {
                    let num = rng.gen_range(0..4096);

                    if rng.gen::<u8>() % 2 == 1 {
                        cache.insert(black_box(num), i);
                    } else {
                        cache.get(&black_box(num));
                    }
                }
            });
        }
    })
}

pub fn insert_random<C: Cache<Key = i32, Value = i32> + Send + Sync, F: Fn() -> C>(t: usize, f: F) {
    let cache = f();

    thread::scope(|s| {
        for _ in 0..t {
            let cache = &cache;
            s.spawn(move || {
                let mut rng = thread_rng();

                for i in 0..black_box(256) {
                    let num = rng.gen_range(0..4096);

                    cache.insert(black_box(num), i);
                }
            });
        }
    })
}

pub fn get_random<C: Cache<Key = i32, Value = i32> + Send + Sync, F: Fn() -> C>(t: usize, f: F) {
    let cache = f();

    let mut rng = thread_rng();

    for i in 0..black_box(256) {
        let num = rng.gen_range(0..4096);
        cache.insert(black_box(num), i);
    }

    thread::scope(|s| {
        for _ in 0..t {
            let cache = &cache;
            s.spawn(move || {
                let mut rng = thread_rng();

                for _ in 0..black_box(256) {
                    let num = rng.gen_range(0..4096);

                    cache.get(&num);
                }
            });
        }
    })
}

pub fn compare_insert_get(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Compare Insert & Get"));
    for t in THREADS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Seckoo {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_get_random(t, || SeckooCache::<_, _, _, 8192>::std())),
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Std-Mutex {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_get_random(t, || Mutex::new(HashMap::new()))),
        );
        /*
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Moka {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_get_random(t, || MokaCache::new(8192))),
        );
        */
    }
    group.finish();
}

pub fn compare_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Compare Insert"));
    for t in THREADS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Seckoo {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_random(t, || SeckooCache::<_, _, _, 8192>::std())),
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Std-Mutex {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_random(t, || Mutex::new(HashMap::new()))),
        );
        /*
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Moka {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_random(t, || MokaCache::new(8192))),
        );
        */
    }
    group.finish();
}

pub fn compare_get(c: &mut Criterion) {
    let seckoo_cache = SeckooCache::<_, _, _, 8192>::std();
    let moka_cache = MokaCache::new(8192);
    let std_mutex = Mutex::new(HashMap::new());

    let mut rng = thread_rng();

    for i in 0..black_box(1024) {
        let num = rng.gen_range(0..4096);
        seckoo_cache.insert(black_box(num), i);
        moka_cache.insert(black_box(num), i);
    }

    let mut group = c.benchmark_group(format!("Compare Get"));
    for t in THREADS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Seckoo {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| get_random(t, || &seckoo_cache)),
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Std-Mutex {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| get_random(t, || &std_mutex)),
        );
        /*
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Moka {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| get_random(t, || &moka_cache)),
        );
        */
    }
    group.finish();
}

criterion_group!(benches, compare_insert_get, compare_insert, compare_get);
criterion_main!(benches);
