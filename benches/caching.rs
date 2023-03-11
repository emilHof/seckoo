use std::collections::hash_map::RandomState;
use std::hash::Hash;
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

pub fn compare_caches(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Bench Throughput With Variable Threads"));
    for t in THREADS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Seckoo {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_get_random(t, || SeckooCache::<_, _, _, 4096>::std())),
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("Moka {t} Thread(s)")),
            &t,
            |b, &t| b.iter(|| insert_get_random(t, || MokaCache::new(4096))),
        );
    }
}

criterion_group!(benches, compare_caches);
criterion_main!(benches);
