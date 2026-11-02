//! Random numbers, seeded and reproducible.
//!
//! `Mt19937` reproduces CPython's `random.Random(seed).random()` (hunch's audit draws) and numpy's legacy
//! `RandomState(seed).shuffle` (scikit-learn's `KFold(shuffle=True, random_state=seed)`), so understudy's folds and
//! audits are hunch's. `Xoshiro` drives everything understudy adds (initial weights, minibatch order).

use serde::{Deserialize, Serialize};

const N: usize = 624;
const M: usize = 397;

/// The Mersenne Twister, as CPython and numpy implement it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Mt19937 {
    mt: Vec<u32>,
    idx: usize,
}

impl Mt19937 {
    /// `init_genrand(s)`: numpy's `RandomState(s)` for an integer seed.
    pub fn numpy(seed: u32) -> Self {
        let mut mt = vec![0u32; N];
        mt[0] = seed;
        for i in 1..N {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: N }
    }

    /// `init_by_array(key)`: CPython's `random.Random(seed)` for a non-negative integer seed (its 32-bit words).
    pub fn python(seed: u64) -> Self {
        let key: Vec<u32> = if seed >> 32 == 0 { vec![seed as u32] } else { vec![seed as u32, (seed >> 32) as u32] };
        let mut s = Self::numpy(19650218);
        let mt = &mut s.mt;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        s
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            for k in 0..N {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % N] & 0x7fff_ffff);
                let mut v = self.mt[(k + M) % N] ^ (y >> 1);
                if y & 1 == 1 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.random()`: 53 random bits in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// numpy's legacy `random_interval(max)`: uniform in 0..=max by masked rejection.
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        if max <= 0xffff_ffff {
            loop {
                let v = (self.next_u32() as u64) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = (((self.next_u32() as u64) << 32) | self.next_u32() as u64) & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// numpy's legacy `RandomState.shuffle` of a 1-d array.
    pub fn shuffle<T>(&mut self, a: &mut [T]) {
        for i in (1..a.len()).rev() {
            let j = self.interval(i as u64) as usize;
            a.swap(i, j);
        }
    }
}

/// xoshiro256**, seeded by splitmix64.
#[derive(Clone, Debug)]
pub struct Xoshiro([u64; 4]);

impl Xoshiro {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for x in &mut s {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *x = v ^ (v >> 31);
        }
        Xoshiro(s)
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    pub fn shuffle<T>(&mut self, a: &mut [T]) {
        for i in (1..a.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            a.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_random() {
        // python -c "import random; r = random.Random(0); print(r.random(), r.random())"
        let mut r = Mt19937::python(0);
        assert_eq!(r.random(), 0.8444218515250481);
        assert_eq!(r.random(), 0.7579544029403025);
    }

    #[test]
    fn numpy_shuffle() {
        // python -c "import numpy as np; a = np.arange(10); np.random.RandomState(0).shuffle(a); print(a)"
        let mut a: Vec<usize> = (0..10).collect();
        Mt19937::numpy(0).shuffle(&mut a);
        assert_eq!(a, vec![2, 8, 4, 9, 1, 6, 7, 3, 0, 5]);
    }
}
